#!/usr/bin/env python3
"""
Memory-efficient UNILID training on FineWeb2 parquet data.

Reads flat parquet files from a directory (one per language, e.g. aai_Latn.parquet),
trains a shared base tokenizer on sampled data, then adapts per-language probability
distributions in batches to limit disk usage.

Designed for SLURM clusters — see train_cluster.sh.

Usage:
    python train_cluster.py --parquet-dir /path/to/fineweb2/
    python train_cluster.py --parquet-dir /path/to/fineweb2/ --vocab-size 100000 --lang-batch-size 10
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer
from unilid.trainers.language_specific_trainer import LanguageSpecificUnigramLMTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ───────────────────────── parquet helpers ─────────────────────────────


def discover_parquet_files(parquet_dir):
    """Find all .parquet files and extract language codes from filenames.

    Returns dict {lang_code: parquet_path}.
    """
    parquet_dir = Path(parquet_dir)
    lang_files = {}
    for p in sorted(parquet_dir.glob("*.parquet")):
        lang_code = p.stem  # e.g. "aai_Latn" from "aai_Latn.parquet"
        lang_files[lang_code] = str(p)
    return lang_files


def sample_from_parquet(parquet_path, max_lines, seed=42):
    """Read a parquet file and return up to max_lines text samples.

    Expects a 'text' column.
    """
    table = pq.read_table(parquet_path, columns=["text"])
    texts = table.column("text").to_pylist()
    del table

    if len(texts) > max_lines:
        rng = random.Random(seed)
        texts = rng.sample(texts, max_lines)

    # Filter empty/whitespace-only
    texts = [t for t in texts if t and t.strip()]
    return texts


def write_texts_to_file(texts, output_path):
    """Write a list of text strings to a file, one per line."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in texts:
            # Replace newlines within a text to keep one-line-per-sample
            f.write(t.replace("\n", " ").strip() + "\n")
    return len(texts)


def parquet_to_txt(parquet_path, output_path):
    """Convert a full parquet file to a text file (one line per row).

    Streams in batches to limit memory.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet_path)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in pf.iter_batches(batch_size=10_000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if text and text.strip():
                    f.write(text.replace("\n", " ").strip() + "\n")
                    count += 1
    return count


# ───────────────────────────── main ──────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient UNILID training on FineWeb2 parquet data",
    )

    # Input
    parser.add_argument("--parquet-dir", type=str, required=True,
                        help="Directory containing {lang}.parquet files")

    # Training
    tr = parser.add_argument_group("Training")
    tr.add_argument("--vocab-size", type=int, default=100_000,
                    help="Vocabulary size (default: 100000)")
    tr.add_argument("--base-samples", type=int, default=10_000,
                    help="Lines to sample per language for base tokenizer (default: 10000)")
    tr.add_argument("--base-training-method", type=str, default="hf",
                    choices=["hf", "bpe", "soft", "hard"],
                    help="Base tokenizer training method (default: hf)")
    tr.add_argument("--per-lang-method", type=str, default="sp",
                    choices=["sp", "soft", "hard"],
                    help="Per-language probability estimation (default: sp)")
    tr.add_argument("--byte-level", dest="byte_level",
                    action="store_true", default=True,
                    help="Use byte-level tokenization (default)")
    tr.add_argument("--char-level", dest="byte_level",
                    action="store_false",
                    help="Use character-level tokenization")
    tr.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")

    # Orchestration
    orch = parser.add_argument_group("Orchestration")
    orch.add_argument("--results-dir", type=str, default=None,
                      help="Output directory (default: results_{K}k)")
    orch.add_argument("--lang-batch-size", type=int, default=10,
                      help="Languages per batch for per-lang training (default: 10)")
    orch.add_argument("--reuse-base", dest="reuse_base",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Skip base training if tokenizer exists (default: True)")
    orch.add_argument("--skip-existing-langs", dest="skip_existing_langs",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Skip already-trained languages (default: True)")
    orch.add_argument("--reuse-sampled", dest="reuse_sampled",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Reuse existing sampled corpus files (default: True)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Optional .unilid output path for packed model")

    args = parser.parse_args()

    # ── set random seed ──
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── resolve paths ──
    vocab_size = args.vocab_size
    results_dir = args.results_dir or f"results_{vocab_size // 1000}k"
    os.makedirs(results_dir, exist_ok=True)

    tokenizers_dir = os.path.join(results_dir, "tokenizers")
    os.makedirs(tokenizers_dir, exist_ok=True)

    sampled_dir = os.path.join(results_dir, "corpus_base_sampled")
    tmp_dir = os.path.join(results_dir, "corpus_tmp")

    base_tok_path = os.path.join(tokenizers_dir, "langspec_base_tokenizer.json")

    logger.info("=" * 60)
    logger.info(
        "TRAINING starting | vocab=%d  base=%s  lang=%s  byte_level=%s  batch=%d",
        vocab_size, args.base_training_method, args.per_lang_method,
        args.byte_level, args.lang_batch_size,
    )
    logger.info("Results dir: %s", results_dir)
    logger.info("=" * 60)

    start_time = time.time()

    # ──────────────────────────────────────────────────────────────────
    # Phase 1: Discover parquet files and sample for base tokenizer
    # ──────────────────────────────────────────────────────────────────

    logger.info("Phase 1: Discovering parquet files and sampling for base tokenizer")

    lang_parquets = discover_parquet_files(args.parquet_dir)
    if not lang_parquets:
        logger.error("No .parquet files found in %s", args.parquet_dir)
        sys.exit(1)

    all_languages = sorted(lang_parquets.keys())
    logger.info("Found %d languages in %s", len(all_languages), args.parquet_dir)

    # Sample from each parquet for base tokenizer training
    os.makedirs(sampled_dir, exist_ok=True)
    sampled_corpus = {}
    sampled_count = 0

    for i, lang in enumerate(all_languages):
        dst = os.path.join(sampled_dir, f"{lang}_train.sampled.txt")

        # Reuse existing sampled file if it exists and flag is set
        if args.reuse_sampled and os.path.isfile(dst):
            sampled_corpus[lang] = dst
            continue

        texts = sample_from_parquet(lang_parquets[lang], args.base_samples, seed=args.seed)
        if not texts:
            logger.warning("No text found in %s, skipping", lang_parquets[lang])
            continue

        n = write_texts_to_file(texts, dst)
        sampled_corpus[lang] = dst
        sampled_count += n
        del texts

        if (i + 1) % 100 == 0:
            logger.info("  Sampled %d/%d languages", i + 1, len(all_languages))

    logger.info(
        "Phase 1 complete: %d languages sampled to %s (%.1fs)",
        len(sampled_corpus), sampled_dir, time.time() - start_time,
    )

    # Update all_languages to only those with valid sampled data
    all_languages = sorted(sampled_corpus.keys())

    # ──────────────────────────────────────────────────────────────────
    # Phase 2: Train base tokenizer
    # ──────────────────────────────────────────────────────────────────

    phase2_start = time.time()
    logger.info("Phase 2: Training base tokenizer")

    base_reused = False
    base_training_time = 0.0

    if os.path.exists(base_tok_path) and args.reuse_base:
        logger.info("Reusing existing base tokenizer at %s", base_tok_path)
        base_reused = True
    else:
        base_tok = StandardUnigramLMTokenizer(
            vocab_size=vocab_size,
            em_mode=args.base_training_method,
            byte_level=args.byte_level,
        )
        train_files = list(sampled_corpus.values())
        logger.info(
            "Training base tokenizer on %d sampled files (<=%d lines/lang)",
            len(train_files), args.base_samples,
        )
        base_t0 = time.time()
        base_tok.train(train_files, output_path=base_tok_path)
        base_training_time = time.time() - base_t0
        del base_tok
        gc.collect()

    logger.info(
        "Phase 2 complete: base tokenizer at %s (%.1fs)",
        base_tok_path, time.time() - phase2_start,
    )

    # ──────────────────────────────────────────────────────────────────
    # Phase 3: Batched per-language training
    # ──────────────────────────────────────────────────────────────────

    phase3_start = time.time()
    logger.info("Phase 3: Batched per-language training")

    # Determine expected tokenizer path for a language
    def _expected_path(lang):
        em_mode = args.per_lang_method if args.per_lang_method != "sp" else "sp"
        return os.path.join(
            tokenizers_dir,
            f"langspec_{em_mode}_{lang}.tokenizer.json",
        )

    # Filter out already-trained languages
    langs_to_train = all_languages
    if args.skip_existing_langs:
        remaining = [l for l in langs_to_train if not os.path.isfile(_expected_path(l))]
        skipped = len(langs_to_train) - len(remaining)
        if skipped > 0:
            logger.info(
                "%d existing lang tokenizers found; %d remaining",
                skipped, len(remaining),
            )
        langs_to_train = remaining

    total_langs = len(langs_to_train)
    batch_size = max(1, args.lang_batch_size)
    num_batches = math.ceil(total_langs / batch_size) if total_langs else 0

    logger.info(
        "Training %d languages in %d batches of up to %d",
        total_langs, num_batches, batch_size,
    )

    os.makedirs(tmp_dir, exist_ok=True)
    tokenizer_paths = {}

    for bidx in range(num_batches):
        batch_start = bidx * batch_size
        batch_langs = langs_to_train[batch_start : batch_start + batch_size]
        if not batch_langs:
            break

        logger.info(
            "Batch %d/%d: %d languages [%s .. %s]",
            bidx + 1, num_batches, len(batch_langs),
            batch_langs[0], batch_langs[-1],
        )

        # Step 1: Convert parquet → tmp txt for this batch
        batch_corpus = {}
        batch_tmp_files = []
        for lang in batch_langs:
            tmp_path = os.path.join(tmp_dir, f"{lang}_train.txt")
            parquet_path = lang_parquets[lang]

            logger.info("  Converting %s -> %s", parquet_path, tmp_path)
            n = parquet_to_txt(parquet_path, tmp_path)
            logger.info("  %s: %d lines", lang, n)

            batch_corpus[lang] = tmp_path
            batch_tmp_files.append(tmp_path)

        # Step 2: Train per-language tokenizers for this batch
        lang_em_mode = args.per_lang_method if args.per_lang_method != "sp" else "soft"
        tok = LanguageSpecificUnigramLMTokenizer(
            vocab_size=vocab_size,
            reestimation_em_mode=lang_em_mode,
            byte_level=args.byte_level,
        )
        try:
            batch_paths = tok.train(
                corpus_info=batch_corpus,
                base_tokenizer_path=base_tok_path,
                output_dir=tokenizers_dir,
                tokenizer_path_format=None,
                load_base_if_exists=True,
                load_tokenizers_if_exists=args.skip_existing_langs,
                use_sentencepiece=(args.per_lang_method == "sp"),
            )
        except subprocess.CalledProcessError as e:
            logger.error("spm_train failed with exit code %d", e.returncode)
            logger.error("spm_train STDERR:\n%s", e.stderr)
            logger.error("spm_train STDOUT:\n%s", e.stdout)
            raise

        if batch_paths:
            tokenizer_paths.update(batch_paths)

        # Step 3: Clean up temporary txt files for this batch
        for tmp_path in batch_tmp_files:
            try:
                os.remove(tmp_path)
                logger.debug("Deleted tmp file: %s", tmp_path)
            except OSError as e:
                logger.warning("Failed to delete %s: %s", tmp_path, e)

        del tok, batch_paths, batch_corpus, batch_tmp_files
        gc.collect()

        logger.info(
            "Batch %d/%d complete (%.1fs elapsed)",
            bidx + 1, num_batches, time.time() - phase3_start,
        )

    # Clean up tmp dir if empty
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    lang_training_time = time.time() - phase3_start

    logger.info(
        "Phase 3 complete: %d languages trained (%.1fs)",
        total_langs, lang_training_time,
    )

    # ──────────────────────────────────────────────────────────────────
    # Phase 4: Convert to .unilid (optional)
    # ──────────────────────────────────────────────────────────────────

    if args.output:
        logger.info("Phase 4: Packing to .unilid format")
        from unilid.model_io import save_unilid
        output_path = save_unilid(Path(results_dir), Path(args.output))
        logger.info("Saved packed model to %s", output_path)

    # ──────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────

    total_training_time = time.time() - start_time

    summary = {
        "command": " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_completed": True,

        "source": {
            "format": "parquet",
            "parquet_dir": os.path.abspath(args.parquet_dir),
            "num_languages_discovered": len(lang_parquets),
            "num_languages_with_data": len(all_languages),
        },

        "method": {
            "vocab_size": vocab_size,
            "base_training_method": args.base_training_method,
            "per_lang_method": args.per_lang_method,
            "byte_level": args.byte_level,
            "seed": args.seed,
            "base_samples_per_lang": args.base_samples,
            "lang_batch_size": args.lang_batch_size,
        },

        "timing": {
            "total_seconds": round(total_training_time, 2),
            "base_tokenizer_seconds": round(base_training_time, 2),
            "language_tokenizers_seconds": round(lang_training_time, 2),
            "base_tokenizer_reused": base_reused,
        },

        "output": {
            "results_dir": os.path.abspath(results_dir),
            "tokenizers_dir": os.path.abspath(tokenizers_dir),
            "base_tokenizer": os.path.abspath(base_tok_path),
            "num_languages_trained_this_run": total_langs,
            "packed_model": os.path.abspath(args.output) if args.output else None,
        },

        "languages": all_languages,
    }

    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(
        "TRAINING COMPLETE | %dK vocab | %d languages trained this run | "
        "%d total languages | %.1fs",
        vocab_size // 1000, total_langs, len(all_languages), total_training_time,
    )
    logger.info("Results: %s/", results_dir)
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
