#!/usr/bin/env python3
"""
Unified UNILID Training Script

Replaces the three nearly-identical training scripts (wili, tatoeba, glotlid)
with one script. Pass exactly one input flag:

  --fasttext FILE     __label__<lang> <text>  (one per line)
  --wili-dir DIR      directory with x_train.txt + y_train.txt
  --tsv FILE          tab-separated: id \\t lang \\t text  (Tatoeba sentences.csv)
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer
from unilid.trainers.language_specific_trainer import LanguageSpecificUnigramLMTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ───────────────────── initial-vocab helpers ──────────────────────────

def _is_hf_tokenizer(path):
    """Check if a file is an HF tokenizer JSON (has model.vocab)."""
    if not path.endswith('.json'):
        return False
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return isinstance(data, dict) and 'model' in data and 'vocab' in data.get('model', {})


def _convert_to_unigram_base(source_path, output_path):
    """Convert any HF tokenizer to a Unigram base tokenizer with uniform log probs.

    Preserves the source's pretokenizer, normalizer, and decoder.
    """
    from tokenizers import Tokenizer
    from unilid.tokenizer_builder import _build_unigramlm_hf_tokenizer_from_lprobs
    from unilid.constants import SPECIAL_TOKENS, MIN_TOKEN_LOG_PROB

    source = Tokenizer.from_file(source_path)
    vocab = source.get_vocab()

    special_set = set(SPECIAL_TOKENS.values())
    unk_token = SPECIAL_TOKENS["unk_token"]

    # Uniform log probs for all non-special tokens
    non_special = [t for t in vocab if t not in special_set]
    uniform_lp = np.log(1.0 / len(non_special)) if non_special else MIN_TOKEN_LOG_PROB

    tuples = []
    for token in sorted(vocab, key=lambda t: vocab[t]):
        if token in special_set:
            tuples.append((token, 0.0))
        else:
            tuples.append((token, uniform_lp))

    unk_id = vocab.get(unk_token, 0)

    tok = _build_unigramlm_hf_tokenizer_from_lprobs(
        tuples, unk_id,
        pretokenizer=source.pre_tokenizer,
        decoder=source.decoder,
    )
    if source.normalizer:
        tok.normalizer = source.normalizer

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(output_path)
    logger.info("Saved Unigram base tokenizer (%d tokens) to %s", len(vocab), output_path)


# ───────────────────────────── data loaders ─────────────────────────────


def load_fasttext(path, max_samples=None):
    """Parse __label__<lang> <text> lines."""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.rstrip("\n")
            if not line or not line.startswith("__label__"):
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            labels.append(parts[0].replace("__label__", ""))
            texts.append(parts[1])
            if (i + 1) % 100_000 == 0:
                logger.info("Loaded %d lines from %s", i + 1, path)
    logger.info("Loaded %d samples from %s (fasttext)", len(texts), path)
    return texts, labels


def load_wili(data_dir, max_samples=None):
    """Parse x_train.txt + y_train.txt side-by-side."""
    x_path = os.path.join(data_dir, "x_train.txt")
    y_path = os.path.join(data_dir, "y_train.txt")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"Expected x_train.txt and y_train.txt under {data_dir}"
        )
    texts, labels = [], []
    with open(x_path, "r", encoding="utf-8", errors="ignore") as fx, \
         open(y_path, "r", encoding="utf-8", errors="ignore") as fy:
        for i, (x_line, y_line) in enumerate(zip(fx, fy)):
            if max_samples is not None and i >= max_samples:
                break
            x_line = x_line.rstrip("\n")
            y_line = y_line.rstrip("\n")
            if not x_line or not y_line:
                continue
            texts.append(x_line)
            labels.append(y_line)
    logger.info("Loaded %d samples from %s (wili)", len(texts), data_dir)
    return texts, labels


def load_tsv(path, max_samples=None):
    """Parse tab-separated: id \\t lang \\t text (Tatoeba sentences.csv)."""
    texts, labels = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            labels.append(parts[1])
            texts.append(parts[2])
            if (i + 1) % 200_000 == 0:
                logger.info("Parsed %d TSV rows from %s", i + 1, path)
    logger.info("Loaded %d samples from %s (tsv)", len(texts), path)
    return texts, labels



# ───────────────────────── corpus preparation ───────────────────────────


def prepare_corpus(texts, labels, corpus_dir):
    """Group by language, write {lang}_train.txt files.

    Returns (corpus_info, lang_sample_counts, total_samples).
    """
    lang_texts = defaultdict(list)
    for t, l in zip(texts, labels):
        lang_texts[l].append(t)

    cdir = Path(corpus_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    corpus_info = {}
    lang_sample_counts = {}
    total = 0
    for lang in sorted(lang_texts):
        lines = lang_texts[lang]
        if not lines:
            continue
        out_path = cdir / f"{lang}_train.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in lines:
                f.write(s + "\n")
        corpus_info[lang] = str(out_path)
        lang_sample_counts[lang] = len(lines)
        total += len(lines)

    preview = list(corpus_info)[:20]
    for lang in preview:
        logger.info("  %s: %d samples", lang, lang_sample_counts[lang])
    if len(corpus_info) > 20:
        logger.info("  ... and %d more languages", len(corpus_info) - 20)
    logger.info(
        "Prepared corpus: %d samples across %d languages -> %s",
        total, len(corpus_info), cdir,
    )
    return corpus_info, lang_sample_counts, total


def sample_corpus(corpus_info, out_dir, max_per_lang):
    """Head-k sampling per language."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampled = {}
    for lang, full_path in corpus_info.items():
        dst = out / f"{lang}_train.sampled.txt"
        count = 0
        with open(full_path, "r", encoding="utf-8", errors="replace") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                fout.write(line)
                count += 1
                if count >= max_per_lang:
                    break
        sampled[lang] = str(dst)
    return sampled


# ─────────────────── reuse existing corpus from disk ────────────────────


def reuse_corpus_from_dir(corpus_dir):
    """Scan an existing corpus dir for *_train.txt files.

    Returns (corpus_info, languages, lang_sample_counts, total_samples).
    """
    existing = []
    for name in os.listdir(corpus_dir):
        if name.endswith("_train.txt"):
            existing.append(os.path.join(corpus_dir, name))
    if not existing:
        return None, None, None, 0
    corpus_info = {}
    lang_sample_counts = {}
    total = 0
    for fp in sorted(existing):
        lang = os.path.basename(fp)[:-10]  # strip _train.txt
        corpus_info[lang] = fp
        count = 0
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                count += 1
        lang_sample_counts[lang] = count
        total += count
    return corpus_info, list(corpus_info.keys()), lang_sample_counts, total


# ──────────────────────────────── main ──────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unified UNILID Training Script",
    )

    # Input — pass exactly one
    inp = parser.add_mutually_exclusive_group()
    inp.add_argument("--fasttext", metavar="FILE", type=str, default=None,
                     help="FastText file: __label__<lang> <text> (one per line)")
    inp.add_argument("--wili-dir", metavar="DIR", type=str, default=None,
                     help="WILI directory containing x_train.txt + y_train.txt")
    inp.add_argument("--tsv", metavar="FILE", type=str, default=None,
                     help="Tatoeba TSV file: id<tab>lang<tab>text")

    # Training
    tr = parser.add_argument_group("Training")
    tr.add_argument("--vocab-size", type=int, default=None,
                    help="Vocabulary size (default: 100000, or inferred from --initial-vocab)")
    tr.add_argument("--base-training-method", type=str, default="hf",
                    choices=["hf", "soft", "hard"],
                    help="Base tokenizer training: hf = HuggingFace UnigramTrainer, "
                         "soft/hard = custom EM (default: hf)")
    tr.add_argument("--lang-training-method", type=str, default="sp",
                    choices=["sp", "soft", "hard"],
                    help="Per-language reestimation: sp = SentencePiece, "
                         "soft/hard = custom EM (default: sp)")
    tr.add_argument("--byte-level", dest="byte_level",
                    action=argparse.BooleanOptionalAction, default=True)
    tr.add_argument("--initial-vocab", metavar="FILE", type=str, default=None,
                    help="Seed vocabulary from an existing tokenizer "
                         "(e.g. LLaMA/Mistral tokenizer.json, or a plain text "
                         "file with one token per line). The base tokenizer "
                         "will reuse this vocabulary and learn new Unigram "
                         "log-probabilities on your training corpus.")
    tr.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    tr.add_argument("--max-samples", type=int, default=None,
                    help="Debug: limit input lines")

    # Sampling
    sa = parser.add_argument_group("Sampling")
    sa.add_argument("--max-base-samples-per-lang", type=int, default=10_000,
                    help="Subsample for base tokenizer (default: 10000)")
    sa.add_argument("--max-lang-samples-per-lang", type=int, default=None,
                    help="Cap per-language training data")
    sa.add_argument("--shared-samples-per-lang", type=int, default=None,
                    help="Single shared sample for both base + lang")

    # Orchestration
    orch = parser.add_argument_group("Orchestration")
    orch.add_argument("--lang-batch-size", type=int, default=10,
                      help="Languages per batch (default: 10)")
    orch.add_argument("--results-dir", type=str, default=None,
                      help="Output dir (default: results_{K}k)")
    orch.add_argument("--corpus-dir", type=str, default=None,
                      help="Pre-split corpus dir to reuse")
    orch.add_argument("--base-tokenizer-path", type=str, default=None,
                      help="Reuse existing base tokenizer")
    orch.add_argument("--reuse-corpus", dest="reuse_corpus",
                      action=argparse.BooleanOptionalAction, default=True)
    orch.add_argument("--reuse-base", dest="reuse_base",
                      action=argparse.BooleanOptionalAction, default=True)
    orch.add_argument("--skip-existing-langs", dest="skip_existing_langs",
                      action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    # ── validate input ──
    if not args.fasttext and not args.wili_dir and not args.tsv and not args.corpus_dir:
        parser.error("Provide one of: --fasttext FILE, --wili-dir DIR, --tsv FILE, or --corpus-dir DIR")

    # ── set random seed ──
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── resolve vocab size ──
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    elif args.initial_vocab:
        import json as _json
        with open(args.initial_vocab, 'r', encoding='utf-8') as _f:
            _data = _json.load(_f)
        if 'model' in _data and 'vocab' in _data['model']:
            vocab_size = len(_data['model']['vocab'])
            logger.info("Inferred vocab_size=%d from --initial-vocab %s",
                        vocab_size, args.initial_vocab)
        else:
            vocab_size = sum(1 for line in open(args.initial_vocab) if line.strip())
            logger.info("Inferred vocab_size=%d from --initial-vocab %s (line count)",
                        vocab_size, args.initial_vocab)
    else:
        vocab_size = 100_000
    results_dir = args.results_dir or f"results_{vocab_size // 1000}k"
    os.makedirs(results_dir, exist_ok=True)

    logger.info("TRAINING starting | vocab=%d  base=%s  lang=%s  byte_level=%s",
                vocab_size, args.base_training_method, args.lang_training_method,
                args.byte_level)

    # ── resolve corpus dir ──
    resolved_corpus_dir = args.corpus_dir or os.path.join(results_dir, "corpus")

    # ── load or reuse corpus ──
    corpus_info = None
    final_languages = None
    lang_sample_counts = None
    total_samples = None

    if args.reuse_corpus and os.path.isdir(resolved_corpus_dir):
        corpus_info, final_languages, lang_sample_counts, total_samples = (
            reuse_corpus_from_dir(resolved_corpus_dir)
        )
        if corpus_info:
            logger.info(
                "Reusing existing corpus in %s (%d languages, %d samples)",
                resolved_corpus_dir, len(final_languages), total_samples,
            )

    if corpus_info is None:
        # Load raw data based on the explicit input flag
        if args.fasttext:
            texts, labels = load_fasttext(args.fasttext, max_samples=args.max_samples)
        elif args.wili_dir:
            texts, labels = load_wili(args.wili_dir, max_samples=args.max_samples)
        elif args.tsv:
            texts, labels = load_tsv(args.tsv, max_samples=args.max_samples)
        else:
            parser.error("No input data: provide --fasttext, --wili-dir, or --tsv")

        corpus_info, lang_sample_counts, total_samples = prepare_corpus(
            texts, labels, resolved_corpus_dir
        )
        final_languages = list(corpus_info.keys())

    # ── output dirs ──
    tokenizers_dir = os.path.join(results_dir, "tokenizers")
    os.makedirs(tokenizers_dir, exist_ok=True)

    start_time = time.time()

    # ── shared sampling ──
    shared_sampling_n = (
        args.shared_samples_per_lang
        if args.shared_samples_per_lang and args.shared_samples_per_lang > 0
        else None
    )
    shared_sampled_corpus = None

    if shared_sampling_n:
        shared_dir = os.path.join(results_dir, "corpus_shared_sampled")
        if args.reuse_corpus and os.path.isdir(shared_dir):
            existing = [
                f for f in os.listdir(shared_dir)
                if f.endswith("_train.sampled.txt")
            ]
            if existing:
                logger.info("Reusing shared sampled corpus in %s", shared_dir)
                shared_sampled_corpus = {}
                for name in sorted(existing):
                    lang = name.split("_train.")[0]
                    shared_sampled_corpus[lang] = os.path.join(shared_dir, name)
        if shared_sampled_corpus is None:
            logger.info(
                "Building shared sampled corpus (<=%d lines/lang) -> %s",
                shared_sampling_n, shared_dir,
            )
            shared_sampled_corpus = sample_corpus(
                corpus_info, shared_dir, shared_sampling_n
            )

    # ── base tokenizer ──
    base_tok_path = args.base_tokenizer_path or os.path.join(
        tokenizers_dir, "langspec_base_tokenizer.json"
    )
    base_reused = False
    base_training_time = 0.0
    if os.path.exists(base_tok_path) and args.reuse_base:
        logger.info("Reusing base tokenizer at %s", base_tok_path)
        base_reused = True
    elif args.initial_vocab and _is_hf_tokenizer(args.initial_vocab):
        # Source is a full HF tokenizer — convert to Unigram, skip training
        logger.info("Converting source tokenizer to Unigram base: %s", args.initial_vocab)
        base_t0 = time.time()
        _convert_to_unigram_base(args.initial_vocab, base_tok_path)
        base_training_time = time.time() - base_t0
        base_reused = False
    else:
        base_tok = StandardUnigramLMTokenizer(
            vocab_size=vocab_size,
            em_mode=args.base_training_method,
            byte_level=args.byte_level,
            initial_vocab_tokens=args.initial_vocab,
        )
        if shared_sampled_corpus is not None:
            train_files = list(shared_sampled_corpus.values())
            logger.info(
                "Training base tokenizer on shared sample (<=%d/lang, %d langs)",
                shared_sampling_n, len(train_files),
            )
        else:
            base_sample_dir = os.path.join(results_dir, "corpus_base_sampled")
            sampled = sample_corpus(
                corpus_info, base_sample_dir, args.max_base_samples_per_lang
            )
            train_files = list(sampled.values())
            logger.info(
                "Training base tokenizer on sampled data (<=%d/lang, %d langs)",
                args.max_base_samples_per_lang, len(train_files),
            )
        base_t0 = time.time()
        base_tok.train(train_files, output_path=base_tok_path)
        base_training_time = time.time() - base_t0

    # ── determine languages to train ──
    def _expected_path(lang):
        return os.path.join(
            tokenizers_dir,
            f"langspec_{args.lang_training_method}_{lang}.tokenizer.json",
        )

    all_langs = list(final_languages)
    if args.skip_existing_langs:
        remaining = [l for l in all_langs if not os.path.isfile(_expected_path(l))]
        skipped = len(all_langs) - len(remaining)
        logger.info(
            "%d existing lang tokenizers found; %d remaining", skipped, len(remaining)
        )
        all_langs = remaining

    # ── per-language sampling ──
    lang_sample_info = None
    if shared_sampled_corpus is not None:
        lang_sample_info = shared_sampled_corpus
    elif args.max_lang_samples_per_lang and args.max_lang_samples_per_lang > 0:
        lang_sample_dir = os.path.join(results_dir, "corpus_langsampled")
        lang_sample_info = sample_corpus(
            corpus_info, lang_sample_dir, args.max_lang_samples_per_lang
        )

    # ── batched per-language training ──
    total_langs = len(all_langs)
    batch_size = max(1, args.lang_batch_size)
    num_batches = math.ceil(total_langs / batch_size) if total_langs else 0
    tokenizer_paths = {}

    logger.info(
        "Training %d languages in %d batches of up to %d",
        total_langs, num_batches, batch_size,
    )

    for bidx in range(num_batches):
        start = bidx * batch_size
        batch_langs = all_langs[start : start + batch_size]
        if not batch_langs:
            break

        if lang_sample_info is not None:
            batch_corpus = {l: lang_sample_info[l] for l in batch_langs}
        else:
            batch_corpus = {l: corpus_info[l] for l in batch_langs}

        logger.info(
            "Batch %d/%d: %d languages [%s .. %s]",
            bidx + 1, num_batches, len(batch_langs),
            batch_langs[0], batch_langs[-1],
        )

        lang_em_mode = args.lang_training_method if args.lang_training_method != "sp" else "soft"
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
                use_sentencepiece=(args.lang_training_method == "sp"),
            )
        except subprocess.CalledProcessError as e:
            logger.error("spm_train failed with exit code %d", e.returncode)
            logger.error("spm_train STDERR:\n%s", e.stderr)
            logger.error("spm_train STDOUT:\n%s", e.stdout)
            raise
        if batch_paths:
            tokenizer_paths.update(batch_paths)
        del tok, batch_paths, batch_corpus
        gc.collect()

    lang_training_time = time.time() - start_time - base_training_time
    total_training_time = time.time() - start_time

    logger.info("TRAINING completed in %.2f seconds", total_training_time)

    # ── summary ──
    fmt = (
        "fasttext" if args.fasttext
        else "wili" if args.wili_dir
        else "tsv" if args.tsv
        else "corpus"
    )
    summary = {
        # What was run
        "command": " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_completed": True,

        # Source data
        "source": {
            "format": fmt,
            "path": os.path.abspath(
                args.fasttext or args.wili_dir or args.tsv or args.corpus_dir
            ),
            "max_samples": args.max_samples,
            "total_samples": int(total_samples) if total_samples is not None else None,
            "num_languages": len(final_languages),
            "samples_per_language": lang_sample_counts,
        },

        # Training method
        "method": {
            "vocab_size": vocab_size,
            "base_training_method": args.base_training_method,
            "lang_training_method": args.lang_training_method,
            "byte_level": args.byte_level,
            "seed": args.seed,
            "initial_vocab": os.path.abspath(args.initial_vocab) if args.initial_vocab else None,
            "lang_batch_size": args.lang_batch_size,
            "sampling": {
                "mode": (
                    "shared" if shared_sampled_corpus is not None
                    else "separate" if (args.max_lang_samples_per_lang and args.max_lang_samples_per_lang > 0)
                    else "base_only"
                ),
                "max_base_samples_per_lang": args.max_base_samples_per_lang,
                "max_lang_samples_per_lang": args.max_lang_samples_per_lang,
                "shared_samples_per_lang": shared_sampling_n,
            },
            "reuse": {
                "reuse_corpus": args.reuse_corpus,
                "reuse_base": args.reuse_base,
                "skip_existing_langs": args.skip_existing_langs,
            },
        },

        # Timing
        "timing": {
            "total_seconds": round(total_training_time, 2),
            "base_tokenizer_seconds": round(base_training_time, 2),
            "language_tokenizers_seconds": round(lang_training_time, 2),
            "base_tokenizer_reused": base_reused,
        },

        # Output files
        "output": {
            "results_dir": os.path.abspath(results_dir),
            "corpus_dir": os.path.abspath(resolved_corpus_dir),
            "tokenizers_dir": os.path.abspath(tokenizers_dir),
            "base_tokenizer": os.path.abspath(base_tok_path),
            "num_languages_trained_this_run": total_langs,
            "language_tokenizers": tokenizer_paths,
        },

        # All languages in corpus (for reference)
        "languages": final_languages,
    }

    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(
        "TRAINING COMPLETE | %dK vocab | %d languages trained this run | "
        "%d total languages | %.1fs",
        vocab_size // 1000, total_langs, len(final_languages), total_training_time,
    )
    logger.info("Results: %s/", results_dir)
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
