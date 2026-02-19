#!/usr/bin/env python3
"""
Parameterized tokenizer training for the 3x3 SLURM grid.

One script, 9 jobs. Each job trains one combination of:
  --base-method:  apertus | bpe | unigram
  --grouping:     per_lang | fine_family | coarse_family

Data domains handled within each job:
  - Language:  fineweb2/{lang_Script}.parquet  +  fineweb/CC-MAIN-*.parquet (English)
  - Math:      finemath + infimath + megamath  (one "math" group)
  - Code:      starcoder/{lang}.parquet

Usage:
    python train_tokenizer.py --base-method apertus --grouping per_lang
    python train_tokenizer.py --base-method bpe --grouping fine_family
    python train_tokenizer.py --base-method unigram --grouping coarse_family

Submit all 9:
    bash submit_all.sh
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_tokenizer")


# ═══════════════════════════════════════════════════════════════
# Default cluster paths
# ═══════════════════════════════════════════════════════════════

DATA_DIR = "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/tokenizer_training_dataset"
APERTUS_PATH = "/users/ayavuz/tokenizer_train/tokenizer.json"
COARSE_FAMILIES_PATH = "/users/ayavuz/tokenizer_train/coarse_families.json"
FINE_FAMILIES_PATH = "/users/ayavuz/tokenizer_train/fine_families.json"
CODE_FAMILIES_PATH = (
    "/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/"
    "tokenizer_training_dataset/starcoder/code_lang_families.yaml"
)


# ═══════════════════════════════════════════════════════════════
# Parquet I/O helpers
# ═══════════════════════════════════════════════════════════════


def discover_parquets(directory, pattern="*.parquet"):
    """Return sorted {stem: absolute_path} for all matching parquets."""
    d = Path(directory)
    if not d.is_dir():
        logger.warning("Directory not found: %s", directory)
        return {}
    return {p.stem: str(p) for p in sorted(d.glob(pattern))}


def sample_from_parquet(parquet_path, max_lines, seed=42, text_col="text"):
    """Read a parquet and return up to max_lines sampled texts."""
    table = pq.read_table(parquet_path, columns=[text_col])
    texts = table.column(text_col).to_pylist()
    del table
    if len(texts) > max_lines:
        texts = random.Random(seed).sample(texts, max_lines)
    return [t for t in texts if t and t.strip()]


def write_texts(texts, path):
    """Write texts to file, one per line. Returns line count."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ").strip() + "\n")
    return len(texts)


def parquets_to_txt(parquet_paths, output_path, text_col="text"):
    """Stream one or more parquets into a single text file. Returns line count."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ppath in parquet_paths:
            pf = pq.ParquetFile(ppath)
            for batch in pf.iter_batches(batch_size=10_000, columns=[text_col]):
                for text in batch.column(text_col).to_pylist():
                    if text and text.strip():
                        f.write(text.replace("\n", " ").strip() + "\n")
                        count += 1
    return count


# ═══════════════════════════════════════════════════════════════
# Family / grouping helpers
# ═══════════════════════════════════════════════════════════════


def load_lang_families(json_path):
    """Load a family JSON → inverted mapping {lang_Script: family_name}."""
    with open(json_path) as f:
        families = json.load(f)
    mapping = {}
    for family, members in families.items():
        for lang_script in members:
            mapping[lang_script] = family
    return mapping


def load_code_families(yaml_path):
    """Load code families YAML → inverted mapping {code_lang: [family_names]}.

    A code language may appear in multiple families (overlaps are intentional).
    """
    with open(yaml_path) as f:
        families = yaml.safe_load(f)
    mapping = defaultdict(list)
    for family, members in families.items():
        if members:
            for lang in members:
                mapping[lang].append(family)
    return dict(mapping)


def build_groups(
    lang_parquets,    # {lang_Script: parquet_path}
    eng_shards,       # {shard_name: parquet_path}
    code_parquets,    # {code_lang: parquet_path}
    math_parquets,    # {name: parquet_path}
    grouping,         # "per_lang" | "fine_family" | "coarse_family"
    lang_family_map,  # {lang_Script: family} or None
    code_family_map,  # {code_lang: [families]} or None
):
    """Build {group_name: [parquet_path, ...]}."""
    groups = defaultdict(list)

    if grouping == "per_lang":
        # Each lang_Script is its own group
        for lang, path in lang_parquets.items():
            groups[lang].append(path)
        # English fineweb shards merge into eng_Latn
        for _, path in eng_shards.items():
            groups["eng_Latn"].append(path)
        # Each code language is its own group
        for code_lang, path in code_parquets.items():
            groups[f"code_{code_lang}"].append(path)
        # Math is one group
        for _, path in math_parquets.items():
            groups["math"].append(path)

    else:
        # Family-level grouping for language data
        for lang, path in lang_parquets.items():
            family = lang_family_map.get(lang)
            if family is None:
                logger.warning("No family mapping for %s — treating as singleton", lang)
                family = lang
            groups[family].append(path)

        # English shards → same family as eng_Latn
        eng_family = lang_family_map.get("eng_Latn", "eng_Latn")
        for _, path in eng_shards.items():
            groups[eng_family].append(path)

        # Code data → code families (a language may appear in multiple families)
        for code_lang, path in code_parquets.items():
            families = code_family_map.get(code_lang) if code_family_map else None
            if families is None:
                logger.warning("No code family for %s — treating as singleton", code_lang)
                groups[f"code_{code_lang}"].append(path)
            else:
                for fam in families:
                    groups[fam].append(path)

        # Math is always one group
        for _, path in math_parquets.items():
            groups["math"].append(path)

    return dict(groups)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Train one tokenizer configuration (base-method × grouping)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Core ──
    parser.add_argument(
        "--base-method", required=True,
        choices=["apertus", "bpe", "unigram"],
        help="How to build the base tokenizer",
    )
    parser.add_argument(
        "--grouping", required=True,
        choices=["per_lang", "fine_family", "coarse_family"],
        help="How to group data for per-group training",
    )

    # ── Paths ──
    paths = parser.add_argument_group("Paths")
    paths.add_argument("--data-dir", default=DATA_DIR,
                       help="Root data directory on cluster")
    paths.add_argument("--apertus-path", default=APERTUS_PATH,
                       help="Apertus tokenizer.json (pretok source for all methods)")
    paths.add_argument("--coarse-families", default=COARSE_FAMILIES_PATH)
    paths.add_argument("--fine-families", default=FINE_FAMILIES_PATH)
    paths.add_argument("--code-families", default=CODE_FAMILIES_PATH)
    paths.add_argument("--results-dir", default=None,
                       help="Output dir (default: results_{method}_{grouping})")

    # ── Training ──
    tr = parser.add_argument_group("Training")
    tr.add_argument("--vocab-size", type=int, default=100_000,
                    help="Base vocab size (ignored for --base-method apertus)")
    tr.add_argument("--base-samples", type=int, default=10_000,
                    help="Lines to sample per source for base tokenizer training")
    tr.add_argument("--batch-size", type=int, default=10,
                    help="Groups per batch for per-group training")
    tr.add_argument("--num-em-iterations", type=int, default=10,
                    help="SentencePiece EM iterations for per-group re-estimation")
    tr.add_argument("--seed", type=int, default=42)

    # ── Resumability ──
    sk = parser.add_argument_group("Resumability")
    sk.add_argument("--reuse-base", action=argparse.BooleanOptionalAction, default=True,
                    help="Skip base training if base_tokenizer.json exists")
    sk.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                    help="Skip groups that already have a tokenizer file")
    sk.add_argument("--reuse-sampled", action=argparse.BooleanOptionalAction, default=True,
                    help="Reuse existing sampled corpus files")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Resolve output directory ──
    results_dir = args.results_dir or f"results_{args.base_method}_{args.grouping}"
    os.makedirs(results_dir, exist_ok=True)

    tokenizers_dir = os.path.join(results_dir, "tokenizers")
    os.makedirs(tokenizers_dir, exist_ok=True)

    sampled_dir = os.path.join(results_dir, "base_corpus")
    tmp_dir = os.path.join(results_dir, "tmp")

    base_tok_path = os.path.join(tokenizers_dir, "base_tokenizer.json")

    logger.info("=" * 60)
    logger.info("JOB: %s × %s", args.base_method, args.grouping)
    logger.info("Results: %s", results_dir)
    logger.info("=" * 60)

    t0 = time.time()

    # ══════════════════════════════════════════════════════════
    # Phase 1: Discover data sources
    # ══════════════════════════════════════════════════════════

    logger.info("Phase 1: Discovering data sources")
    data_dir = args.data_dir

    lang_parquets = discover_parquets(os.path.join(data_dir, "fineweb2"))
    eng_shards = discover_parquets(os.path.join(data_dir, "fineweb"))
    code_parquets = discover_parquets(os.path.join(data_dir, "starcoder"))

    math_parquets = {}
    for name in ["finemath", "infimath", "megamath"]:
        p = os.path.join(data_dir, name, "data.parquet")
        if os.path.isfile(p):
            math_parquets[name] = p
        else:
            logger.warning("Math parquet not found: %s", p)

    logger.info(
        "Found: %d lang, %d eng shards, %d code, %d math",
        len(lang_parquets), len(eng_shards), len(code_parquets), len(math_parquets),
    )

    # ══════════════════════════════════════════════════════════
    # Phase 2: Build group mapping
    # ══════════════════════════════════════════════════════════

    logger.info("Phase 2: Building groups (%s)", args.grouping)

    lang_family_map = None
    code_family_map = None

    if args.grouping == "fine_family":
        lang_family_map = load_lang_families(args.fine_families)
    elif args.grouping == "coarse_family":
        lang_family_map = load_lang_families(args.coarse_families)

    if args.grouping != "per_lang":
        code_family_map = load_code_families(args.code_families)

    groups = build_groups(
        lang_parquets, eng_shards, code_parquets, math_parquets,
        args.grouping, lang_family_map, code_family_map,
    )

    logger.info("Built %d groups", len(groups))
    # Log the 10 largest groups
    for g, paths in sorted(groups.items(), key=lambda x: -len(x[1]))[:10]:
        logger.info("  %-30s  %d parquets", g, len(paths))

    # ══════════════════════════════════════════════════════════
    # Phase 3: Sample for base tokenizer
    # ══════════════════════════════════════════════════════════

    if args.base_method == "apertus":
        logger.info("Phase 3: Skipped (using apertus tokenizer as base)")
        base_tok_path = args.apertus_path
    else:
        logger.info("Phase 3: Sampling from all sources for base tokenizer")
        os.makedirs(sampled_dir, exist_ok=True)

        # Collect every unique parquet across all sources
        all_source_parquets = {}
        for lang, path in lang_parquets.items():
            all_source_parquets[f"lang_{lang}"] = path
        for shard, path in eng_shards.items():
            all_source_parquets[f"eng_{shard}"] = path
        for code_lang, path in code_parquets.items():
            all_source_parquets[f"code_{code_lang}"] = path
        for name, path in math_parquets.items():
            all_source_parquets[f"math_{name}"] = path

        sampled_files = []
        for i, (key, ppath) in enumerate(sorted(all_source_parquets.items())):
            dst = os.path.join(sampled_dir, f"{key}.sampled.txt")

            if args.reuse_sampled and os.path.isfile(dst):
                sampled_files.append(dst)
                continue

            texts = sample_from_parquet(ppath, args.base_samples, seed=args.seed)
            if texts:
                write_texts(texts, dst)
                sampled_files.append(dst)
            del texts

            if (i + 1) % 200 == 0:
                logger.info("  Sampled %d/%d sources", i + 1, len(all_source_parquets))

        logger.info(
            "Phase 3 done: %d sampled files in %s (%.1fs)",
            len(sampled_files), sampled_dir, time.time() - t0,
        )

        # ══════════════════════════════════════════════════════
        # Phase 4: Train base tokenizer
        # ══════════════════════════════════════════════════════

        logger.info("Phase 4: Training base tokenizer (%s)", args.base_method)

        if os.path.isfile(base_tok_path) and args.reuse_base:
            logger.info("Reusing existing base tokenizer at %s", base_tok_path)
        else:
            from tokenizers import Tokenizer as HFTokenizer
            from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer

            # Load apertus to extract pretokenizer / decoder / normalizer
            apertus_tok = HFTokenizer.from_file(args.apertus_path)

            em_mode = "bpe" if args.base_method == "bpe" else "hf"
            base_trainer = StandardUnigramLMTokenizer(
                vocab_size=args.vocab_size,
                em_mode=em_mode,
                byte_level=True,
            )
            # Inject apertus components so the new tokenizer uses the same
            # pretokenization / decoding as apertus
            base_trainer._source_pretokenizer = apertus_tok.pre_tokenizer
            base_trainer._source_decoder = apertus_tok.decoder
            base_trainer._source_normalizer = apertus_tok.normalizer

            logger.info(
                "Training %s base tokenizer on %d sampled files (vocab=%d)",
                em_mode.upper(), len(sampled_files), args.vocab_size,
            )
            t_base = time.time()
            base_trainer.train(sampled_files, output_path=base_tok_path)
            logger.info("Base tokenizer trained in %.1fs", time.time() - t_base)

            del base_trainer, apertus_tok
            gc.collect()

    logger.info("Base tokenizer: %s", base_tok_path)

    # ══════════════════════════════════════════════════════════
    # Phase 5: Batched per-group training
    # ══════════════════════════════════════════════════════════

    logger.info("Phase 5: Per-group training (%d groups)", len(groups))

    all_group_names = sorted(groups.keys())

    def group_tok_path(name):
        return os.path.join(tokenizers_dir, f"langspec_sp_{name}.tokenizer.json")

    # Filter already-trained groups
    if args.skip_existing:
        remaining = [g for g in all_group_names if not os.path.isfile(group_tok_path(g))]
        skipped = len(all_group_names) - len(remaining)
        if skipped:
            logger.info("Skipping %d existing groups, %d remaining", skipped, len(remaining))
        groups_to_train = remaining
    else:
        groups_to_train = all_group_names

    batch_size = max(1, args.batch_size)
    num_batches = math.ceil(len(groups_to_train) / batch_size) if groups_to_train else 0

    logger.info(
        "Training %d groups in %d batches of %d",
        len(groups_to_train), num_batches, batch_size,
    )

    os.makedirs(tmp_dir, exist_ok=True)

    from unilid.trainers.language_specific_trainer import LanguageSpecificUnigramLMTokenizer

    phase5_t0 = time.time()

    for bidx in range(num_batches):
        batch = groups_to_train[bidx * batch_size : (bidx + 1) * batch_size]
        logger.info(
            "Batch %d/%d: %d groups [%s .. %s]",
            bidx + 1, num_batches, len(batch), batch[0], batch[-1],
        )

        # Step 1: Write parquets → tmp txt for this batch
        batch_corpus = {}
        batch_tmp_files = []

        for group_name in batch:
            tmp_path = os.path.join(tmp_dir, f"{group_name}.txt")
            parquet_paths = groups[group_name]

            logger.info("  %s: %d parquets → %s", group_name, len(parquet_paths), tmp_path)
            n = parquets_to_txt(parquet_paths, tmp_path)
            logger.info("  %s: %d lines written", group_name, n)

            if n > 0:
                batch_corpus[group_name] = tmp_path
                batch_tmp_files.append(tmp_path)
            else:
                logger.warning("  %s: no data, skipping", group_name)

        if not batch_corpus:
            logger.warning("Batch %d: empty, skipping", bidx + 1)
            continue

        # Step 2: Train per-group tokenizers
        tok = LanguageSpecificUnigramLMTokenizer(
            vocab_size=args.vocab_size,
            reestimation_em_mode="soft",
            num_iterations=args.num_em_iterations,
            byte_level=True,
        )

        try:
            tok.train(
                corpus_info=batch_corpus,
                base_tokenizer_path=base_tok_path,
                output_dir=tokenizers_dir,
                tokenizer_path_format=os.path.join(
                    tokenizers_dir, "langspec_sp_{lang_code}.tokenizer.json"
                ),
                load_base_if_exists=True,
                load_tokenizers_if_exists=args.skip_existing,
                use_sentencepiece=True,
                num_reestimation_iterations=args.num_em_iterations,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                "spm_train failed (exit %d) for batch %d\nSTDERR: %s",
                e.returncode, bidx + 1, e.stderr,
            )
            raise

        # Step 3: Clean up tmp files
        for f in batch_tmp_files:
            try:
                os.remove(f)
            except OSError:
                pass

        del tok, batch_corpus
        gc.collect()

        logger.info(
            "Batch %d/%d done (%.1fs elapsed)",
            bidx + 1, num_batches, time.time() - phase5_t0,
        )

    # Clean tmp dir if empty
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════

    total_time = time.time() - t0
    trained_count = sum(1 for g in all_group_names if os.path.isfile(group_tok_path(g)))

    summary = {
        "job": f"{args.base_method}_{args.grouping}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_method": args.base_method,
        "grouping": args.grouping,
        "vocab_size": args.vocab_size,
        "num_em_iterations": args.num_em_iterations,
        "seed": args.seed,
        "total_groups": len(all_group_names),
        "trained_groups": trained_count,
        "training_time_seconds": round(total_time, 1),
        "base_tokenizer": os.path.abspath(base_tok_path),
        "results_dir": os.path.abspath(results_dir),
        "data_sources": {
            "lang_parquets": len(lang_parquets),
            "eng_shards": len(eng_shards),
            "code_parquets": len(code_parquets),
            "math_parquets": len(math_parquets),
        },
        "groups": all_group_names,
    }

    summary_path = os.path.join(results_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(
        "DONE: %s × %s | %d/%d groups trained | %.1fs",
        args.base_method, args.grouping, trained_count, len(all_group_names), total_time,
    )
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
