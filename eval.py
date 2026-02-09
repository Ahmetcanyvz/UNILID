#!/usr/bin/env python3
"""
UNILID evaluation and prediction script.

Modes:
    1. Prediction (default): Stream predictions to output file
       python eval.py --model model.unilid --input texts.txt --output predictions.tsv

    2. Evaluation (--fasttext): Evaluate on labeled fastText-format file
       python eval.py --model model.unilid --input test.txt --fasttext --output results.json
"""
import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer


def _get_vocab_with_scores(tok: Tokenizer):
    """Extract vocab with scores from HF Unigram tokenizer."""
    state = tok.model.__getstate__()
    attributes = json.loads(state.decode("utf-8"))
    return attributes["vocab"]


def load_model_from_unilid(model_path: Path):
    """Load from .unilid file."""
    from unilid.model_io import load_unilid

    base_tok, weights, langs = load_unilid(model_path)

    print("Pushing weights to Rust cache...", file=sys.stderr)
    base_tok.model.set_weight_sets(np.array(weights).tolist())

    pre_tok = base_tok.pre_tokenizer
    normalizer = base_tok.normalizer

    del weights
    gc.collect()

    return base_tok, pre_tok, normalizer, langs


def load_model_from_dir(model_dir: Path):
    """Load model with streaming to minimize memory."""
    tok_dir = model_dir / "tokenizers"
    base_json = tok_dir / "langspec_base_tokenizer.json"

    if not base_json.exists():
        lang_files = sorted(tok_dir.glob("langspec_soft_*.tokenizer.json"))
        if not lang_files:
            lang_files = sorted(tok_dir.glob("langspec_sp_*.tokenizer.json"))
        if lang_files:
            base_json = lang_files[0]
        else:
            raise FileNotFoundError(f"No tokenizers found in {tok_dir}")

    print(f"Loading base tokenizer: {base_json.name}", file=sys.stderr)
    base_tok = Tokenizer.from_file(str(base_json))
    ref_vocab = base_tok.get_vocab()
    V = len(ref_vocab)

    pre_tok = base_tok.pre_tokenizer
    normalizer = base_tok.normalizer

    lang_files = sorted(tok_dir.glob("langspec_soft_*.tokenizer.json"))
    if not lang_files:
        lang_files = sorted(tok_dir.glob("langspec_sp_*.tokenizer.json"))

    print(f"Found {len(lang_files)} language tokenizers, vocab size {V}", file=sys.stderr)

    very_neg = np.float32(-1e30)
    langs = []
    weights = np.full((len(lang_files), V), very_neg, dtype=np.float32)

    for i, lang_path in enumerate(tqdm(lang_files, desc="Loading weights", file=sys.stderr)):
        lang_code = lang_path.stem
        for prefix in ["langspec_soft_", "langspec_sp_"]:
            lang_code = lang_code.replace(prefix, "")
        lang_code = lang_code.replace(".tokenizer", "")

        tok = Tokenizer.from_file(str(lang_path))
        vocab_scores = _get_vocab_with_scores(tok)
        scores = {token: score for token, score in vocab_scores}

        for token, rid in ref_vocab.items():
            lp = scores.get(token)
            if lp is not None:
                weights[i, rid] = np.float32(lp)

        langs.append(lang_code)
        del tok, scores

        if (i + 1) % 50 == 0:
            gc.collect()

    gc.collect()

    print("Pushing weights to Rust cache...", file=sys.stderr)
    base_tok.model.set_weight_sets(weights.tolist())
    del weights
    gc.collect()

    return base_tok, pre_tok, normalizer, langs


def load_model(model_path: Path):
    """Load model from .unilid file or tokenizers directory."""
    model_path = Path(model_path)

    if model_path.suffix == ".unilid" or (model_path.is_file() and model_path.name.endswith(".unilid")):
        return load_model_from_unilid(model_path)
    elif model_path.is_dir():
        return load_model_from_dir(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path}")


def preprocess(text: str, pre_tok, normalizer):
    """Preprocess text with normalizer and pretokenizer. Returns None if empty."""
    if normalizer:
        text = normalizer.normalize_str(text)
    if pre_tok:
        pre = pre_tok.pre_tokenize_str(text)
        if not pre:
            return None
        return "".join(piece[0] for piece in pre)
    return text if text else None


# =============================================================================
# PREDICTION MODE (streaming)
# =============================================================================

def count_lines(file_path: Path) -> int:
    """Count lines in file for progress bar."""
    count = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return count


def predict_streaming(model, pre_tok, normalizer, langs, input_path: Path, output_path: Path,
                      batch_size: int = 10000, output_format: str = "tsv"):
    """Stream predictions to output file with minimal memory usage."""
    print(f"Counting lines...", file=sys.stderr)
    total_lines = count_lines(input_path)
    print(f"Total lines: {total_lines}", file=sys.stderr)

    processed = 0
    skipped = 0
    start = time.perf_counter()

    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        # Write header for TSV
        if output_format == "tsv":
            fout.write("text\tlang\tscore\n")

        batch_texts = []
        batch_originals = []

        pbar = tqdm(total=total_lines, desc="Predicting", file=sys.stderr)

        for line in fin:
            text = line.rstrip("\n\r")
            pbar.update(1)

            if not text:
                skipped += 1
                continue

            pt = preprocess(text, pre_tok, normalizer)
            if not pt:
                skipped += 1
                # Write empty prediction
                if output_format == "tsv":
                    fout.write(f"{text}\t\t\n")
                else:
                    fout.write(json.dumps({"text": text, "lang": None, "score": None}) + "\n")
                continue

            batch_texts.append(pt)
            batch_originals.append(text)

            # Process batch
            if len(batch_texts) >= batch_size:
                results = model.best_of_cached_weight_sets_batch(batch_texts)

                for orig_text, (idx, toks, score) in zip(batch_originals, results):
                    lang = langs[idx]
                    if output_format == "tsv":
                        fout.write(f"{orig_text}\t{lang}\t{score:.4f}\n")
                    else:
                        fout.write(json.dumps({"text": orig_text, "lang": lang, "score": score}) + "\n")

                processed += len(batch_texts)
                batch_texts.clear()
                batch_originals.clear()

        # Process remaining
        if batch_texts:
            results = model.best_of_cached_weight_sets_batch(batch_texts)

            for orig_text, (idx, toks, score) in zip(batch_originals, results):
                lang = langs[idx]
                if output_format == "tsv":
                    fout.write(f"{orig_text}\t{lang}\t{score:.4f}\n")
                else:
                    fout.write(json.dumps({"text": orig_text, "lang": lang, "score": score}) + "\n")

            processed += len(batch_texts)

        pbar.close()

    elapsed = time.perf_counter() - start
    print(f"\nProcessed: {processed}, Skipped: {skipped}", file=sys.stderr)
    print(f"Time: {elapsed:.2f}s, Speed: {processed/elapsed:.2f} samples/s", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)


# =============================================================================
# EVALUATION MODE (fastText format)
# =============================================================================

def load_fasttext_data(test_path: Path, model_labels: set, lang_only: bool = False):
    """Load fastText format test file."""
    label_re = re.compile(r"^__label__(\S+)")

    def extract_lang(label):
        return label.split("_", 1)[0] if lang_only else label

    model_labels_check = {extract_lang(l) for l in model_labels} if lang_only else model_labels

    texts, gold_labels = [], []
    skipped = 0

    with open(test_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = label_re.match(line)
            if not m:
                continue
            label = m.group(1)
            parts = line.split(" ", 1)
            text = parts[1] if len(parts) == 2 else ""

            label_check = extract_lang(label)
            if label_check not in model_labels_check:
                skipped += 1
                continue

            texts.append(text)
            gold_labels.append(label)

    return texts, gold_labels, skipped


def compute_metrics(confusion_counts: Counter):
    """Compute metrics from confusion counts."""
    gold_labels = set()
    for gold, pred in confusion_counts.keys():
        gold_labels.add(gold)
    labels = sorted(gold_labels)

    total = sum(confusion_counts.values())
    correct = sum(c for (g, p), c in confusion_counts.items() if g == p)
    accuracy = correct / total if total else 0

    gold_totals, pred_totals = {}, {}
    for (g, p), c in confusion_counts.items():
        gold_totals[g] = gold_totals.get(g, 0) + c
        pred_totals[p] = pred_totals.get(p, 0) + c

    f1s, precs, recs = [], [], []
    for label in labels:
        tp = confusion_counts.get((label, label), 0)
        fp = pred_totals.get(label, 0) - tp
        fn = gold_totals.get(label, 0) - tp

        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    return {
        "total_samples": total,
        "num_labels": len(labels),
        "accuracy": accuracy,
        "macro_f1": np.mean(f1s),
        "macro_precision": np.mean(precs),
        "macro_recall": np.mean(recs),
    }


def evaluate_batch(model, pre_tok, normalizer, langs, texts, gold_labels, batch_size=10000, lang_only=False):
    """Evaluate using batch parallel inference."""
    def extract_lang(label):
        return label.split("_", 1)[0] if lang_only else label

    confusion_counts = Counter()
    num_chunks = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=num_chunks, desc="Evaluating", file=sys.stderr):
        chunk_texts = texts[i:i + batch_size]
        chunk_gold = gold_labels[i:i + batch_size]

        preprocessed, valid_idx = [], []
        for j, text in enumerate(chunk_texts):
            pt = preprocess(text, pre_tok, normalizer)
            if pt:
                preprocessed.append(pt)
                valid_idx.append(j)

        if not preprocessed:
            continue

        results = model.best_of_cached_weight_sets_batch(preprocessed)

        for k, (idx, toks, score) in enumerate(results):
            orig_j = valid_idx[k]
            pred = extract_lang(langs[idx])
            gold = extract_lang(chunk_gold[orig_j])
            confusion_counts[(gold, pred)] += 1

    return confusion_counts


def evaluate_single(model, pre_tok, normalizer, langs, texts, gold_labels, lang_only=False):
    """Evaluate using single-core inference."""
    def extract_lang(label):
        return label.split("_", 1)[0] if lang_only else label

    confusion_counts = Counter()

    for text, gold_label in tqdm(zip(texts, gold_labels), total=len(texts), desc="Evaluating", file=sys.stderr):
        pt = preprocess(text, pre_tok, normalizer)
        if not pt:
            continue

        idx, toks, score = model.best_of_cached_weight_sets(pt)
        pred = extract_lang(langs[idx])
        gold = extract_lang(gold_label)
        confusion_counts[(gold, pred)] += 1

    return confusion_counts


def run_evaluation(model, pre_tok, normalizer, langs, input_path: Path, output_path: Path,
                   batch_size: int, single_core: bool, lang_only: bool):
    """Run evaluation on fastText-format file."""
    print(f"Loading test data from {input_path}...", file=sys.stderr)
    texts, gold_labels, skipped = load_fasttext_data(input_path, set(langs), lang_only)
    print(f"Loaded {len(texts)} samples ({skipped} skipped - not in model)", file=sys.stderr)

    if not texts:
        print("No samples to evaluate!", file=sys.stderr)
        return

    mode = "single-core" if single_core else "batch-parallel"
    print(f"Evaluating ({mode})...", file=sys.stderr)

    start = time.perf_counter()
    if single_core:
        confusion = evaluate_single(model, pre_tok, normalizer, langs, texts, gold_labels, lang_only)
    else:
        confusion = evaluate_batch(model, pre_tok, normalizer, langs, texts, gold_labels, batch_size, lang_only)
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(confusion)
    metrics["inference_time_s"] = elapsed
    metrics["samples_per_second"] = len(texts) / elapsed
    metrics["mode"] = mode

    print(f"\n{'='*50}", file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print('='*50, file=sys.stderr)
    print(f"Total samples: {metrics['total_samples']}", file=sys.stderr)
    print(f"Num languages: {metrics['num_labels']}", file=sys.stderr)
    print(f"Accuracy: {metrics['accuracy']:.6f}", file=sys.stderr)
    print(f"Macro F1: {metrics['macro_f1']:.6f}", file=sys.stderr)
    print(f"Macro Precision: {metrics['macro_precision']:.6f}", file=sys.stderr)
    print(f"Macro Recall: {metrics['macro_recall']:.6f}", file=sys.stderr)
    print(f"\nInference time: {elapsed:.2f}s", file=sys.stderr)
    print(f"Samples/second: {metrics['samples_per_second']:.2f}", file=sys.stderr)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="UNILID prediction and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on plain text file (one text per line)
  python eval.py --model model.unilid --input texts.txt --output predictions.tsv

  # Predict with JSONL output
  python eval.py --model model.unilid --input texts.txt --output predictions.jsonl --format jsonl

  # Evaluate on fastText-format file
  python eval.py --model model.unilid --input test.txt --fasttext --output results.json

  # Evaluate with language-only labels (ignore script)
  python eval.py --model model.unilid --input test.txt --fasttext --lang-only
        """
    )
    parser.add_argument("--model", type=Path, required=True, help="Model path (.unilid file or directory)")
    parser.add_argument("--input", type=Path, required=True, help="Input file path")
    parser.add_argument("--output", type=Path, default=None, help="Output file path")
    parser.add_argument("--fasttext", action="store_true", help="Evaluate on fastText-format file with __label__ prefix")
    parser.add_argument("--format", choices=["tsv", "jsonl"], default="tsv", help="Output format for predictions (default: tsv)")
    parser.add_argument("--single-core", action="store_true", help="Use single-core inference")
    parser.add_argument("--lang-only", action="store_true", help="Use language-only labels (ignore script)")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for parallel inference")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...", file=sys.stderr)
    base_tok, pre_tok, normalizer, langs = load_model(args.model)
    model = base_tok.model
    print(f"Loaded {len(langs)} languages", file=sys.stderr)

    if args.fasttext:
        # Evaluation mode
        run_evaluation(model, pre_tok, normalizer, langs, args.input, args.output,
                       args.batch_size, args.single_core, args.lang_only)
    else:
        # Prediction mode
        if args.output is None:
            # Default output path
            suffix = ".jsonl" if args.format == "jsonl" else ".tsv"
            args.output = args.input.with_suffix(suffix).with_stem(args.input.stem + "_predictions")

        predict_streaming(model, pre_tok, normalizer, langs, args.input, args.output,
                          args.batch_size, args.format)


if __name__ == "__main__":
    main()
