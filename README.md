# UNILID

Fast multilingual language identification using unigram language models. A
shared token vocabulary is trained across languages, each language gets its own
token probability distribution over it, and a text is labeled with the language
that scores it highest (Rust-accelerated, Rayon-parallel batch inference). The
released model covers 1,940 language-script combinations and ships with
calibrated inference (default), per-language decision thresholds that can be
extended one language at a time, and language subsetting for latency-sensitive
deployments.

## Installation

Prerequisites:

- Python 3.9 or newer
- A Rust toolchain (`cargo`; install via [rustup](https://rustup.rs)) for the mandatory tokenizers build
- cmake and a C++ compiler, only if you build the optional SentencePiece CLI (used by the `sp` training method)

```bash
# Clone with submodules (required for the custom tokenizers)
git clone --recurse-submodules https://github.com/Ahmetcanyvz/UNILID.git && cd UNILID

python3 -m venv .venv
source .venv/bin/activate

# Install unilid (prediction needs only this plus the tokenizers build below)
pip install -e .

# Build the custom tokenizers extension (REQUIRED for inference; standard
# HuggingFace tokenizers will NOT work)
pip uninstall tokenizers -y
pip install maturin
cd tokenizers/bindings/python
unset CONDA_PREFIX  # if using conda
maturin develop --release
cd ../../..
```

If already cloned without submodules: `git submodule update --init --recursive`.

Verify the build:

```python
from tokenizers.models import Unigram
for method in ("set_weight_sets", "set_weight_sets_numpy",
               "top_k_of_cached_weight_sets_batch",
               "tokens_of_cached_weight_set_batch"):
    assert hasattr(Unigram, method), f"custom tokenizers build missing {method}; rebuild the submodule"
print("tokenizers extension OK")
```

These are the methods model loading itself checks for; an older build fails at
load time with the same rebuild instruction.

Optional extras: `pip install -e ".[train]"` for training (torch, transformers,
ujson), `pip install -e ".[dev]"` for the test suite
(`python -m pytest tests/`). The `sp` training method additionally needs the
forked SentencePiece CLI; see [Training](#training-a-model) below.

## Download a pre-trained model

| Model | Languages | Training Data | Calibration | Download |
|-------|-----------|---------------|-------------|----------|
| unilid-1940-calibrated | 1940 language-script combinations | 60M samples | bundled (version-2 file) | [HuggingFace Hub](https://huggingface.co/cmeister/unilid-1940) |
| unilid-1940 | 1940 language-script combinations | 60M samples | none (version-1 file) | [polybox](https://polybox.ethz.ch/index.php/s/Kbb9TWkSSgQ8yoS) |

Both files contain the same trained model; the calibrated file additionally
bundles the calibration artifact (160 KB). The file is 780 MB; loading builds
the float32 weight matrix in memory, so plan for roughly 2 to 3 GB of free RAM.

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('cmeister/unilid-1940', 'unilid-1940-calibrated.unilid', local_dir='.'))"
# or directly:
# wget https://huggingface.co/cmeister/unilid-1940/resolve/main/unilid-1940-calibrated.unilid
```

## Predict

```python
from unilid import load_model

model = load_model("unilid-1940-calibrated.unilid")
lang, tokens, score = model.predict("The quick brown fox jumps over the lazy dog.")
print(lang)  # 'eng_Latn'

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
]
for text, (lang, tokens, score) in zip(texts, model.predict_batch(texts)):
    print(f"{lang}: {text[:50]}")
```

Prediction defaults to **calibrated inference**: a shared constant replaces
each language's unseen-token log-probabilities at load time, and close
decisions that land in two error-prone groups of languages are re-examined
against each language's own threshold. On the GlotLID-C test pool this raises
macro F1 from 0.929 to 0.957. The mechanism, the constants, and the full
measured effects (including the evaluations where calibration lowers a metric)
are described in [REPRODUCING.md](REPRODUCING.md) and specified in the UNILID
paper.

The original release's uncalibrated behavior is one flag away:

```python
base_model = load_model("unilid-1940-calibrated.unilid", calibrated=False)
```

Loading a model that has no calibration artifact (a version-1 `.unilid` file,
including the polybox release and self-trained models) with default arguments
raises `UnilidCalibrationError`; pass `calibrated=False` for such files.

Batch inference uses Rayon and defaults to all CPU cores; limit it with
`RAYON_NUM_THREADS=4 python ...`.

## Restrict to a subset of languages

Scoring cost is linear in the number of languages, so restricting the model to
the languages a deployment can actually encounter reduces latency
proportionally (1,940 to 100 languages cuts the scoring loop by roughly 19x).

```python
model = load_model("unilid-1940-calibrated.unilid",
                   languages=["eng_Latn", "deu_Latn", "fra_Latn"])
```

or write a smaller model file once and load it anywhere:

```bash
unilid-calibrate subset unilid-1940-calibrated.unilid -o european.unilid \
    --langs-file my_languages.txt
```

Base (uncalibrated) predictions on a subset are exact: the decision rule is an
argmax over the included languages. Under calibrated inference the
re-examination thresholds are carried over from the full model by default;
because removing languages can only raise a prediction's margin, carried
thresholds make re-examination fire at most as often as calibrated, and
otherwise behave identically. Re-estimating the thresholds against the subset
model is optional: pass `--recalibrate <corpus_dir>` (expects
`<corpus_dir>/<lang>_train.txt` per retained low-resource language) to run the
per-language threshold recipe against the subset.

## Add your own language

A new language can be added to an existing calibrated model without retraining
anything else. A complete runnable walkthrough on toy data ships with the
repository: `bash examples/add_language/run_example.sh` builds a small
calibrated model from scratch, adds a fourth language, and compares predictions
before and after (see
[examples/add_language/README.md](examples/add_language/README.md)).

```bash
unilid-add-language unilid-1940-calibrated.unilid xyz_Latn xyz_train.txt -o extended.unilid
```

```python
from unilid import add_language
summary = add_language("unilid-1940-calibrated.unilid", "xyz_Latn",
                       "xyz_train.txt", "extended.unilid")
```

This trains the new language's token distribution over the model's existing
vocabulary (fixed-vocabulary EM; the default `sp` method needs the SentencePiece
CLI, `--method em` is pure Python), appends its weight row (existing rows are
copied byte-identically), and calibrates it from its own data alone: a
re-examination threshold is estimated from its own training lines when it has
fewer than 18,000 of them, and it becomes a replacement candidate only with at
least 100,000. Three caveats are documented in
[examples/add_language/README.md](examples/add_language/README.md) and the
paper: the four-language high-entropy group is not recomputed (the one
non-incremental piece of the calibration), existing languages' thresholds are
kept, and the 100,000-line candidate requirement is a per-artifact constant
that an uncapped corpus deployment should choose deliberately.

`unilid-calibrate` manages the calibration artifact directly:

| Subcommand | Does |
|------------|------|
| `export MODEL -o cal.json` | Write the bundled calibration to standalone JSON |
| `bundle MODEL cal.json -o OUT` | Attach a calibration to a version-1 model (writes version 2) |
| `estimate MODEL LANG train.txt -o OUT` | Re-estimate one language's threshold |
| `subset MODEL -o OUT --langs ... [--recalibrate DIR]` | Restrict to a language subset |

## Training a model

`train.py` is the single entry point for training a model from scratch. It
loads data, splits per-language corpus files, trains a shared base tokenizer,
re-estimates per-language token probabilities, and saves everything plus a
`training_summary.json`. A model trained this way is a base model with no
calibration artifact; load it with `calibrated=False` (deriving a full
calibration is described in the paper's development-protocol appendix, and
[examples/add_language/build_calibration.py](examples/add_language/build_calibration.py)
shows the bootstrap on a small model).

The default per-language method (`sp`) requires the forked SentencePiece CLI
(both the pip package from the `[train]` extra and the compiled binary):

```bash
cd sentencepiece && mkdir -p build && cd build
cmake .. && make -j$(nproc) && sudo make install   # or -DCMAKE_INSTALL_PREFIX=$HOME/.local
cd ../..
```

Verify with `spm_train --help`. The pure-Python `soft`/`hard` methods need no
binary.

### Input formats

Provide exactly one of `--fasttext`, `--wili-dir`, or `--tsv`.

`--fasttext FILE`: one sample per line, `__label__` plus the language code, a
space, then the text (the format of [GlotLID](https://github.com/cisnlp/GlotLID)
and [FastText LID](https://fasttext.cc/docs/en/language-identification.html)):

```
__label__eng Hello world
__label__deu Hallo Welt
```

`--wili-dir DIR`: a directory with aligned `x_train.txt` (texts) and
`y_train.txt` (language codes), as in
[WiLI-2018](https://zenodo.org/record/841984).

`--tsv FILE`: tab-separated `id`, `lang`, `text`, as in
[Tatoeba](https://tatoeba.org/en/downloads)'s `sentences.csv`.

### Examples

```bash
# GlotLID-format data, 100K vocab, byte-level
python train.py --fasttext data/glotlid/train.txt --vocab-size 100000 \
    --byte-level --max-base-samples-per-lang 10000

# Custom EM for both base and languages (no SentencePiece binary needed)
python train.py --wili-dir data/wili/ --vocab-size 50000 \
    --base-training-method soft --per-lang-counts-method hard

# Resume a partially completed run
python train.py --fasttext data/train.txt --vocab-size 100000 \
    --results-dir results_100k --reuse-corpus --reuse-base --skip-existing-langs

# Seed the vocabulary from an existing tokenizer (e.g. LLaMA)
python train.py --fasttext data/train.txt --vocab-size 100000 \
    --initial-vocab path/to/llama/tokenizer.json
```

### All flags

**Input** (mutually exclusive, one required): `--fasttext FILE`,
`--wili-dir DIR`, `--tsv FILE`.

**Training**:

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-size` | `100000` | Vocabulary size |
| `--base-training-method` | `hf` | Base tokenizer training: `hf` (HuggingFace UnigramTrainer), `bpe` (HuggingFace BPE), `soft` (custom soft-EM), `hard` (custom hard-EM) |
| `--per-lang-counts-method` | `sp` | Per-language probability estimation: `sp` (SentencePiece EM, C implementation), `soft` (custom soft-EM), `hard` (custom hard-EM). All use EM; `sp` is fastest. |
| `--byte-level / --char-level` | `--byte-level` | Byte-level or character-level tokenization |
| `--initial-vocab FILE` | None | Seed vocabulary from an existing tokenizer (`.json`) or a text file (one token per line) |
| `--seed` | `42` | Random seed |
| `--max-samples` | None | Limit total input lines (for debugging) |

**Sampling**:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-base-samples-per-lang` | `10000` | Max samples per language for base tokenizer training |
| `--max-lang-samples-per-lang` | None | Cap per-language training data |
| `--shared-samples-per-lang` | None | Use the same subsample for base and per-language training |

**Orchestration**:

| Flag | Default | Description |
|------|---------|-------------|
| `--lang-batch-size` | `10` | Languages trained per batch (controls memory) |
| `--results-dir` | `results_{K}k` | Output directory |
| `--corpus-dir` | None | Reuse a pre-split corpus directory |
| `--base-tokenizer-path` | None | Path to load/save the base tokenizer |
| `--reuse-corpus / --no-reuse-corpus` | `True` | Reuse existing corpus files if found |
| `--reuse-base / --no-reuse-base` | `True` | Reuse an existing base tokenizer if found |
| `--skip-existing-langs / --no-skip-existing-langs` | `True` | Skip languages with existing tokenizers |

### Output structure

```
results_100k/
  training_summary.json           # Full training config, timing, file paths
  corpus/                         # Per-language text files
  corpus_base_sampled/            # Subsampled files for base training
  tokenizers/
    langspec_base_tokenizer.json  # Shared base tokenizer
    langspec_sp_eng.tokenizer.json
    ...                           # Per-language tokenizers with metadata
```

## The .unilid model format

A trained model packs into a single binary file (one file instead of hundreds
of JSON tokenizers, roughly 16x smaller, memory-mapped weights):

```
Header (32 bytes):
  magic "UNILID\x00\x00"; version uint32 (1 = base model; 2 = calibration
  appended); num_langs, vocab_size, base_tok_len, langs_len (uint32); 4
  reserved bytes
Body:
  base_tokenizer JSON; langs JSON array; weights float32[num_langs * vocab_size]
Version 2 only, after the weights:
  calibration_len uint64 little-endian; calibration JSON
```

The stored weights are always the base (unclamped) matrix; the unseen-token
constant is applied at load time when calibrated inference is active, so one
file serves both modes. Package version 0.1.0 rejects version-2 files with an
error rather than silently returning base predictions.

Pack, unpack, and bundle:

```bash
unilid-convert results_100k -o my_model.unilid                          # pack (version 1)
unilid-convert results_100k -o my_model.unilid --calibration cal.json   # pack + bundle (version 2)
unilid-convert model.unilid --unpack                                    # unpack (writes calibration.json for version 2)
python convert.py ...                                                   # same pack/unpack arguments
```

`unpack_unilid` writes the bundled calibration to `calibration.json` beside the
tokenizers, and `save_unilid` refuses to repack a directory containing
`calibration.json` unless the calibration is passed explicitly, so a calibrated
model cannot be downgraded to version 1 by accident.

## Evaluation and bulk prediction

`eval.py` streams predictions over a text file or evaluates against labeled
data. It scores in base (uncalibrated) mode and refuses a version-2 model file
unless `--base` is passed; calibrated predictions are produced through the
Python API.

```bash
# Predictions to TSV (text \t lang \t score) or JSONL
python eval.py --model model.unilid --input texts.txt --output predictions.tsv
python eval.py --model model.unilid --input texts.txt --output out.jsonl --format jsonl

# Metrics (accuracy, macro F1/precision/recall, throughput) on fastText-format labels
python eval.py --model model.unilid --input test.txt --fasttext --output results.json
python eval.py --model model.unilid --input test.txt --fasttext --lang-only  # ignore scripts
```

## Python API

Inference:

```python
from unilid import load_model

model = load_model("unilid-1940-calibrated.unilid")            # calibrated (default)
model = load_model("model.unilid", calibrated=False)           # base behavior
model = load_model("model.unilid", calibration="cal.json")     # standalone calibration file
model = load_model("model.unilid", languages=["eng_Latn", "deu_Latn"])  # subset

lang, tokens, score = model.predict("Hello world")
results = model.predict_batch(["Hello world", "Hallo Welt"])

model.num_languages, model.langs        # language inventory
model.calibrated                        # True under the default
model.calibration                       # the loaded Calibration (or None)
model.last_reexamination_stats          # per-group counts of the last batch

tok = model.tokenizer                   # the shared base tokenizer
tok.encode("Hello world").tokens
```

`predict` and `predict_batch` return `(lang, tokens, score)`; under calibrated
inference, `tokens` and `score` are the segmentation and score under the
finally predicted language.

Training and customization:

```python
from unilid import (
    add_language,                       # add one language to a calibrated model
    train_standard_tokenizer,           # one shared tokenizer over languages
    train_language_specific_tokenizer,  # base + per-language distributions
    train_tokmix,                       # merge per-language vocabularies
    save_unilid, unpack_unilid,         # pack/unpack model files
    Calibration, estimate_tau,          # calibration artifact + threshold recipe
)
```

The training helpers need the `[train]` extra; they are imported lazily so a
prediction-only install works without it. `CorpusTokenizer`
(`python -m unilid.corpus_tokenizer`) batch-tokenizes corpora with existing
tokenizers. Lower-level trainer classes (`StandardUnigramLMTokenizer`,
`LanguageSpecificUnigramLMTokenizer`, `EMUnigramTrainer`) expose `encode_lang`,
`best_language_encode`, and the EM internals.

## Reproducing the paper's results

[REPRODUCING.md](REPRODUCING.md) maps the released files and modes to the
paper's reported rows, describes the calibrated-inference mechanism in detail,
lists the measured effects on GlotLID-C, UDHR, and CommonLID, and states the
evaluation conventions. The paper is the specification of record.

## Experimental: forward marginalization

`predict` and `predict_batch` accept `forward=True` in base mode, which scores
each language by marginalizing over all segmentations (log-sum-exp) instead of
taking the single best segmentation:

```python
lang, tokens, score = base_model.predict(text, forward=True)   # ~2x slower
```

This mode exists for experimentation; the two decodings give nearly identical
accuracy in practice and there is no established use case where forward
scoring is preferable, so Viterbi (the default) is recommended. `forward=True`
is defined for the base model only and raises under calibrated inference: the
calibration thresholds are percentiles of Viterbi margins, so marginalized
scores would not match them.

## Project structure

```
UNILID/
  train.py                         # Training entry point (CLI)
  eval.py                          # Bulk prediction / evaluation (CLI, base inference)
  convert.py                       # Pack/unpack .unilid files (CLI)
  REPRODUCING.md                   # Paper-results reproduction guide
  sentencepiece/                   # Forked SentencePiece (git submodule)
  tokenizers/                      # Forked HF tokenizers with fast inference (git submodule)
  examples/add_language/           # Runnable add-language walkthrough on toy data
  tests/                           # Unit + integration tests (pytest)
  unilid/
    model_io.py                    # .unilid format I/O, UnilidModel
    calibration.py                 # Calibration artifact, clamp, gate + walk, thresholds
    add_language.py                # add_language() and unilid-add-language
    calibrate_cli.py               # unilid-calibrate (export/bundle/estimate/subset)
    api.py                         # Training convenience functions
    trainers/                      # Base + per-language trainers (EM, SentencePiece)
    algorithms/                    # Viterbi, forward-backward, EM accumulation
    constants.py, encoding.py, pruning.py, corpus_tokenizer.py, ...
```
