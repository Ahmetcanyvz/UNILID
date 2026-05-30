# UNILID

Fast multilingual language identification using unigram language models. Trains a shared vocabulary across languages, then re-estimates per-language token probabilities via EM (using either SentencePiece's C implementation or a custom Python implementation). Supports byte-level tokenization, vocabulary seeding from external tokenizers, and Rayon-parallel batch inference.

## Quick Start

```python
from unilid import load_model

model = load_model("model.unilid")
lang, tokens, score = model.predict("The quick brown fox jumps over the lazy dog.")
print(lang)  # 'eng'

# Batch prediction (parallel)
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
]
results = model.predict_batch(texts)
for text, (lang, tokens, score) in zip(texts, results):
    print(f"{lang}: {text[:50]}...")
```

### Decoding: Viterbi vs. Forward Marginalization

By default, `predict` / `predict_batch` use **Viterbi** decoding, scoring each language by its single most likely segmentation. You can switch to **exact marginalization** via the forward algorithm, which sums probabilities over all valid segmentations:

```python
# Viterbi (default) — fastest, recommended
lang, tokens, score = model.predict(text)

# Forward marginalization — exact p(s | ℓ), ~2x slower
lang, tokens, score = model.predict(text, forward=True)
results = model.predict_batch(texts, forward=True)
```

The two modes give nearly identical accuracy in practice; Viterbi is recommended unless you need exact marginal likelihoods. The `eval_glotlid.py` and `eval_wili.py` scripts expose the same option via `--forward`.

## Pre-trained Models

| Model | Languages | Training Data | Download |
|-------|-----------|---------------|----------|
| unilid-1940 | 1940 language-script combinations | 60M samples | [Download](https://polybox.ethz.ch/index.php/s/Kbb9TWkSSgQ8yoS) |

## Installation

```bash
# Clone with submodules (required for custom tokenizers)
git clone --recurse-submodules https://github.com/Ahmetcanyvz/UNILID.git && cd UNILID

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install unilid
pip install -e .

# For training capabilities:
pip install -e ".[train]"

# Build custom tokenizers (REQUIRED for inference)
pip uninstall tokenizers -y
pip install maturin
cd tokenizers/bindings/python
unset CONDA_PREFIX  # if using conda
maturin develop --release
cd ../../..
```

If already cloned without submodules:
```bash
git submodule update --init --recursive
```

**Note:** The custom tokenizers library is required for inference. It provides Rust-accelerated parallel batch processing via Rayon. Standard HuggingFace tokenizers will NOT work.

### SentencePiece Setup

The default language-specific training method (`--per-lang-counts-method sp`) requires a [forked SentencePiece](https://github.com/cimeister/sentencepiece.git) CLI binary. The fork (branch `fixed-vocab-em`) adds fixed-vocabulary EM re-estimation support.

You need **both** the Python package (installed above via pip) and the compiled CLI:

```bash
git submodule update --init --recursive
cd sentencepiece
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..
```

To install without sudo:
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install
# Add to your shell profile:
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
```

Verify: `spm_train --help` should print usage info.

### Verify Installation

```python
from tokenizers import Tokenizer
t = Tokenizer.from_file("path/to/tokenizer.json")
assert hasattr(t.model, "set_weight_sets"), "Custom tokenizers not installed!"
```

The custom tokenizers provides:
- `set_weight_sets()` - Cache per-language weights in Rust
- `best_of_cached_weight_sets()` - Single text inference
- `best_of_cached_weight_sets_batch()` - Rayon-parallel batch inference (~10x faster)

## Training

`train.py` is the single entry point for all training. It:
1. Loads data (FastText, WILI, or TSV format)
2. Splits into per-language corpus files
3. Trains a shared base tokenizer (HuggingFace UnigramTrainer)
4. Re-estimates per-language token probabilities (SentencePiece or EM)
5. Saves all tokenizers and a `training_summary.json`

### Input Formats

Provide exactly one of `--fasttext`, `--wili-dir`, or `--tsv`.

#### `--fasttext FILE`

One sample per line. Each line starts with `__label__` followed by the language code, a space, then the text. Used by [GlotLID](https://github.com/cisnlp/GlotLID) and [FastText LID](https://fasttext.cc/docs/en/language-identification.html).

```
__label__eng Hello world
__label__deu Hallo Welt
__label__fra Bonjour le monde
```

#### `--wili-dir DIR`

Points to a directory containing two files with aligned lines: `x_train.txt` (one text per line) and `y_train.txt` (one language code per line). Used by [WiLI-2018](https://zenodo.org/record/841984).

```
x_train.txt:          y_train.txt:
Hello world           eng
Hallo Welt            deu
Bonjour le monde      fra
```

#### `--tsv FILE`

Tab-separated file with three columns: `id`, `lang`, `text`. Used by [Tatoeba](https://tatoeba.org/en/downloads) (`sentences.csv`).

```
1234	eng	Hello world
5678	deu	Hallo Welt
9012	fra	Bonjour le monde
```

### Examples

Train on GlotLID (FastText format), 100K vocab, byte-level:
```bash
python train.py \
    --fasttext data/glotlid/train.txt \
    --vocab-size 100000 \
    --byte-level \
    --max-base-samples-per-lang 10000
```

Train on WILI, 50K vocab, custom EM for both base and languages:
```bash
python train.py \
    --wili-dir data/wili/ \
    --vocab-size 50000 \
    --base-training-method soft \
    --per-lang-counts-method hard
```

Resume a partially completed run (skips existing tokenizers):
```bash
python train.py \
    --fasttext data/train.txt \
    --vocab-size 100000 \
    --results-dir results_100k \
    --reuse-corpus \
    --reuse-base \
    --skip-existing-langs
```

Seed from an existing tokenizer (e.g. LLaMA):
```bash
python train.py \
    --fasttext data/train.txt \
    --vocab-size 100000 \
    --initial-vocab path/to/llama/tokenizer.json
```

### All Flags

**Input** (mutually exclusive, one required):

| Flag | Description |
|------|-------------|
| `--fasttext FILE` | FastText `__label__` format |
| `--wili-dir DIR` | WILI directory with `x_train.txt` + `y_train.txt` |
| `--tsv FILE` | Tatoeba tab-separated format |

**Training**:

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab-size` | `100000` | Vocabulary size |
| `--base-training-method` | `hf` | Base tokenizer training: `hf` (HuggingFace UnigramTrainer), `bpe` (HuggingFace BPE), `soft` (custom soft-EM), `hard` (custom hard-EM) |
| `--per-lang-counts-method` | `sp` | Per-language probability estimation: `sp` (SentencePiece EM, C implementation), `soft` (custom soft-EM), `hard` (custom hard-EM). All use EM; `sp` is fastest. |
| `--byte-level / --char-level` | `--byte-level` | Byte-level or character-level tokenization |
| `--initial-vocab FILE` | None | Seed vocabulary from existing tokenizer (`.json`) or text file (one token per line) |
| `--seed` | `42` | Random seed |
| `--max-samples` | None | Limit total input lines (for debugging) |

**Sampling**:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-base-samples-per-lang` | `10000` | Max samples per language for base tokenizer training |
| `--max-lang-samples-per-lang` | None | Cap per-language training data |
| `--shared-samples-per-lang` | None | Use same subsample for both base and per-language training |

**Orchestration**:

| Flag | Default | Description |
|------|---------|-------------|
| `--lang-batch-size` | `10` | Languages trained per batch (controls memory) |
| `--results-dir` | `results_{K}k` | Output directory |
| `--corpus-dir` | None | Reuse pre-split corpus directory |
| `--base-tokenizer-path` | None | Path to load/save base tokenizer (loads if exists and `--reuse-base`) |
| `--reuse-corpus / --no-reuse-corpus` | `True` | Reuse existing corpus files if found |
| `--reuse-base / --no-reuse-base` | `True` | Reuse existing base tokenizer if found |
| `--skip-existing-langs / --no-skip-existing-langs` | `True` | Skip languages with existing tokenizers |

### Output Structure

```
results_100k/
  training_summary.json           # Full training config, timing, file paths
  corpus/                         # Per-language text files
    eng_train.txt
    deu_train.txt
    ...
  corpus_base_sampled/            # Subsampled files for base training
  tokenizers/
    langspec_base_tokenizer.json  # Shared base tokenizer
    langspec_sp_eng.tokenizer.json
    langspec_sp_deu.tokenizer.json
    ...                           # Per-language tokenizers with metadata
```

## Model Format (.unilid)

For efficient storage and fast loading, trained models can be converted to a single `.unilid` binary file.

### Format Specification

```
.unilid format (custom binary):
  Header (32 bytes):
    - magic: 8 bytes "UNILID\x00\x00"
    - version: uint32
    - num_langs: uint32
    - vocab_size: uint32
    - base_tok_len: uint32
    - langs_len: uint32
    - reserved: 4 bytes
  Body:
    - base_tokenizer JSON (base_tok_len bytes, utf-8)
    - langs JSON array (langs_len bytes, utf-8)
    - weights: float32[num_langs * vocab_size]
```

### Benefits

- **Single file**: One `.unilid` file vs hundreds of JSON files
- **Fast loading**: Memory-mapped weights for instant access
- **Compact**: ~16x smaller than raw JSON tokenizers
- **Memory efficient**: Batch loading with periodic GC

### Converting Models

Pack tokenizers directory to `.unilid`:

```bash
python convert.py results_100k
python convert.py results_100k -o my_model.unilid
```

Unpack `.unilid` back to tokenizers directory:

```bash
python convert.py model.unilid --unpack
python convert.py model.unilid --unpack -o tokenizers_dir/
```

Or via Python:

```python
from unilid import save_unilid, unpack_unilid

# Pack
save_unilid("results_100k", "model.unilid")

# Unpack
unpack_unilid("model.unilid", "tokenizers_dir/")
```

## Prediction & Evaluation

The `eval.py` script supports two modes:

### Prediction Mode (default)

Stream predictions to output file with minimal memory usage:

```bash
# Predict on plain text file (one text per line)
python eval.py --model model.unilid --input texts.txt --output predictions.tsv

# JSONL output format
python eval.py --model model.unilid --input texts.txt --output predictions.jsonl --format jsonl

# Custom batch size
python eval.py --model model.unilid --input texts.txt --output out.tsv --batch-size 5000
```

Output formats:
- **TSV**: `text\tlang\tscore` (default)
- **JSONL**: `{"text": "...", "lang": "...", "score": ...}`

### Parallelism

Batch inference uses Rayon (Rust) which defaults to all available CPU cores. Control with:

```bash
# Limit to 4 threads
RAYON_NUM_THREADS=4 python eval.py --model model.unilid --input texts.txt --output out.tsv

# Single-core (bypasses Rayon entirely)
python eval.py --model model.unilid --input texts.txt --output out.tsv --single-core
```

### Evaluation Mode (--fasttext)

Evaluate on labeled fastText-format files and compute metrics:

```bash
# Evaluate on fastText-format file
python eval.py --model model.unilid --input test.txt --fasttext --output results.json

# Language-only labels (ignore script variants)
python eval.py --model model.unilid --input test.txt --fasttext --lang-only

# Single-core mode (for debugging/profiling)
python eval.py --model model.unilid --input test.txt --fasttext --single-core
```

FastText format (input):
```
__label__eng Hello world
__label__deu Hallo Welt
__label__fra Bonjour le monde
```

Output metrics:
- **Accuracy**: Overall correct predictions
- **Macro F1**: Average F1 across all languages
- **Macro Precision/Recall**: Average precision/recall
- **Samples/second**: Inference throughput

## Python API

### Training

```python
from unilid import (
    train_standard_tokenizer,
    train_language_specific_tokenizer,
    train_lang_tokenizers,
    train_tokmix,
)

# Train a standard tokenizer on all languages jointly
path = train_standard_tokenizer(
    corpus_info={"en": "en.txt", "de": "de.txt"},
    vocab_size=8000,
    em_mode="soft",       # "soft", "hard", "hf", or "bpe"
    byte_level=True,
)

# Train base + per-language tokenizers
paths = train_language_specific_tokenizer(
    corpus_info={"en": "en.txt", "de": "de.txt"},
    vocab_size=8000,
    reestimation_em_mode="soft",
    byte_level=True,
    base_tokenizer_path="base.json",
)

# Merge per-language tokenizers via TokMix
merged_path = train_tokmix(
    per_lang_paths={"en": "en.json", "de": "de.json"},
    k_final=16000,
    output_path="merged.json",
)
```

### Inference

#### Simple API (recommended)

```python
from unilid import load_model

# Load from .unilid file or tokenizers directory
model = load_model("model.unilid")
# or: model = load_model("results_100k/")

# Single text prediction
lang, tokens, score = model.predict("Hello world")
print(f"Language: {lang}, Score: {score}")

# Batch prediction (Rayon-parallel, ~10x faster)
texts = ["Hello world", "Hallo Welt", "Bonjour le monde"]
results = model.predict_batch(texts)
for text, (lang, tokens, score) in zip(texts, results):
    print(f"{text} -> {lang} ({score:.2f})")

# Access model info
print(f"Languages: {model.num_languages}")
print(f"Available: {model.langs[:5]}...")

# Access the base tokenizer for encoding/decoding
tok = model.tokenizer
encoded = tok.encode("Hello world")
print(encoded.ids)      # token IDs
print(encoded.tokens)   # token strings
decoded = tok.decode(encoded.ids)
```

#### Low-level API

```python
from unilid import StandardUnigramLMTokenizer, LanguageSpecificUnigramLMTokenizer

# Standard tokenizer
tok = StandardUnigramLMTokenizer()
tok.load("tokenizer.json")
ids = tok.encode("Hello world")
text = tok.decode(ids)

# Language-specific tokenizer
tok = LanguageSpecificUnigramLMTokenizer()
tok.load(
    base_path="base.json",
    language_paths={"en": "en.json", "de": "de.json"},
)

# Encode with a specific language model
enc = tok.encode_lang("Hallo Welt", "de")

# Find best language for a text
best_lang, tokens = tok.best_language_encode("Hello world")

# Batch: find best language for many texts
best_langs, encodings = tok.best_language_encode_batch(["Hello", "Hallo", "Bonjour"])
```

### Batch Corpus Tokenization

```python
from unilid import CorpusTokenizer

ct = CorpusTokenizer(
    corpus_info={
        "en": {"splits": {"train": "en_train.txt", "dev": "en_dev.txt"}},
        "de": {"splits": {"train": "de_train.txt", "dev": "de_dev.txt"}},
    },
    tokenizers={
        "my_tok": {
            "class": "standard",
            "path": "tokenizer.json",
            "tokenized_data_path": "output/my_tok/",
            "batch_size": 4096,
        }
    },
)
results = ct.tokenize_all()
```

CLI:
```bash
python -m unilid.corpus_tokenizer --config config.json --output_dir ./output
```

## Project Structure

```
UNILID/
  train.py                         # Unified training script (CLI)
  eval.py                          # Evaluation script (CLI)
  convert.py                       # Convert to .unilid format (CLI)
  sentencepiece/                   # Forked SentencePiece (git submodule)
  tokenizers/                      # Forked HF tokenizers with fast inference (git submodule)
  unilid/
    __init__.py
    api.py                         # High-level convenience functions
    model_io.py                    # .unilid format I/O, UnilidModel class
    constants.py                   # Special tokens, defaults
    token_encoding.py              # HF <-> SP token format conversion
    vocab_io.py                    # Vocabulary file I/O
    tokenizer_builder.py           # HF Tokenizer construction
    encoding.py                    # Text encoding dispatch
    pruning.py                     # Vocabulary pruning & TokMix merge
    metadata.py                    # Tokenizer metadata I/O
    corpus_tokenizer.py            # Batch corpus tokenization
    algorithms/
      viterbi.py                   # Viterbi segmentation
      forward_backward.py          # Forward-backward (log-space)
      loss.py                      # Token loss computation
      accumulate.py                # EM usage accumulation
    trainers/
      em_trainer.py                # Fixed-vocab EM trainer
      em_loop.py                   # EM iteration logic
      pruning_strategy.py          # Pruning score computation
      standard_trainer.py          # StandardUnigramLMTokenizer
      language_specific_trainer.py # LanguageSpecificUnigramLMTokenizer
```
