# UNILID

Fast multilingual language identification using unigram language models. Trains a shared vocabulary across languages, then re-estimates per-language token probabilities via EM (using either SentencePiece's C implementation or a custom Python implementation). Supports byte-level tokenization, vocabulary seeding from external tokenizers, Rayon-parallel batch inference, and calibrated inference (the default), whose per-language decision thresholds can be extended one language at a time.

## Quick Start

```python
from unilid import load_model

model = load_model("unilid-1940-calibrated.unilid")   # calibrated inference (default)
lang, tokens, score = model.predict("The quick brown fox jumps over the lazy dog.")
print(lang)  # 'eng_Latn'

# Batch prediction (parallel)
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
]
results = model.predict_batch(texts)
for text, (lang, tokens, score) in zip(texts, results):
    print(f"{lang}: {text[:50]}...")

# Base (uncalibrated) behavior of the original release:
base_model = load_model("unilid-1940-calibrated.unilid", calibrated=False)
```

### Decoding: Viterbi vs. Forward Marginalization

By default, `predict` / `predict_batch` use **Viterbi** decoding, scoring each language by its single most likely segmentation. In base (uncalibrated) mode you can switch to **exact marginalization** via the forward algorithm, which sums probabilities over all valid segmentations:

```python
# Viterbi (default; fastest, recommended)
lang, tokens, score = base_model.predict(text)

# Forward marginalization (exact log p(s | l), ~2x slower; base mode only)
lang, tokens, score = base_model.predict(text, forward=True)
results = base_model.predict_batch(texts, forward=True)
```

The two modes give nearly identical accuracy in practice; Viterbi is recommended unless you need exact marginal likelihoods. `forward=True` is defined for the base model only and raises under calibrated inference: the calibration thresholds are percentiles of Viterbi margins, so marginalized scores would not match them.

## Pre-trained Models

| Model | Languages | Training Data | Calibration | Download |
|-------|-----------|---------------|-------------|----------|
| unilid-1940-calibrated | 1940 language-script combinations | 60M samples | bundled (version-2 file) | [HuggingFace Hub](https://huggingface.co/cmeister/unilid-1940) |
| unilid-1940 | 1940 language-script combinations | 60M samples | none (version-1 file) | [polybox](https://polybox.ethz.ch/index.php/s/Kbb9TWkSSgQ8yoS) |

Both files contain the same trained model. The calibrated file additionally bundles the calibration artifact (per-language thresholds, training-line counts, and the constants; 160 KB).

## Calibrated inference

UNILID scores a text under every language and predicts the language with the highest score. Calibrated UNILID keeps this decision rule and adds two corrections, derived in the UNILID paper from an error analysis of the base model. This section explains what runs at prediction time; the paper documents how each constant was chosen and on which data.

**1. A shared constant for unseen tokens.** Each language's model assigns a log-probability to every token in the shared vocabulary, including tokens that never appeared in that language's training data. Training never leaves a token at probability zero: every token receives at least a minimum probability of 10^-12, and each language's probabilities are then normalized to sum to one. A side effect of that normalization is that the probability an unseen token ends up with differs from language to language, depending on how much training data the language has. Prediction compares scores across languages, so these differing unseen-token values act as a per-language offset added to every unseen token in the text: a text containing tokens unseen by two candidate languages is pushed toward the candidate whose unseen-token value happens to be higher, for no linguistic reason. The correction removes the offset. At load time, every unseen-token log-probability that lies above the shared constant c = -21 (in natural log units) is lowered to exactly c. Values already at or below c stay as trained, and the distributions are not renormalized afterwards.

**2. Re-examining close decisions that land in two groups of languages.** The margin of a prediction is the best language's score minus the second-best language's score; a small margin means the decision was close. The base model's errors concentrate in predictions INTO two groups: languages with fewer than 18,000 training samples, whose estimated distributions stay close to their uniform initialization and therefore give moderate probability to text from many languages, and a group of four larger languages whose distributions are unusually flat for their script (Scots, Banjar, Aragonese, West Flemish; the identification criteria are in the paper). When the predicted language belongs to either group and the margin falls below that language's own threshold, the prediction is re-examined: it moves to the highest-ranked of the candidates ranked 2 to 5 that has at least 100,000 training samples and a score within 21 natural-log units of the best score. If no candidate qualifies, the prediction stays unchanged. Each threshold is a percentile of the margins that language's own training lines produce, so a threshold can be computed for a new language without touching any other language. 26 of the 1,080 languages in the first group had fewer than 200 usable calibration lines; they receive no threshold and are never re-examined.

**Measured effect.** On the GlotLID-C test pool (45.4M lines, 1,940 languages), macro F1 rises from 0.929 (base) to 0.957 (calibrated). On UDHR, parallel data where every language has a similar number of test samples, macro F1 moves from 0.859 to 0.838: re-examination also moves some correct low-margin predictions, which lowers macro F1 on data where every language has similar sample counts. On CommonLID, an out-of-domain evaluation on web-crawled text with 109 labels, macrolanguage-aware accuracy (a prediction counts as correct when it matches the label at the language or macrolanguage level) rises from 0.845 to 0.860, while tag-level macro F1 moves from 0.723 to 0.715; calibration lowers the number of lines predicted as languages outside the 109-label set from 32,901 to 25,884, which raises accuracy, and re-examination moves some correct low-margin lines, which lowers tag-level macro F1. Together the three results locate where the gains appear: test data whose per-language line counts follow a collection's natural imbalance, over a label set that includes under-resourced languages. Use `calibrated=False` where the base behavior is wanted.

### Migration note for existing users

Version 0.2.0 makes calibrated inference the default. Loading a model without a calibration artifact (any version-1 `.unilid` file, including the original polybox release and self-trained models) with default arguments raises `UnilidCalibrationError`. Either download the calibrated model file, or pass `calibrated=False`:

```python
model = load_model("unilid-1940.unilid", calibrated=False)
```

Results published for the base model are reproduced with `calibrated=False` (`eval.py` refuses a calibrated model file unless `--base` is passed, so the two modes cannot be mixed up silently).

## Adding your own language

A new language can be added to an existing calibrated model without retraining anything else. The workflow trains the new language's token distribution over the model's existing vocabulary, appends it to the model file, and calibrates it from its own data alone:

```bash
unilid-add-language unilid-1940-calibrated.unilid xyz_Latn xyz_train.txt -o extended.unilid
```

or in Python:

```python
from unilid import add_language

summary = add_language("unilid-1940-calibrated.unilid", "xyz_Latn",
                       "xyz_train.txt", "extended.unilid")
```

What this does:

1. Trains the new language's distribution over the existing shared vocabulary (fixed-vocabulary EM, 20 rounds, minimum token probability 10^-12). The default method (`sp`) needs the forked SentencePiece binary (see Installation); `--method em` is a pure-Python alternative that has not been verified against the release's end-to-end chain.
2. Appends the new weight row at the last index. Existing rows are copied byte-identically.
3. Applies the calibration recipe to the new language only: its unseen-token values are subject to the shared constant c like every other language's; if it has fewer than 18,000 training lines, its re-examination threshold is estimated from its own training lines by the same recipe as the released thresholds (languages with fewer than 200 usable lines are excluded and never re-examined); with 18,000 or more lines it receives no threshold. It becomes a replacement candidate if and only if it has at least 100,000 training lines.

Three caveats, stated in full in the paper:

- The four-language high-entropy group is not recomputed. Its identification needs entropy statistics across all languages and a validation scoring pass, so it is the one non-incremental part of the calibration. The shipped group stays as released.
- Existing languages' thresholds are kept. A new language changes the margin distributions they were estimated on; re-deriving all thresholds requires the released calibration pipeline, not the incremental command.
- The 100,000-line requirement on replacement candidates coincides with the per-language training-data cap of the released model's corpus. For a corpus without that cap, the requirement is still "at least 100,000 training lines" (the `replacement_min_n` constant in the calibration artifact), and whether that value fits an uncapped deployment is a decision to make and document.

`unilid-calibrate` manages the calibration artifact directly: `export` writes the bundled artifact to JSON, `bundle` attaches a calibration JSON to a version-1 model file, and `estimate` re-estimates one language's threshold from its training file.

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

The default language-specific training method (`--per-lang-counts-method sp`, also used by `unilid-add-language`) requires a [forked SentencePiece](https://github.com/cimeister/sentencepiece.git) CLI binary. The fork (branch `fixed-vocab-em`) adds fixed-vocabulary EM re-estimation support.

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
assert hasattr(t.model, "tokens_of_cached_weight_set_batch"), "Rebuild the tokenizers submodule (calibrated inference needs it)"
```

The custom tokenizers provides:
- `set_weight_sets()` - Cache per-language weights in Rust
- `best_of_cached_weight_sets()` - Single text inference
- `best_of_cached_weight_sets_batch()` - Rayon-parallel batch inference (~10x faster)
- `top_k_of_cached_weight_sets_batch()` - Top-k languages with scores (used by calibrated inference)
- `tokens_of_cached_weight_set_batch()` - Segmentation under a specified language (used by calibrated inference)

## Training

`train.py` is the single entry point for all training. It:
1. Loads data (FastText, WILI, or TSV format)
2. Splits into per-language corpus files
3. Trains a shared base tokenizer (HuggingFace UnigramTrainer)
4. Re-estimates per-language token probabilities (SentencePiece or EM)
5. Saves all tokenizers and a `training_summary.json`

A model trained this way is a base model with no calibration artifact; load it with `calibrated=False`. Deriving a full calibration for a new model (thresholds for every language below 18,000 training lines and the high-entropy group) is described in the paper's development-protocol appendix.

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
    - version: uint32 (1 = base model; 2 = calibration section appended)
    - num_langs: uint32
    - vocab_size: uint32
    - base_tok_len: uint32
    - langs_len: uint32
    - reserved: 4 bytes
  Body:
    - base_tokenizer JSON (base_tok_len bytes, utf-8)
    - langs JSON array (langs_len bytes, utf-8)
    - weights: float32[num_langs * vocab_size]
  Version 2 only, after the weights:
    - calibration_len: uint64 little-endian
    - calibration JSON (calibration_len bytes, utf-8)
```

The stored weights are always the base (unclamped) matrix; the unseen-token constant is applied at load time when calibrated inference is active, so one file serves both modes. Package version 0.1.0 rejects version-2 files with an error rather than silently returning base predictions. `unpack_unilid` writes the bundled calibration to `calibration.json` beside the tokenizers; `save_unilid` refuses to repack a directory containing `calibration.json` unless the calibration is passed explicitly, so a calibrated model cannot be downgraded to version 1 by accident.

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
unilid-convert results_100k -o my_model.unilid --calibration calibration.json  # bundle a calibration
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

The `eval.py` script scores with base (uncalibrated) inference. Given a version-2 (calibrated) model file it exits with an error unless `--base` is passed; calibrated predictions are produced through the Python API (`model.predict_batch(...)`). It supports two modes:

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
model = load_model("unilid-1940-calibrated.unilid")            # calibrated (default)
model = load_model("model.unilid", calibrated=False)           # base behavior
model = load_model("model.unilid", calibration="cal.json")     # standalone calibration file

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
print(f"Calibrated: {model.calibrated}")            # True under the default
print(model.last_reexamination_stats)               # per-group counts of the last batch

# Access the base tokenizer for encoding/decoding
tok = model.tokenizer
encoded = tok.encode("Hello world")
print(encoded.ids)      # token IDs
print(encoded.tokens)   # token strings
decoded = tok.decode(encoded.ids)
```

`model.predict` and `model.predict_batch` return `(lang, tokens, score)` in both modes; under calibrated inference, `tokens` and `score` are the segmentation and score under the finally predicted language. `predict_normalized(_batch)` is defined for the base model only and raises under calibrated inference.

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
  eval.py                          # Evaluation script (CLI, base inference)
  convert.py                       # Convert to .unilid format (CLI)
  sentencepiece/                   # Forked SentencePiece (git submodule)
  tokenizers/                      # Forked HF tokenizers with fast inference (git submodule)
  tests/                           # Unit tests (pytest)
  unilid/
    __init__.py
    api.py                         # High-level convenience functions
    model_io.py                    # .unilid format I/O, UnilidModel class
    calibration.py                 # Calibration artifact, clamp, gate + walk, threshold estimation
    add_language.py                # add_language() and unilid-add-language CLI
    calibrate_cli.py               # unilid-calibrate CLI (export/bundle/estimate)
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
