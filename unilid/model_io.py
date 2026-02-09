"""
UNILID model I/O utilities.

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
"""
import gc
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer

__version__ = 1
MAGIC = b"UNILID\x00\x00"
HEADER_FMT = "<8sIIIII4x"  # magic, version, num_langs, vocab_size, base_tok_len, langs_len, padding
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def _get_vocab_with_scores(tok: Tokenizer) -> List[Tuple[str, float]]:
    """Extract vocab with scores from HF Unigram tokenizer."""
    state = tok.model.__getstate__()
    attributes = json.loads(state.decode("utf-8"))
    return attributes["vocab"]


def save_unilid(model_dir: Path, output_path: Path) -> Path:
    """
    Convert tokenizers directory to .unilid format.

    Args:
        model_dir: Directory containing tokenizers/ folder
        output_path: Output .unilid file path

    Returns:
        Path to saved .unilid file
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path).with_suffix(".unilid")

    tok_dir = model_dir / "tokenizers"
    if not tok_dir.exists():
        tok_dir = model_dir

    # Find base tokenizer
    base_path = tok_dir / "langspec_base_tokenizer.json"
    if not base_path.exists():
        lang_files = sorted(tok_dir.glob("langspec_*.tokenizer.json"))
        if lang_files:
            base_path = lang_files[0]
        else:
            raise FileNotFoundError(f"No tokenizers found in {tok_dir}")

    print(f"Loading base tokenizer: {base_path.name}")
    base_tok = Tokenizer.from_file(str(base_path))
    ref_vocab = base_tok.get_vocab()
    V = len(ref_vocab)
    base_tok_bytes = base_tok.to_str().encode("utf-8")

    # Find language tokenizers
    lang_files = sorted(tok_dir.glob("langspec_soft_*.tokenizer.json"))
    if not lang_files:
        lang_files = sorted(tok_dir.glob("langspec_sp_*.tokenizer.json"))

    print(f"Found {len(lang_files)} language tokenizers, vocab size {V}")

    # Extract weights
    very_neg = np.float32(-1e30)
    langs = []
    weights = np.empty((len(lang_files), V), dtype=np.float32)

    for i, lang_path in enumerate(lang_files):
        lang_code = lang_path.stem
        for prefix in ["langspec_soft_", "langspec_sp_"]:
            lang_code = lang_code.replace(prefix, "")
        lang_code = lang_code.replace(".tokenizer", "")

        tok = Tokenizer.from_file(str(lang_path))
        vocab_scores = _get_vocab_with_scores(tok)
        scores = {token: score for token, score in vocab_scores}

        weights[i, :] = very_neg
        for token, rid in ref_vocab.items():
            lp = scores.get(token)
            if lp is not None:
                weights[i, rid] = np.float32(lp)

        langs.append(lang_code)
        del tok, scores

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(lang_files)} languages")
            gc.collect()

    print(f"Weights shape: {weights.shape}")
    langs_bytes = json.dumps(langs).encode("utf-8")

    # Write file
    with open(output_path, "wb") as f:
        # Header
        header = struct.pack(
            HEADER_FMT,
            MAGIC,
            __version__,
            len(langs),
            V,
            len(base_tok_bytes),
            len(langs_bytes),
        )
        f.write(header)

        # Body
        f.write(base_tok_bytes)
        f.write(langs_bytes)
        f.write(weights.tobytes())

    # Report size
    orig_size = sum(f.stat().st_size for f in tok_dir.glob("*.json"))
    new_size = output_path.stat().st_size
    print(f"\nOriginal: {orig_size / 1e6:.1f} MB ({len(lang_files) + 1} files)")
    print(f"Packed: {new_size / 1e6:.1f} MB (1 file)")
    print(f"Ratio: {orig_size / new_size:.2f}x")
    print(f"Saved to: {output_path}")

    return output_path


def load_unilid(model_path: Path) -> Tuple[Tokenizer, np.ndarray, List[str]]:
    """
    Load .unilid model (memory-mapped weights for speed).

    Args:
        model_path: Path to .unilid file

    Returns:
        Tuple of (base_tokenizer, weights_array, langs_list)
    """
    model_path = Path(model_path)

    with open(model_path, "rb") as f:
        # Read header
        header = f.read(HEADER_SIZE)
        magic, version, num_langs, vocab_size, base_tok_len, langs_len = struct.unpack(HEADER_FMT, header)

        if magic != MAGIC:
            raise ValueError(f"Invalid .unilid file (bad magic)")
        if version > __version__:
            raise ValueError(f"Unsupported version {version}")

        # Read body
        base_tok_bytes = f.read(base_tok_len)
        langs_bytes = f.read(langs_len)
        weights_offset = f.tell()

    # Parse
    base_tok = Tokenizer.from_str(base_tok_bytes.decode("utf-8"))
    langs = json.loads(langs_bytes.decode("utf-8"))

    # Memory-map weights for fast loading
    weights = np.memmap(
        model_path,
        dtype=np.float32,
        mode="r",
        offset=weights_offset,
        shape=(num_langs, vocab_size),
    )

    print(f"Loaded {len(langs)} languages, vocab size {vocab_size}")
    return base_tok, weights, langs


def unpack_unilid(model_path: Path, output_dir: Path) -> Path:
    """
    Unpack .unilid file to tokenizers directory.

    Args:
        model_path: Path to .unilid file
        output_dir: Output directory for tokenizers

    Returns:
        Path to output directory
    """
    from tqdm import tqdm

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load .unilid
    base_tok, weights, langs = load_unilid(model_path)
    ref_vocab = base_tok.get_vocab()
    vocab_list = sorted(ref_vocab.items(), key=lambda x: x[1])  # Sort by id

    # Save base tokenizer
    base_path = output_dir / "langspec_base_tokenizer.json"
    base_tok.save(str(base_path))
    print(f"Saved base tokenizer: {base_path}")

    # Get base vocab with scores for template
    base_vocab_scores = _get_vocab_with_scores(base_tok)
    token_to_idx = {token: i for i, (token, score) in enumerate(base_vocab_scores)}

    # Create per-language tokenizers
    print(f"Unpacking {len(langs)} language tokenizers...")

    # Get base state template (remove 'type' as it's not a constructor arg)
    base_state = json.loads(base_tok.model.__getstate__().decode("utf-8"))
    base_state.pop("type", None)

    for lang_idx, lang in enumerate(tqdm(langs, desc="Unpacking")):
        # Build new vocab with this language's scores
        new_vocab = []
        for token, tok_id in vocab_list:
            score = float(weights[lang_idx, tok_id])
            new_vocab.append((token, score))

        # Create state with new vocab
        state = base_state.copy()
        state["vocab"] = new_vocab

        # Create new tokenizer with modified model
        lang_tok = Tokenizer.from_str(base_tok.to_str())
        lang_tok.model = lang_tok.model.__class__(**state)

        # Save
        lang_path = output_dir / f"langspec_sp_{lang}.tokenizer.json"
        lang_tok.save(str(lang_path))

    print(f"\nUnpacked to: {output_dir}")
    print(f"  - 1 base tokenizer")
    print(f"  - {len(langs)} language tokenizers")
    return output_dir


class UnilidModel:
    """High-level model wrapper for inference."""

    def __init__(self, model_path):
        """
        Load model from .unilid file or tokenizers directory.

        Args:
            model_path: Path to .unilid file or directory containing tokenizers/
        """
        model_path = Path(model_path)

        if model_path.suffix == ".unilid" or (model_path.is_file() and str(model_path).endswith(".unilid")):
            self._load_from_unilid(model_path)
        elif model_path.is_dir():
            self._load_from_dir(model_path)
        else:
            raise ValueError(f"Unknown model format: {model_path}. Expected .unilid file or directory.")

    def _load_from_unilid(self, model_path: Path):
        """Load from .unilid file."""
        base_tok, weights, self.langs = load_unilid(model_path)

        print("Pushing weights to Rust cache...")
        base_tok.model.set_weight_sets(np.array(weights).tolist())

        self.tokenizer = base_tok
        self.model = base_tok.model
        self.pre_tok = base_tok.pre_tokenizer
        self.normalizer = base_tok.normalizer
        self._lang_to_idx = {lang: i for i, lang in enumerate(self.langs)}

        del weights
        gc.collect()
        print("Model ready")

    @classmethod
    def from_dir(cls, model_dir: Path) -> "UnilidModel":
        """Load directly from tokenizers directory (deprecated, use __init__ instead)."""
        return cls(model_dir)

    def _load_from_dir(self, model_dir: Path):
        """Load from tokenizers directory (streaming)."""
        from tqdm import tqdm

        model_dir = Path(model_dir)
        tok_dir = model_dir / "tokenizers"
        if not tok_dir.exists():
            tok_dir = model_dir

        base_path = tok_dir / "langspec_base_tokenizer.json"
        if not base_path.exists():
            lang_files = sorted(tok_dir.glob("langspec_*.tokenizer.json"))
            if lang_files:
                base_path = lang_files[0]

        base_tok = Tokenizer.from_file(str(base_path))
        ref_vocab = base_tok.get_vocab()
        V = len(ref_vocab)

        lang_files = sorted(tok_dir.glob("langspec_soft_*.tokenizer.json"))
        if not lang_files:
            lang_files = sorted(tok_dir.glob("langspec_sp_*.tokenizer.json"))

        print(f"Loading {len(lang_files)} languages, vocab size {V}")

        # Pre-allocate numpy array (float32 = 4 bytes vs Python float = 8+ bytes)
        very_neg = np.float32(-1e30)
        langs = []
        weights = np.full((len(lang_files), V), very_neg, dtype=np.float32)

        # Load in batches to minimize peak memory from tokenizer objects
        BATCH_SIZE = 20
        for batch_start in tqdm(range(0, len(lang_files), BATCH_SIZE), desc="Loading"):
            batch_end = min(batch_start + BATCH_SIZE, len(lang_files))
            batch_files = lang_files[batch_start:batch_end]

            # Load batch of tokenizers, extract weights
            for i, lang_path in enumerate(batch_files):
                idx = batch_start + i
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
                        weights[idx, rid] = np.float32(lp)

                langs.append(lang_code)
                del tok, scores

            # Clean up after each batch
            gc.collect()

        print("Pushing weights to Rust cache...")
        base_tok.model.set_weight_sets(weights.tolist())

        self.tokenizer = base_tok
        self.model = base_tok.model
        self.pre_tok = base_tok.pre_tokenizer
        self.normalizer = base_tok.normalizer
        self.langs = langs
        self._lang_to_idx = {lang: i for i, lang in enumerate(langs)}

        del weights
        gc.collect()
        print("Model ready")

    def preprocess(self, text: str) -> Optional[str]:
        """Apply normalizer and pretokenizer."""
        if self.normalizer:
            text = self.normalizer.normalize_str(text)
        if self.pre_tok:
            pre = self.pre_tok.pre_tokenize_str(text)
            if not pre:
                return None
            return "".join(piece[0] for piece in pre)
        return text if text else None

    def predict(self, text: str) -> Tuple[str, List[str], float]:
        """Predict language for a single text."""
        pt = self.preprocess(text)
        if not pt:
            return None, [], float("-inf")
        idx, tokens, score = self.model.best_of_cached_weight_sets(pt)
        return self.langs[idx], tokens, score

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, List[str], float]]:
        """Predict languages for multiple texts (Rayon parallel)."""
        preprocessed, valid_idx = [], []
        for i, text in enumerate(texts):
            pt = self.preprocess(text)
            if pt:
                preprocessed.append(pt)
                valid_idx.append(i)

        if not preprocessed:
            return [(None, [], float("-inf"))] * len(texts)

        batch_results = self.model.best_of_cached_weight_sets_batch(preprocessed)

        results = [(None, [], float("-inf"))] * len(texts)
        for j, (idx, tokens, score) in enumerate(batch_results):
            results[valid_idx[j]] = (self.langs[idx], tokens, score)
        return results

    @property
    def num_languages(self) -> int:
        return len(self.langs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert model to .unilid format")
    parser.add_argument("model_dir", type=Path, help="Model directory")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.model_dir / "model.unilid")
    save_unilid(args.model_dir, output)
