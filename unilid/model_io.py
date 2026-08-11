"""
UNILID model I/O utilities.

.unilid format (custom binary):
  Header (32 bytes):
    - magic: 8 bytes "UNILID\x00\x00"
    - version: uint32 (1 = base model only; 2 = calibration section appended)
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
    - calibration JSON (calibration_len bytes, utf-8; schema in unilid.calibration)
"""
import gc
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer

from .calibration import Calibration, UnilidCalibrationError, \
    apply_unseen_token_constant, re_examine
from .constants import MISSING_TOKEN_FILL_LOG_PROB

# Highest container version this reader accepts; distinct from the per-write
# version, which depends on whether a calibration is bundled (a single shared
# constant would silently change every base-model write when raised).
FORMAT_VERSION_MAX = 2
VERSION_BASE = 1
VERSION_CALIBRATED = 2
__version__ = VERSION_BASE  # kept for backward compatibility of importers
MAGIC = b"UNILID\x00\x00"
HEADER_FMT = "<8sIIIII4x"  # magic, version, num_langs, vocab_size, base_tok_len, langs_len, padding
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CAL_LEN_FMT = "<Q"
CAL_LEN_SIZE = struct.calcsize(CAL_LEN_FMT)


def _get_vocab_with_scores(tok: Tokenizer) -> List[Tuple[str, float]]:
    """Extract vocab with scores from HF Unigram tokenizer."""
    state = tok.model.__getstate__()
    attributes = json.loads(state.decode("utf-8"))
    return attributes["vocab"]


def write_unilid(output_path: Path, base_tok_bytes: bytes, langs: List[str],
                 weights: np.ndarray,
                 calibration: Optional[Calibration] = None) -> Path:
    """Write a .unilid container from explicit components.

    ``langs`` order defines the weight-row order verbatim (no filename-based
    discovery or re-sorting happens here). Writes version 1 without a
    calibration, byte-identical to the historical format; version 2 with one.
    """
    output_path = Path(output_path)
    if weights.dtype != np.float32:
        raise ValueError(f"weights must be float32, got {weights.dtype}")
    if weights.ndim != 2 or weights.shape[0] != len(langs):
        raise ValueError(
            f"weights shape {weights.shape} does not match {len(langs)} languages")
    langs_bytes = json.dumps(langs).encode("utf-8")
    version = VERSION_CALIBRATED if calibration is not None else VERSION_BASE

    with open(output_path, "wb") as f:
        f.write(struct.pack(
            HEADER_FMT, MAGIC, version, len(langs), weights.shape[1],
            len(base_tok_bytes), len(langs_bytes)))
        f.write(base_tok_bytes)
        f.write(langs_bytes)
        f.write(np.ascontiguousarray(weights).tobytes())
        if calibration is not None:
            cal_bytes = calibration.to_json_bytes()
            f.write(struct.pack(CAL_LEN_FMT, len(cal_bytes)))
            f.write(cal_bytes)
    return output_path


def save_unilid(model_dir: Path, output_path: Path,
                calibration: Optional[Calibration] = None) -> Path:
    """
    Convert tokenizers directory to .unilid format.

    Args:
        model_dir: Directory containing tokenizers/ folder
        output_path: Output .unilid file path
        calibration: Optional calibration artifact to bundle (writes a version-2
            container; without it the output is byte-identical to the
            historical version-1 format)

    Returns:
        Path to saved .unilid file
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path).with_suffix(".unilid")

    tok_dir = model_dir / "tokenizers"
    if not tok_dir.exists():
        tok_dir = model_dir

    sidecar = tok_dir / "calibration.json"
    if calibration is None and sidecar.exists():
        raise UnilidCalibrationError(
            f"{sidecar} exists (this directory was unpacked from a calibrated "
            f"model) but no calibration was passed; packing without it would "
            f"silently downgrade the model to an uncalibrated version-1 file. "
            f"Pass calibration=Calibration.from_json_file({str(sidecar)!r}) "
            f"(or --calibration on the CLI), or delete the file to pack a "
            f"base model deliberately.")

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
    very_neg = np.float32(MISSING_TOKEN_FILL_LOG_PROB)
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
    write_unilid(output_path, base_tok_bytes, langs, weights, calibration)

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
    base_tok_bytes, weights, langs = load_unilid_raw(model_path)
    base_tok = Tokenizer.from_str(base_tok_bytes.decode("utf-8"))
    print(f"Loaded {len(langs)} languages, vocab size {weights.shape[1]}")
    return base_tok, weights, langs


def load_unilid_raw(model_path: Path) -> Tuple[bytes, np.ndarray, List[str]]:
    """Load a .unilid container returning the base tokenizer's ORIGINAL bytes
    (not a re-serialization), the memmapped weights, and the language list.
    Used wherever the base-tokenizer section must be copied through
    byte-identically (add_language, calibrate CLI)."""
    model_path = Path(model_path)

    with open(model_path, "rb") as f:
        # Read header
        header = f.read(HEADER_SIZE)
        magic, version, num_langs, vocab_size, base_tok_len, langs_len = struct.unpack(HEADER_FMT, header)

        if magic != MAGIC:
            raise ValueError(f"Invalid .unilid file (bad magic)")
        if version > FORMAT_VERSION_MAX:
            raise ValueError(
                f"Unsupported .unilid version {version} (this unilid reads up "
                f"to {FORMAT_VERSION_MAX}); upgrade the unilid package")

        # Read body
        base_tok_bytes = f.read(base_tok_len)
        langs_bytes = f.read(langs_len)
        weights_offset = f.tell()

    langs = json.loads(langs_bytes.decode("utf-8"))

    # Memory-map weights for fast loading
    weights = np.memmap(
        model_path,
        dtype=np.float32,
        mode="r",
        offset=weights_offset,
        shape=(num_langs, vocab_size),
    )
    return base_tok_bytes, weights, langs


def subset_rows(weights: np.ndarray, langs: List[str],
                languages: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Select the weight rows for ``languages``, preserving the MODEL's row
    order (not the argument's). Returns (subset_weights, subset_langs); the
    subset array is materialized (reading only the selected rows when
    ``weights`` is a memmap). Unknown, duplicate, or empty selections are
    errors."""
    if not languages:
        raise ValueError("languages must be a non-empty list")
    seen = set()
    for lang in languages:
        if lang in seen:
            raise ValueError(f"duplicate language in subset: {lang!r}")
        seen.add(lang)
    lang_set = set(langs)
    unknown = [l for l in languages if l not in lang_set]
    if unknown:
        raise ValueError(
            f"{len(unknown)} requested language(s) not in the model, "
            f"first: {unknown[:5]}")
    keep_idx = [i for i, l in enumerate(langs) if l in seen]
    sub_langs = [langs[i] for i in keep_idx]
    sub_weights = np.asarray(weights)[keep_idx]
    return sub_weights, sub_langs


def read_calibration(model_path: Path) -> Optional[Calibration]:
    """Read the bundled calibration from a .unilid file.

    Returns None for a version-1 file. For a version-2 file the calibration
    section is required: a truncated or absent section is a corruption error,
    never silently ignored.
    """
    model_path = Path(model_path)
    with open(model_path, "rb") as f:
        header = f.read(HEADER_SIZE)
        magic, version, num_langs, vocab_size, base_tok_len, langs_len = struct.unpack(HEADER_FMT, header)
        if magic != MAGIC:
            raise ValueError(f"Invalid .unilid file (bad magic)")
        if version > FORMAT_VERSION_MAX:
            raise ValueError(
                f"Unsupported .unilid version {version} (this unilid reads up "
                f"to {FORMAT_VERSION_MAX}); upgrade the unilid package")
        if version < VERSION_CALIBRATED:
            return None
        cal_offset = (HEADER_SIZE + base_tok_len + langs_len
                      + num_langs * vocab_size * 4)
        f.seek(cal_offset)
        len_bytes = f.read(CAL_LEN_SIZE)
        if len(len_bytes) != CAL_LEN_SIZE:
            raise UnilidCalibrationError(
                f"version-{version} .unilid file is truncated: calibration "
                f"length field missing at offset {cal_offset} in {model_path}")
        (cal_len,) = struct.unpack(CAL_LEN_FMT, len_bytes)
        cal_bytes = f.read(cal_len)
        if len(cal_bytes) != cal_len:
            raise UnilidCalibrationError(
                f".unilid calibration section is truncated: expected "
                f"{cal_len} bytes, found {len(cal_bytes)} in {model_path}")
    return Calibration.from_json_bytes(cal_bytes)


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

    calibration = read_calibration(model_path)
    if calibration is not None:
        cal_path = output_dir / "calibration.json"
        calibration.to_json_file(cal_path)
        print(f"Saved calibration: {cal_path}")

    print(f"\nUnpacked to: {output_dir}")
    print(f"  - 1 base tokenizer")
    print(f"  - {len(langs)} language tokenizers")
    return output_dir


class UnilidModel:
    """High-level model wrapper for inference.

    Calibrated inference (the paper's "Calibrated UniLID") is the default: the
    unseen-token constant is applied to the weight matrix at load time and
    low-margin predictions into the two re-examined groups are re-examined at
    prediction time. Pass ``calibrated=False`` for the base model's behavior.
    """

    def __init__(self, model_path, calibrated: bool = True,
                 calibration=None, languages: Optional[List[str]] = None):
        """
        Load model from .unilid file or tokenizers directory.

        Args:
            model_path: Path to .unilid file or directory containing tokenizers/
            calibrated: Use calibrated inference (default). Requires a
                calibration artifact: bundled in a version-2 .unilid file, or
                supplied via ``calibration``. Raises UnilidCalibrationError if
                neither is present.
            calibration: Path to a standalone calibration JSON. Only valid when
                the model file does not already bundle one.
            languages: Restrict the model to this subset of its languages
                (scoring cost is linear in the number of languages). Row order
                follows the model, not this list. Under calibrated inference
                the thresholds are carried over from the full model, which
                makes re-examination fire at most as often as calibrated (a
                smaller candidate set can only raise a line's margin); use
                `unilid-calibrate subset --recalibrate` to re-estimate them.
        """
        model_path = Path(model_path)
        self.calibrated = False
        # Set only when calibrated inference is active: self.calibration is not
        # None if and only if self.calibrated (a bundled artifact is NOT parsed
        # onto the instance in base mode, so audits can trust the attribute).
        self.calibration = None
        self.last_reexamination_stats = None

        if not calibrated and calibration is not None:
            raise UnilidCalibrationError(
                "calibration= was given together with calibrated=False; drop "
                "one (a supplied calibration would be silently unused)")

        if model_path.suffix == ".unilid" or (model_path.is_file() and str(model_path).endswith(".unilid")):
            bundled = read_calibration(model_path)
            if bundled is not None and calibration is not None:
                raise UnilidCalibrationError(
                    f"{model_path} already bundles a calibration; drop the "
                    f"calibration= argument (unpack and re-bundle to replace it)")
            cal = bundled if bundled is not None else (
                Calibration.from_json_file(calibration)
                if calibration is not None else None)
            self._load_from_unilid(model_path, calibrated=calibrated,
                                   cal=cal, source=str(model_path),
                                   languages=languages)
        elif model_path.is_dir():
            cal = (Calibration.from_json_file(calibration)
                   if calibration is not None else None)
            self._load_from_dir(model_path, calibrated=calibrated, cal=cal,
                                languages=languages)
        else:
            raise ValueError(f"Unknown model format: {model_path}. Expected .unilid file or directory.")

    def _subset_and_report(self, weights, langs, languages, cal,
                           calibrated: bool):
        """Apply the language subset and, when calibrated, filter the
        calibration and state the carried-thresholds consequence."""
        weights, langs = subset_rows(weights, langs, languages)
        print(f"Restricted to {len(langs)} of the model's languages")
        if calibrated and cal is not None:
            cal = cal.subset_for(langs)
            print("Re-examination thresholds are carried from the full model "
                  "(margins against a smaller candidate set are at least as "
                  "large, so re-examination fires at most as often as "
                  "calibrated); unilid-calibrate subset --recalibrate "
                  "re-estimates them")
        return weights, langs, cal

    def _require_scorer_methods(self):
        """The pinned tokenizers fork provides the numpy weight-loading path and
        the calibrated-inference scorers; an older build of the extension is a
        setup error, reported with the fix rather than degraded silently."""
        missing = [name for name in ("set_weight_sets_numpy",
                                     "top_k_of_cached_weight_sets_batch",
                                     "tokens_of_cached_weight_set_batch")
                   if not hasattr(self.model, name)]
        if missing:
            raise RuntimeError(
                f"the installed tokenizers extension is missing {missing}; "
                f"rebuild the bundled fork (cd tokenizers/bindings/python && "
                f"maturin develop --release) as described in the README")

    def _init_calibrated(self, weights: np.ndarray, cal, source: str):
        """Validate the calibration against this model, apply the unseen-token
        constant, and push the clamped matrix to the Rust cache."""
        if cal is None:
            raise UnilidCalibrationError(
                f"calibrated=True (the default) but {source} carries no "
                f"calibration artifact. Either load the calibrated model "
                f"release (a version-2 .unilid file), pass calibration=<path "
                f"to calibration.json>, or pass calibrated=False for base "
                f"(uncalibrated) inference.")
        self._runtime = cal.runtime_for(self.langs)
        w_cal, n_mod = apply_unseen_token_constant(
            weights, cal.unseen_token_constant)
        print(f"Applied unseen-token constant {cal.unseen_token_constant} "
              f"({n_mod}/{len(self.langs)} languages modified)")
        self.model.set_weight_sets_numpy(w_cal)
        del w_cal
        self.calibration = cal
        self.calibrated = True

    def _load_from_unilid(self, model_path: Path, calibrated: bool = True,
                          cal=None, source: str = "", languages=None):
        """Load from .unilid file."""
        base_tok, weights, self.langs = load_unilid(model_path)
        if languages is not None:
            weights, self.langs, cal = self._subset_and_report(
                weights, self.langs, languages, cal, calibrated)

        self.tokenizer = base_tok
        self.model = base_tok.model
        self.pre_tok = base_tok.pre_tokenizer
        self.normalizer = base_tok.normalizer
        self._lang_to_idx = {lang: i for i, lang in enumerate(self.langs)}

        print("Pushing weights to Rust cache...")
        self._require_scorer_methods()
        if calibrated:
            self._init_calibrated(weights, cal, source or str(model_path))
        else:
            # memmap passes straight through the buffer protocol: the rows are
            # copied into the Rust cache without materializing a Python list.
            self.model.set_weight_sets_numpy(weights)

        del weights
        gc.collect()
        print("Model ready")

    @classmethod
    def from_dir(cls, model_dir: Path) -> "UnilidModel":
        """Load directly from tokenizers directory (deprecated, use __init__ instead)."""
        return cls(model_dir)

    def _load_from_dir(self, model_dir: Path, calibrated: bool = True,
                       cal=None, languages=None):
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
        very_neg = np.float32(MISSING_TOKEN_FILL_LOG_PROB)
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

        if languages is not None:
            weights, langs, cal = self._subset_and_report(
                weights, langs, languages, cal, calibrated)

        self.tokenizer = base_tok
        self.model = base_tok.model
        self.pre_tok = base_tok.pre_tokenizer
        self.normalizer = base_tok.normalizer
        self.langs = langs
        self._lang_to_idx = {lang: i for i, lang in enumerate(langs)}

        print("Pushing weights to Rust cache...")
        self._require_scorer_methods()
        if calibrated:
            self._init_calibrated(weights, cal, str(model_dir))
        else:
            self.model.set_weight_sets_numpy(weights)

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

    def predict(self, text: str, forward: bool = False) -> Tuple[str, List[str], float]:
        """Predict language for a single text. ``forward=True`` scores by
        marginalizing over all segmentations (base mode only: the calibration
        thresholds are defined on Viterbi margins)."""
        if self.calibrated:
            if forward:
                self._require_base_mode("predict(forward=True)")
            return self.predict_batch([text])[0]
        pt = self.preprocess(text)
        if not pt:
            return None, [], float("-inf")
        if forward:
            idx, tokens, score = self.model.best_of_cached_weight_sets_forward(pt)
        else:
            idx, tokens, score = self.model.best_of_cached_weight_sets(pt)
        return self.langs[idx], tokens, score

    def predict_batch(self, texts: List[str], forward: bool = False) -> List[Tuple[str, List[str], float]]:
        """Predict languages for multiple texts (Rayon parallel).
        ``forward=True`` scores by marginalizing over all segmentations (base
        mode only: the calibration thresholds are defined on Viterbi margins).
        """
        if forward and self.calibrated:
            self._require_base_mode("predict_batch(forward=True)")
        preprocessed, valid_idx = [], []
        for i, text in enumerate(texts):
            pt = self.preprocess(text)
            if pt:
                preprocessed.append(pt)
                valid_idx.append(i)

        if not preprocessed:
            return [(None, [], float("-inf"))] * len(texts)

        if self.calibrated:
            batch_results = self._predict_batch_calibrated(preprocessed)
        elif forward:
            batch_results = self.model.best_of_cached_weight_sets_forward_batch(preprocessed)
        else:
            batch_results = self.model.best_of_cached_weight_sets_batch(preprocessed)

        results = [(None, [], float("-inf"))] * len(texts)
        for j, (idx, tokens, score) in enumerate(batch_results):
            results[valid_idx[j]] = (self.langs[idx], tokens, score)
        return results

    def _predict_batch_calibrated(self, preprocessed: List[str]
                                  ) -> List[Tuple[int, List[str], float]]:
        """Calibrated inference over preprocessed texts: top-k scoring on the
        clamped matrix, gate + replacement walk, then one segmentation pass
        under each text's final language. Returns (lang_idx, tokens, score)
        like best_of_cached_weight_sets_batch; the score is the segmentation
        pass's score for the final language (same definition as the base path).
        """
        cal = self.calibration
        topk_lists = self.model.top_k_of_cached_weight_sets_batch(
            preprocessed, cal.topk)

        n = len(topk_lists)
        # Materialize as the reference chain does: ids int64, scores float32,
        # unfilled slots id -1 / score -inf (analysis gate_topk arrays).
        ids = np.full((n, cal.topk), -1, dtype=np.int64)
        scores = np.full((n, cal.topk), -np.inf, dtype=np.float32)
        for r, cands in enumerate(topk_lists):
            if not cands:
                raise UnilidCalibrationError(
                    "top-k scoring returned no candidates for a non-empty "
                    "preprocessed text; cannot re-examine")
            for j, (li, s) in enumerate(cands):
                ids[r, j] = li
                scores[r, j] = np.float32(s)

        final, stats = re_examine(ids, scores, self._runtime)
        self.last_reexamination_stats = stats

        final_list = [int(x) for x in final]
        seg = self.model.tokens_of_cached_weight_set_batch(
            preprocessed, final_list)
        return [(final_list[r], tokens, score)
                for r, (tokens, score) in enumerate(seg)]

    def _require_base_mode(self, method: str):
        if self.calibrated:
            raise UnilidCalibrationError(
                f"{method} is only defined for the base model (it would "
                f"otherwise run on the matrix with the unseen-token constant "
                f"applied but without the re-examination, which is neither "
                f"base nor calibrated inference); load with calibrated=False "
                f"to use it")

    def predict_normalized(self, text: str, alpha: float = 1.0) -> Tuple[str, List[str], float]:
        """Predict language using length-normalized scores (score / n_tokens^alpha)."""
        self._require_base_mode("predict_normalized")
        pt = self.preprocess(text)
        if not pt:
            return None, [], float("-inf")
        idx, tokens, score = self.model.best_of_cached_weight_sets_normalized(pt, alpha)
        return self.langs[idx], tokens, score

    def predict_normalized_batch(self, texts: List[str], alpha: float = 1.0) -> List[Tuple[str, List[str], float]]:
        """Predict languages using length-normalized scores (Rayon parallel)."""
        self._require_base_mode("predict_normalized_batch")
        preprocessed, valid_idx = [], []
        for i, text in enumerate(texts):
            pt = self.preprocess(text)
            if pt:
                preprocessed.append(pt)
                valid_idx.append(i)

        if not preprocessed:
            return [(None, [], float("-inf"))] * len(texts)

        batch_results = self.model.best_of_cached_weight_sets_normalized_batch(preprocessed, alpha)

        results = [(None, [], float("-inf"))] * len(texts)
        for j, (idx, tokens, score) in enumerate(batch_results):
            results[valid_idx[j]] = (self.langs[idx], tokens, score)
        return results

    @property
    def num_languages(self) -> int:
        return len(self.langs)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert between tokenizers directories and .unilid files")
    parser.add_argument("input", type=Path,
                        help="Model directory (pack) or .unilid file (--unpack)")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--unpack", action="store_true",
                        help="Unpack a .unilid file back to a tokenizers "
                             "directory (writes calibration.json for a "
                             "version-2 file)")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Calibration JSON to bundle when packing (writes "
                             "a version-2 container)")
    args = parser.parse_args(argv)
    if args.unpack:
        if args.calibration:
            parser.error("--calibration only applies when packing")
        if args.input.suffix != ".unilid":
            parser.error(f"--unpack expects a .unilid file, got {args.input}")
        output = args.output or args.input.with_suffix("")
        unpack_unilid(args.input, output)
    else:
        output = args.output or (args.input / "model.unilid")
        cal = (Calibration.from_json_file(args.calibration)
               if args.calibration else None)
        save_unilid(args.input, output, calibration=cal)


if __name__ == "__main__":
    main()
