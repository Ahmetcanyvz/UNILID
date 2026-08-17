import importlib

from unilid.model_io import (
    UnilidModel,
    load_unilid,
    read_calibration,
    save_unilid,
    unpack_unilid,
    write_unilid,
)
from unilid.calibration import Calibration, TauRow, UnilidCalibrationError, estimate_tau
from unilid.add_language import add_language

__version__ = "0.3.0"

# Training-side classes and functions are imported lazily (PEP 562): they pull
# in the optional [train] dependencies (torch, ujson, transformers), and a
# prediction-only install must not need those to `from unilid import
# load_model`.
_LAZY_ATTRS = {
    "StandardUnigramLMTokenizer": "unilid.trainers.standard_trainer",
    "LanguageSpecificUnigramLMTokenizer": "unilid.trainers.language_specific_trainer",
    "EMUnigramTrainer": "unilid.trainers.em_trainer",
    "CorpusTokenizer": "unilid.corpus_tokenizer",
    "load_tokenizer_from_config": "unilid.api",
    "train_standard_tokenizer": "unilid.api",
    "train_lang_tokenizers": "unilid.api",
    "train_tokmix": "unilid.api",
    "train_language_specific_tokenizer": "unilid.api",
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))


def load_model(path, calibrated: bool = True, calibration=None,
               languages=None):
    """
    Load a UNILID model for language identification.

    Args:
        path: Path to .unilid file or tokenizers directory
        calibrated: Use calibrated inference (default). Requires a calibration
            artifact (bundled in a version-2 .unilid file or passed via
            ``calibration``); raises UnilidCalibrationError otherwise. Pass
            ``calibrated=False`` for the base model's behavior.
        calibration: Path to a standalone calibration JSON.
        languages: Restrict the model to this subset of its languages
            (scoring cost is linear in the number of languages). Under
            calibrated inference the re-examination thresholds are carried
            over from the full model; see the README's subsetting section.

    Returns:
        UnilidModel instance ready for prediction

    Example:
        >>> from unilid import load_model
        >>> model = load_model("model.unilid")
        >>> lang, tokens, score = model.predict("Hello world")
        >>> print(lang)  # 'eng'
    """
    return UnilidModel(path, calibrated=calibrated, calibration=calibration,
                       languages=languages)


__all__ = [
    # High-level API
    "load_model",
    "UnilidModel",
    "__version__",
    # Calibration
    "Calibration",
    "TauRow",
    "UnilidCalibrationError",
    "add_language",
    "estimate_tau",
    "read_calibration",
    # Model I/O
    "load_unilid",
    "save_unilid",
    "unpack_unilid",
    "write_unilid",
    # Training (lazy: needs the [train] extra)
    "StandardUnigramLMTokenizer",
    "LanguageSpecificUnigramLMTokenizer",
    "EMUnigramTrainer",
    "CorpusTokenizer",
    "load_tokenizer_from_config",
    "train_standard_tokenizer",
    "train_lang_tokenizers",
    "train_tokmix",
    "train_language_specific_tokenizer",
]
