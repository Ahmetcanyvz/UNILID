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
from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer
from unilid.trainers.language_specific_trainer import LanguageSpecificUnigramLMTokenizer
from unilid.trainers.em_trainer import EMUnigramTrainer
from unilid.corpus_tokenizer import CorpusTokenizer
from unilid.api import (
    load_tokenizer_from_config,
    train_standard_tokenizer,
    train_lang_tokenizers,
    train_tokmix,
    train_language_specific_tokenizer,
)

__version__ = "0.2.0"


def load_model(path, calibrated: bool = True, calibration=None):
    """
    Load a UNILID model for language identification.

    Args:
        path: Path to .unilid file or tokenizers directory
        calibrated: Use calibrated inference (default). Requires a calibration
            artifact (bundled in a version-2 .unilid file or passed via
            ``calibration``); raises UnilidCalibrationError otherwise. Pass
            ``calibrated=False`` for the base model's behavior.
        calibration: Path to a standalone calibration JSON.

    Returns:
        UnilidModel instance ready for prediction

    Example:
        >>> from unilid import load_model
        >>> model = load_model("model.unilid")
        >>> lang, tokens, score = model.predict("Hello world")
        >>> print(lang)  # 'eng'
    """
    return UnilidModel(path, calibrated=calibrated, calibration=calibration)


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
    # Training
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
