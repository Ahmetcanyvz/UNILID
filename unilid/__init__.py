from unilid.model_io import UnilidModel, load_unilid, save_unilid, unpack_unilid
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

__version__ = "0.1.0"


def load_model(path):
    """
    Load a UNILID model for language identification.

    Args:
        path: Path to .unilid file or tokenizers directory

    Returns:
        UnilidModel instance ready for prediction

    Example:
        >>> from unilid import load_model
        >>> model = load_model("model.unilid")
        >>> lang, tokens, score = model.predict("Hello world")
        >>> print(lang)  # 'eng'
    """
    return UnilidModel(path)


__all__ = [
    # High-level API
    "load_model",
    "UnilidModel",
    "__version__",
    # Model I/O
    "load_unilid",
    "save_unilid",
    "unpack_unilid",
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
