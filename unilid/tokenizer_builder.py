import json
from typing import List, Tuple, Any

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import ByteLevel, Sequence

from unilid import constants

import logging
logger = logging.getLogger(__name__)


def set_special_tokens(tokenizer):
    vocab = tokenizer.get_vocab()

    for token_name, token in constants.SPECIAL_TOKENS.items():
        if token in vocab:
            setattr(tokenizer, token_name, token)
            setattr(tokenizer, token_name + '_id', vocab[token])


def load_tokenizer_with_mode(path: str):
    tok = Tokenizer.from_file(path)
    byte_level = isinstance(tok.pre_tokenizer, ByteLevel) or (isinstance(tok.pre_tokenizer, Sequence) and isinstance(tok.pre_tokenizer[1], ByteLevel))
    try:
        byte_fallback = bool(tok.model.byte_fallback)
    except AttributeError:
        byte_fallback = False
    if byte_level and not byte_fallback:
        logger.warning(f"{path} looks byte-level but byte_fallback==False (currently known issue for HF Unigram Tokenizers).")
    return tok, byte_level


def _build_unigramlm_hf_tokenizer_from_lprobs(
    lprobs_tuples: List[Tuple[str, float]],
    unk_id: int,
    pretokenizer: Any = None,
    decoder: Any = None,
    base_tokenizer: Any = None
) -> Tokenizer:
    """
    Turn a plain log probability dictionary into a HF Unigram tokenizer.
    """
    model = Unigram(lprobs_tuples, unk_id)
    tok = Tokenizer(model)
    if base_tokenizer is not None:
        base_json = json.loads(base_tokenizer.to_str())
        model_json = json.loads(tok.to_str())["model"]
        base_json["model"] = model_json
        tok = Tokenizer.from_str(json.dumps(base_json))

    if pretokenizer:
        tok.pre_tokenizer = pretokenizer
    if decoder:
        tok.decoder = decoder
    return tok
