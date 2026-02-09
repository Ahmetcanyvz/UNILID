import json
from typing import Any

from unilid import constants


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _apply_pretok_mapping_hf(text: str, pretok: Any, return_individual_pieces: bool = False, normalizer: Any = None) -> str:
    # Apply normalizer first if provided (e.g. Llama 2 uses normalizer for ▁ prefix)
    if normalizer is not None:
        text = normalizer.normalize_str(text)

    # Apply pre-tokenizer if available
    if pretok is not None:
        normalized = pretok.pre_tokenize_str(text)
        pieces = [piece[0] for piece in normalized]
    else:
        # No pre-tokenizer (e.g. Llama 2) - use normalized text as single piece
        pieces = [text]

    if return_individual_pieces:
        return pieces
    return ["".join(pieces)]


def _hf_to_sp_piece(token: str) -> str:
    """
    HF ByteLevel -> SentencePiece seed text:
      - GPT-2 space marker (byte 0x20) -> '▁'
      - printable bytes -> literal char
      - other bytes -> '<0xXX>'
    """
    out = []
    for ch in token:
        b = constants._UNI_TO_BYTE.get(ch)  # GPT-2 byte-unicode -> raw byte (0..255) or None
        if b is None:
            out.append("\u2581" if ch == " " else ch)
            continue
        if b == 0x20:
            out.append("\u2581")
        elif 33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255:
            out.append(chr(b))
        else:
            out.append(f"<0x{b:02X}>")
    return "".join(out)


def _get_hf_unigram_tokenizer_vocab(tokenizer):
    state = tokenizer.model.__getstate__()
    attributes = json.loads(state.decode("utf-8"))
    vocab = attributes["vocab"]
    if isinstance(vocab, dict):
        # This is the format the BPE vocab comes in
        tuples = sorted(vocab.items(), key=lambda x: x[1])
        return [(tk, constants.MIN_TOKEN_LOG_PROB) for tk, idx in tuples]
    return vocab
