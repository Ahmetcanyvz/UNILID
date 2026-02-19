"""
Validation utilities for UNILID tokenizer training.

Ensures trained tokenizers maintain consistency with the base tokenizer
in terms of pre/post-tokenization components and token ordering.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from tokenizers import Tokenizer

from unilid.token_encoding import _get_hf_unigram_tokenizer_vocab

logger = logging.getLogger(__name__)


def _load_tokenizer_json(tokenizer_path: Path) -> dict:
    """Load tokenizer JSON file as a dictionary."""
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_token_list(tok: Tokenizer) -> List[str]:
    """Extract ordered list of tokens from a HF Tokenizer object."""
    vocab_with_scores = _get_hf_unigram_tokenizer_vocab(tok)
    return [token for token, _score in vocab_with_scores]


def _get_token_list_from_file(tokenizer_path: Path) -> List[str]:
    """Extract ordered list of tokens from a tokenizer JSON file."""
    tok = Tokenizer.from_file(str(tokenizer_path))
    tokens = _get_token_list(tok)
    assert len(tokens) > 0, f"Tokenizer vocab is empty: {tokenizer_path}"
    return tokens


# ---------------------------------------------------------------------------
# Component validation
# ---------------------------------------------------------------------------

def validate_pretokenizer(
    tokenizer_data: dict,
    expected_pretokenizer: dict,
    tokenizer_name: str,
) -> bool:
    """Validate that a tokenizer's pre-tokenizer matches the expected one.

    Returns True if valid, False if mismatch.
    """
    actual = tokenizer_data.get("pre_tokenizer")
    if actual is None and expected_pretokenizer is None:
        return True
    if actual != expected_pretokenizer:
        logger.error(
            "%s tokenizer pre-tokenizer mismatch.\n  Expected: %s\n  Got: %s",
            tokenizer_name, expected_pretokenizer, actual,
        )
        return False
    return True


def validate_decoder(
    tokenizer_data: dict,
    expected_decoder: dict,
    tokenizer_name: str,
) -> bool:
    """Validate that a tokenizer's decoder matches the expected one.

    Returns True if valid, False if mismatch.
    """
    actual = tokenizer_data.get("decoder")
    if actual is None and expected_decoder is None:
        return True
    if actual != expected_decoder:
        logger.error(
            "%s tokenizer decoder mismatch.\n  Expected: %s\n  Got: %s",
            tokenizer_name, expected_decoder, actual,
        )
        return False
    return True


def validate_token_order(
    tokenizer_path: Path,
    expected_tokens: List[str],
    tokenizer_name: str,
) -> bool:
    """Validate that a tokenizer's token ordering matches the expected order.

    Returns True if valid, False if mismatch.
    """
    actual_tokens = _get_token_list_from_file(tokenizer_path)
    if actual_tokens != expected_tokens:
        logger.error(
            "%s tokenizer token order mismatch. "
            "Expected %d tokens, got %d tokens.",
            tokenizer_name, len(expected_tokens), len(actual_tokens),
        )
        return False
    logger.debug("%s: token order matches base tokenizer", tokenizer_name)
    return True


def validate_components(
    tokenizer_data: dict,
    expected_pretokenizer: dict,
    expected_decoder: dict,
    tokenizer_name: str,
) -> bool:
    """Validate pre-tokenizer and decoder components of a tokenizer.

    Returns True if all components match.
    """
    ok = True
    if not validate_pretokenizer(tokenizer_data, expected_pretokenizer, tokenizer_name):
        ok = False
    if not validate_decoder(tokenizer_data, expected_decoder, tokenizer_name):
        ok = False
    if ok:
        logger.debug("%s: pre/post-tokenizer components match base", tokenizer_name)
    return ok


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_tokenizers(
    base_tokenizer_path: Path,
    lang_tokenizer_paths: Optional[Dict[str, Path]] = None,
) -> bool:
    """Validate that all trained tokenizers have:

    1. Identical pre-tokenizer and decoder components as the base
    2. Token order matching the base tokenizer

    Args:
        base_tokenizer_path: Path to the base tokenizer JSON
        lang_tokenizer_paths: Dict mapping lang_code -> path to lang tokenizer

    Returns:
        True if all checks pass, False otherwise.
    """
    logger.info("Validating tokenizers against base: %s", base_tokenizer_path)

    base_tokens = _get_token_list_from_file(base_tokenizer_path)
    base_data = _load_tokenizer_json(base_tokenizer_path)
    expected_pretokenizer = base_data.get("pre_tokenizer")
    expected_decoder = base_data.get("decoder")

    all_ok = True

    if lang_tokenizer_paths:
        for lang, lang_path in lang_tokenizer_paths.items():
            lang_path = Path(lang_path)
            if not lang_path.exists():
                logger.error("Language tokenizer not found: %s", lang_path)
                all_ok = False
                continue

            lang_data = _load_tokenizer_json(lang_path)
            if not validate_components(lang_data, expected_pretokenizer, expected_decoder, lang):
                all_ok = False
            if not validate_token_order(lang_path, base_tokens, lang):
                all_ok = False

    if all_ok:
        n = len(lang_tokenizer_paths) if lang_tokenizer_paths else 0
        logger.info("All validation checks passed (%d language tokenizers)", n)
    else:
        logger.warning("Some validation checks FAILED — see errors above")

    return all_ok
