import math
import numpy as np
from typing import Dict, List, Tuple

from unilid.constants import SPECIAL_TOKENS, MAX_BYTE_TOKEN_LEN, MAX_CHAR_TOKEN_LEN
from unilid.algorithms.viterbi import _segment_word_viterbi

import logging
logger = logging.getLogger(__name__)


def _compute_alternatives_direct(
    token_log_probs: dict,
    unk_token: str,
    byte_level: bool = False
) -> Dict[str, List[Tuple[List[str], float]]]:
    """
    Compute alternatives directly for each token without corpus iteration.
    """
    alternatives = {}

    all_tokens = list(token_log_probs.keys())
    for token in all_tokens:
        if token == unk_token or token in SPECIAL_TOKENS:
            continue

        token_string = token

        orig_len = len(token_log_probs)
        alt_sequence = _get_segmentation_excluding_token(
            token_string,
            token_log_probs,
            unk_token,
            token,
            max_token_len=MAX_BYTE_TOKEN_LEN if byte_level else MAX_CHAR_TOKEN_LEN
        )
        assert len(token_log_probs) == orig_len
        alternatives[token] = [(alt_sequence, 1.0)]

    return alternatives


def _get_segmentation_excluding_token(substring: str, token_log_probs: dict, unk_token: str,
                                excluded_token: str, max_token_len: int) -> List[List[str]]:
    """
    Get alternative tokenization sequences when excluded_token is not available.
    """
    exc_tok_lprob = token_log_probs.pop(excluded_token, None)

    alt_seg = _segment_word_viterbi(substring, token_log_probs, unk_token, max_token_len)
    if exc_tok_lprob:
        token_log_probs[excluded_token] = exc_tok_lprob

    return alt_seg


def _compute_loss_with_alternative_sequences(
    token_to_remove: str,
    counts: Dict[str, float],
    alternative_sequences: Dict[str, List[Tuple[List[str], float]]],
    lp_token: Dict[str, float],
    unk_token: str
) -> float:
    """
    Compute loss accounting for full sequence probabilities.
    Loss = NLL(corpus | vocab_without_token) - NLL(corpus | vocab_with_token)
    """
    if token_to_remove not in counts or counts[token_to_remove] == 0:
        return 0.0

    if token_to_remove not in lp_token:
        return 0.0

    removed_count = counts[token_to_remove]
    current_nll = -removed_count * lp_token[token_to_remove]

    unk_logp = lp_token[unk_token]

    weighted_sequences = alternative_sequences.get(token_to_remove, [])

    if not weighted_sequences:
        new_nll = -removed_count * unk_logp
    else:
        if not np.isclose(sum([tup[1] for tup in weighted_sequences]), 1.0, atol=1e-5):
            logger.error("Alternative seq weights not adding to 1")
        new_nll = 0.0

        for alt_sequence, weight in weighted_sequences:
            if weight <= 0:
                continue

            seq_log_prob = 0.0
            valid_sequence = True

            for token in alt_sequence:
                if token in lp_token:
                    seq_log_prob += lp_token[token]
                else:
                    valid_sequence = False
                    break

            if valid_sequence:
                seq_count = removed_count * weight
                new_nll += -seq_count * seq_log_prob
            else:
                seq_count = removed_count * weight
                new_nll += -seq_count * len(alt_sequence) * unk_logp

    loss = new_nll - current_nll

    return loss
