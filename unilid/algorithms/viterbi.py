import numpy as np

from unilid.constants import MIN_TOKEN_LOG_PROB, MAX_CHAR_TOKEN_LEN, MAX_BYTE_TOKEN_LEN
from unilid.token_encoding import _apply_pretok_mapping_hf

import logging
logger = logging.getLogger(__name__)


def segment_text_viterbi(text,
                         token_log_probs,
                         unk_token="<unk>",
                         max_sub_len=MAX_CHAR_TOKEN_LEN,
                         beam_width=5.0,
                         pretok=None,
                         whitespace_boundaries=None):
    """
    Segment `text` into tokens using a whitespace split + Viterbi subword approach,
    under the distribution given by `token_log_probs`.
    Returns a list of tokens.
    """
    text = _apply_pretok_mapping_hf(text, pretok, whitespace_boundaries)
    if not isinstance(text, list):
        return _segment_word_viterbi(text, token_log_probs, unk_token, max_sub_len, beam_width)

    all_tokens = []
    if not text:
        return all_tokens

    for word in text:
        sub_tokens = _segment_word_viterbi(word, token_log_probs, unk_token, max_sub_len, beam_width)
        all_tokens.extend(sub_tokens)

    return all_tokens


def _segment_word_viterbi(
    word: str,
    token_log_probs: dict,
    unk_token: str = "<unk>",
    max_sub_len: int = MAX_CHAR_TOKEN_LEN,
    beam_width: float = 5.0,
):
    """
    Single-pass Viterbi segmentation closer to SentencePiece implementation.

    This implementation:
    - Considers all possible tokens (including UNK) in a single pass
    - Uses proper beam pruning based on score differences
    - Handles empty strings correctly
    - Matches SentencePiece's algorithm except for their UNK penalty
    """
    n = len(word)
    if n == 0:
        return []

    # Initialize DP tables
    best_score = np.full(n + 1, float("-inf"), dtype=np.float64)
    best_edge = np.full(n + 1, -1, dtype=np.int32)

    # Track the best score seen so far for beam pruning
    best_score_so_far = 0.0
    best_score[0] = 0.0

    # Precompute log probabilities
    unk_logp = token_log_probs.get(unk_token, MIN_TOKEN_LOG_PROB)

    # Single pass: consider all possibilities
    for start in range(n):
        score = best_score[start]
        if score == float("-inf"):
            continue

        # Beam pruning: skip if too far from best
        if score < best_score_so_far - beam_width:
            continue

        # Try all possible token lengths
        found_any = False
        max_end = min(n, start + max_sub_len)

        for end in range(start + 1, max_end + 1):
            sub = word[start:end]

            # Check if this substring is a known token
            if sub in token_log_probs:
                new_score = score + max(token_log_probs[sub], MIN_TOKEN_LOG_PROB)
                found_any = True
            else:
                # Use UNK token for unknown substrings
                if end == start + 1:  # Single character UNK
                    new_score = score + unk_logp
                else:
                    continue  # Skip multi-character unknown sequences

            # Update if this path is better
            if new_score > best_score[end]:
                best_score[end] = new_score
                best_edge[end] = start
                best_score_so_far = max(best_score_so_far, new_score)

        # If no known tokens found, must use single-character UNK
        if not found_any and start + 1 <= n:
            new_score = score + unk_logp
            if new_score > best_score[start + 1]:
                best_score[start + 1] = new_score
                best_edge[start + 1] = start
                best_score_so_far = max(best_score_so_far, new_score)

    # Backtrace to get tokens
    tokens = []
    pos = n
    while pos > 0:
        prev = int(best_edge[pos])
        if prev == -1:
            tokens.append(word[pos-1:pos])
            pos -= 1
        else:
            sub = word[prev:pos]
            if sub in token_log_probs:
                tokens.append(sub)
            else:
                tokens.append(unk_token)
            pos = prev

    return tokens[::-1]
