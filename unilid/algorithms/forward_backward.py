import numpy as np

from unilid.constants import MAX_CHAR_TOKEN_LEN
from unilid.token_encoding import _apply_pretok_mapping_hf

import logging
logger = logging.getLogger(__name__)


def forward_backward_unigram(
    text,
    log_p_lang,
    unk_token="<unk>",
    max_sub_len=MAX_CHAR_TOKEN_LEN,
    pretok=None,
    whitespace_boundaries=False
):
    """
    Forward-backward for UnigramLM in log-space, with UNK gating:
    UNK is allowed exactly at positions where *no real token* can start.

    Args:
        text: input string
        log_p_lang: dict {token: log_prob} including <unk>
        unk_token: the unknown-token string
        max_sub_len: maximum subword length to consider
        pretok: pre-tokenizer to apply
        whitespace_boundaries: whether to split on whitespace

    Returns:
        dict {token: expected_count} (fractional EM counts)
    """
    texts = _apply_pretok_mapping_hf(text, pretok, whitespace_boundaries)

    if unk_token not in log_p_lang or not np.isfinite(log_p_lang[unk_token]):
        raise ValueError(f"'{unk_token}' must be in log_p_lang with a finite log-prob.")
    log_p_unk = log_p_lang[unk_token]

    real_vocab = set(log_p_lang.keys()) - {unk_token}

    token_cache = {}
    def get_token_log_prob(token):
        if token not in token_cache:
            token_cache[token] = log_p_lang[token]
        return token_cache[token]

    log_total_counts = {}
    def add_count(tok, log_count):
        if np.isfinite(log_count):
            current_log_total = log_total_counts.get(tok, -np.inf)
            log_total_counts[tok] = np.logaddexp(current_log_total, log_count)

    for text in texts:
        text = text.rstrip("\n")
        n = len(text)
        if n == 0:
            continue
        can_start = [False] * n
        for i in range(n):
            end = min(n, i + max_sub_len)
            for j in range(i + 1, end + 1):
                if text[i:j] in real_vocab:
                    can_start[i] = True
                    break

        requires_unk = [False] * (n + 1)
        for i in range(n):
            requires_unk[i + 1] = not can_start[i]

        # --- Forward pass (log-space)
        log_alpha = np.full(n + 1, -np.inf)
        log_alpha[0] = 0.0

        for i in range(n):
            if log_alpha[i] == -np.inf:
                continue

            if requires_unk[i + 1]:
                log_alpha[i + 1] = np.logaddexp(log_alpha[i + 1], log_alpha[i] + log_p_unk)

            end = min(n, i + max_sub_len)
            for j in range(i + 1, end + 1):
                sub = text[i:j]
                if sub in real_vocab:
                    log_alpha[j] = np.logaddexp(log_alpha[j], log_alpha[i] + log_p_lang[sub])

        log_Z = log_alpha[n]
        if not np.isfinite(log_Z):
            raise RuntimeError(
                "Failed to reach the end of the word. Numerical stability issues."
            )

        # --- Backward pass (log-space)
        log_beta = np.full(n + 1, -np.inf)
        log_beta[n] = 0.0

        for i in range(n - 1, -1, -1):
            if requires_unk[i + 1]:
                log_beta[i] = np.logaddexp(log_beta[i], log_p_unk + log_beta[i + 1])

            end = min(n, i + max_sub_len)
            for j in range(i + 1, end + 1):
                sub = text[i:j]
                if sub in real_vocab:
                    log_beta[i] = np.logaddexp(log_beta[i], log_p_lang[sub] + log_beta[j])

        if not np.isclose(log_beta[0], log_Z, rtol=1e-8, atol=1e-8):
            logger.warning("Large floating point errors...")

        for i in range(n):
            if log_alpha[i] == -np.inf:
                continue

            if requires_unk[i + 1] and log_beta[i + 1] != -np.inf:
                add_count(unk_token, log_alpha[i] + log_p_unk + log_beta[i + 1] - log_Z)

            end = min(n, i + max_sub_len)
            for j in range(i + 1, end + 1):
                sub = text[i:j]
                if sub in real_vocab and log_beta[j] != -np.inf:
                    add_count(sub, log_alpha[i] + log_p_lang[sub] + log_beta[j] - log_Z)

    final_counts = {
        token: np.exp(log_total)
        for token, log_total in log_total_counts.items()
    }

    return final_counts
