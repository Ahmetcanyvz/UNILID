import multiprocessing
from collections import defaultdict

from unilid.constants import MAX_BYTE_TOKEN_LEN, MAX_CHAR_TOKEN_LEN
from unilid.algorithms.viterbi import segment_text_viterbi
from unilid.algorithms.forward_backward import forward_backward_unigram

import logging
logger = logging.getLogger(__name__)


def _accumulate_usage_worker_soft(args):
    """
    Worker for the 'soft' E-step, i.e. forward-backward to accumulate fractional usage.
    """
    chunk_lines, p_lang, unk_token, whitespace_boundaries, pretok, max_token_len = args

    c_dict = defaultdict(float)
    total = 0.0

    for line in chunk_lines:
        usage_line = forward_backward_unigram(line,
                                              p_lang,
                                              unk_token=unk_token,
                                              pretok=pretok,
                                              max_sub_len=max_token_len,
                                              whitespace_boundaries=whitespace_boundaries)
        line_count = sum(usage_line.values())
        total += line_count
        for tk, frac in usage_line.items():
            c_dict[tk] += frac

    return (dict(c_dict), total)


def _accumulate_usage_worker_hard(args):
    """
    Top-level function for multiprocessing usage, with optimized batch processing.
    """
    chunk_lines, logp_lang, unk_token, whitespace_boundaries, pretok, max_token_len = args

    c_dict = defaultdict(float)
    total = 0

    batch_size = 50

    for i in range(0, len(chunk_lines), batch_size):
        batch = chunk_lines[i:i+batch_size]

        for line in batch:
            tokens = segment_text_viterbi(
                line,
                logp_lang,
                unk_token=unk_token,
                max_sub_len=max_token_len,
                pretok=pretok,
                whitespace_boundaries=whitespace_boundaries
            )

            total += len(tokens)
            for tk in tokens:
                c_dict[tk] += 1

    return (dict(c_dict), total)


def _accumulate_usage(train_file,
                      log_plang,
                      unk_token,
                      num_processes=None,
                      mode='soft',
                      whitespace_boundaries=False,
                      pretokenizer=None,
                      byte_level=False):
        """
        For a single language's corpus:
          - segment each line with distribution log_plang
          - collect usage counts c(token), total usage
        We'll do it in parallel to speed up big corpora.
        """
        if num_processes is None:
            num_processes = min(20, multiprocessing.cpu_count()-1)

        lines = []
        logger.info(f"Reading lines from {train_file}...")
        with open(train_file, 'r', encoding='utf-8', errors="replace") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if line:
                    lines.append(line)

        if not lines:
            return (defaultdict(float), 0)

        n = len(lines)
        chunk_size = max(1, n // num_processes)
        chunks = []
        idx = 0
        while idx < n:
            chunk = lines[idx:idx + chunk_size]
            idx += chunk_size
            chunks.append(chunk)

        max_token_len = MAX_BYTE_TOKEN_LEN if byte_level else MAX_CHAR_TOKEN_LEN
        worker_args = []
        for ch in chunks:
            worker_args.append(
                (ch, log_plang, unk_token, whitespace_boundaries, pretokenizer, max_token_len)
            )

        with multiprocessing.Pool(processes=num_processes) as pool:
            if mode == 'hard':
                results = pool.map(_accumulate_usage_worker_hard, worker_args)
            else:
                results = pool.map(_accumulate_usage_worker_soft, worker_args)

        c_dict = defaultdict(float)
        total = 0
        for partial_cdict, partial_total in results:
            total += partial_total
            for tk, cnt in partial_cdict.items():
                c_dict[tk] += cnt

        return (c_dict, total)
