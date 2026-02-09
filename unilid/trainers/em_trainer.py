import logging

from unilid.algorithms.accumulate import _accumulate_usage
from unilid.constants import MIN_TOKEN_PROB

logger = logging.getLogger(__name__)


class EMUnigramTrainer:
    """
    EM-based trainer for a fixed vocabulary Unigram distribution.
    Uses the same EM approach as StandardUnigramLMTokenizer for consistency.
    """

    def __init__(self, vocab, unk_token="<unk>", special_tokens=None,
                 max_iterations=20, em_mode="hard", convergence_threshold=1e-6,
                 whitespace_token_boundaries=True,
                 pretokenizer=None, byte_level=False):
        self.vocab = vocab
        self.unk_token = unk_token
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>"]
        self.max_iterations = max_iterations
        self.em_mode = em_mode
        self.convergence_threshold = convergence_threshold
        self.whitespace_token_boundaries = whitespace_token_boundaries
        self.pretokenizer = pretokenizer
        self.byte_level = byte_level

        if self.em_mode not in ["hard", "soft"]:
            logger.warning(f"Invalid em_mode '{em_mode}', defaulting to 'hard'")
            self.em_mode = "hard"

        self.vocab_set = set(self.vocab.keys())

        for tok in self.special_tokens:
            if tok not in self.vocab_set:
                logger.info(f"Adding missing special token '{tok}' to vocab.")
                self.vocab[tok] = len(self.vocab)
                self.vocab_set.add(tok)

        self.token_probs = None

    def train(self, corpus_file, num_processes=None):
        """
        Run the EM algorithm over `corpus_file`.

        Returns:
            dict(token -> float probability)
        """
        logger.info(f"Starting EM training on {corpus_file} with mode={self.em_mode}")

        vocab_size = len(self.vocab)
        p_token = {t: 1.0 / vocab_size for t in self.vocab}

        prev_p_token = None
        for iteration in range(self.max_iterations):
            logger.info(f"EM iteration {iteration+1}/{self.max_iterations}")

            usage_counts = {tk: 0.0 for tk in self.vocab}

            if iteration > 0:
                prev_p_token = p_token.copy()

            c_dict, total_subwords = _accumulate_usage(
                corpus_file,
                p_token,
                self.unk_token,
                num_processes=num_processes,
                mode=self.em_mode,
                whitespace_boundaries=self.whitespace_token_boundaries,
                pretokenizer=self.pretokenizer,
                byte_level=self.byte_level
            )
            usage_counts.update(c_dict)
            unk_occ = usage_counts[self.unk_token]
            unk_percent = unk_occ/total_subwords
            logger.info(f"{unk_occ} UNK counts out of {total_subwords} total subwords ({unk_percent:.5f}%)")

            if total_subwords > 0:
                for tk in p_token:
                    p_token[tk] = usage_counts[tk] / total_subwords
            else:
                logger.warning("No usage => fallback to uniform distribution.")
                for tk in self.vocab:
                    p_token[tk] = 1.0 / len(self.vocab)

            if prev_p_token and iteration > 0:
                diffs = [abs(p_token[tk] - prev_p_token[tk]) for tk in self.vocab]
                total_diff = sum(diffs)
                logger.info(f"Iteration {iteration+1} parameter change: {total_diff:.8f}")

                if total_diff < self.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration+1}, total_diff={total_diff:.8g}")
                    break

        total_prob = sum(p_token.values())
        if total_prob > 0:
            p_token = {tk: prob/total_prob for tk, prob in p_token.items()}

        for tk in self.vocab:
            p_token[tk] = max(p_token[tk], MIN_TOKEN_PROB)

        logger.info(f"Resetting UNK prob to {p_token[self.unk_token]}")
        logger.info(f"EM training completed with {self.max_iterations} iterations")

        return p_token
