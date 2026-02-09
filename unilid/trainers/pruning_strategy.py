import os
import hashlib
import multiprocessing
import platform
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np

from unilid.algorithms.loss import _compute_loss_with_alternative_sequences
from unilid.pruning import compute_intersection_pruning_tokens
from unilid.metadata import _save_tokenizer_metadata

import logging
logger = logging.getLogger(__name__)


class PruningStrategyMixin:
    """Token pruning, loss computation, training metadata."""

    def _compute_token_loss(
        self,
        token: str,
        counts_dict: Dict[str, float],
        lprob_dict: Dict[str, float],
        alternatives_dict: Optional[Dict] = None
    ) -> float:
        """
        Compute loss for a single token using the configured pruning criterion.
        """
        if self.pruning_criterion == "probability":
            return np.exp(lprob_dict[token])
        elif self.pruning_criterion == "approx_loss":
            return counts_dict[token] * lprob_dict[token]
        else:  # exact_loss
            return _compute_loss_with_alternative_sequences(
                token, counts_dict, alternatives_dict, lprob_dict, self.unk_token
            )

    def _compute_standard_pruning_scores(
        self,
        state,
        counts: Dict[str, float],
        alternative_sequences: Optional[Dict] = None
    ) -> List[Tuple[str, float]]:
        """
        Compute pruning scores for standard (non-multigram) case.
        """
        under_consideration = state.all_tokens - state.pinned
        return [
            (tk, self._compute_token_loss(tk, counts, state.global_lprobs, alternative_sequences))
            for tk in under_consideration
        ]

    def _compute_multigram_pruning_scores(
        self,
        state,
        per_lang_losses: Dict[str, List[Tuple[str, float]]],
        dataset_sizes: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Compute aggregated pruning scores from pre-computed per-language losses.
        """
        per_lang_loss_dict = {
            lc: {token: loss for token, loss in losses}
            for lc, losses in per_lang_losses.items()
        }

        under_consideration = set(per_lang_loss_dict[state.lang_codes[0]].keys())

        aggregated_losses = []
        for tk in under_consideration:
            total_loss = 0.0
            for lc in state.lang_codes:
                lang_loss = per_lang_loss_dict[lc][tk]

                if self.dataset_size_normalization and dataset_sizes and dataset_sizes.get(lc, 0) > 0:
                    normalized_lang_loss = lang_loss / dataset_sizes[lc]
                else:
                    normalized_lang_loss = lang_loss

                total_loss += self.mixture_weights[lc] * normalized_lang_loss

            aggregated_losses.append((tk, total_loss))

        return aggregated_losses

    def _compute_per_language_losses(
        self,
        per_lang_lprobs: Dict[str, Dict[str, float]],
        per_lang_counts: Dict[str, Dict[str, float]],
        under_consideration: Set[str],
        lang_alternative_sequences: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute raw per-language losses without aggregation for intersection pruning.
        """
        return {
            lc: [
                (tk, self._compute_token_loss(
                    tk, per_lang_counts[lc], per_lang_lprobs[lc],
                    lang_alternative_sequences[lc] if lang_alternative_sequences else None
                ))
                for tk in under_consideration
            ]
            for lc in self.lang_codes
        }

    def _compute_intersection_pruning_tokens(self, under_consideration, per_lang_losses, k_drop):
        """
        Find tokens to prune using rank-based intersection approach.
        """
        return compute_intersection_pruning_tokens(
            per_lang_losses, k_drop, self.lang_codes, under_consideration
        )

    @staticmethod
    def _tokens_to_remove(token_loss: List[Tuple[str, float]], k_drop: int):
        if k_drop <= 0:
            return set()
        token_loss.sort(key=lambda x: x[1])
        to_remove = {tk for tk, _ in token_loss[:k_drop]}
        return to_remove

    def _log_iteration_diagnostics(
        self,
        iteration: int,
        state,
        prev_global_lprobs: Dict[str, float],
        prev_per_lang_lprobs: Optional[Dict[str, Dict[str, float]]],
        observed_tokens: Set[str],
        unk_count: float,
        total_counts: float
    ) -> None:
        """Log diagnostic information for the current EM iteration."""
        logger.info(f"Observed {len(observed_tokens)} unique tokens out of {len(state.all_tokens)} possible tokens across corpus...")
        logger.info(f"Observed {unk_count} UNK tokens out of {total_counts}...")

        unk_prob = np.exp(state.global_lprobs.get(self.unk_token, -np.inf))
        l1_delta = 0.5 * sum(abs(np.exp(state.global_lprobs.get(tk, -np.inf)) - np.exp(prev_global_lprobs.get(tk, -np.inf))) for tk in state.all_tokens)
        logger.info(" UNK prob: %.8g | Δ-dist: %.3e | min=%.3e max=%.3e",
                   unk_prob, l1_delta, np.exp(min(state.global_lprobs.values())), np.exp(max(state.global_lprobs.values())))

        if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
            self._training_metadata["training_metrics"]["convergence"]["unk_probs"].append(float(unk_prob))
            self._training_metadata["training_metrics"]["convergence"]["l1_deltas"].append(float(l1_delta))

        if state.per_lang:
            for lc in state.lang_codes:
                up = np.exp(state.per_lang_lprobs[lc][self.unk_token])
                d = 0.5 * sum(abs(np.exp(state.per_lang_lprobs[lc].get(tk, -np.inf)) - np.exp(prev_per_lang_lprobs[lc].get(tk, -np.inf)))
                             for tk in state.all_tokens)
                logger.info("      [%s] UNK=%.8g | Δ-dist=%.3e", lc, up, d)

    def _add_metadata_to_saved_tokenizer(self, output_path):
        """Save metadata to a separate .metadata.json file alongside the tokenizer"""
        try:
            metadata_path = _save_tokenizer_metadata(output_path, self._training_metadata)
            if metadata_path:
                logger.debug(f"Saved metadata to {metadata_path}")
            else:
                logger.warning(f"Failed to save metadata for {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save metadata for {output_path}: {e}")

    def _collect_training_metadata(self, corpus_files, corpus_info=None):
        """Collect metadata about training configuration and corpus"""
        metadata = {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "tokenizer_class": self.__class__.__name__,
            "tokenizer_type": self._get_tokenizer_type(),
        }

        training_config = {
            "vocab_size": self.vocab_size,
            "em_mode": self.em_mode,
            "num_iterations": self.num_iterations,
            "multigram": self.multigram,
            "byte_level": self.byte_level,
            "dataset_size_normalization": self.dataset_size_normalization,
            "use_intersection_pruning": self.use_intersection_pruning,
            "pruning_criterion": self.pruning_criterion,
            "special_tokens": list(self.special_tokens)
        }

        if self.initial_vocab_tokens:
            training_config["initial_vocab_source"] = "custom_file"
            training_config["initial_vocab_file"] = self.initial_vocab_tokens
        else:
            training_config["initial_vocab_source"] = "hf_generated"

        if self.multigram:
            training_config["mixture_weights"] = dict(self.mixture_weights)

        metadata["training_config"] = training_config

        corpus_metadata = self._collect_corpus_metadata(corpus_files, corpus_info)
        metadata["corpus_info"] = corpus_metadata

        metadata["environment_info"] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "num_processes": multiprocessing.cpu_count()
        }

        return metadata

    def _collect_corpus_metadata(self, corpus_files, corpus_info=None):
        """Collect metadata about training corpus"""
        corpus_metadata = {
            "training_files": corpus_files or [],
            "total_files": len(corpus_files) if corpus_files else 0,
        }

        if corpus_info:
            corpus_metadata["language_codes"] = list(corpus_info.keys())
            corpus_metadata["per_language_files"] = dict(corpus_info)

        corpus_sizes = {}
        file_checksums = {}
        total_lines = 0

        files_to_check = corpus_files or []
        if corpus_info:
            files_to_check = list(corpus_info.values())

        for file_path in files_to_check:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    file_checksums[file_path] = file_hash

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f)
                    corpus_sizes[file_path] = lines
                    total_lines += lines

        corpus_metadata["corpus_sizes"] = corpus_sizes
        corpus_metadata["file_checksums"] = file_checksums
        corpus_metadata["total_lines"] = total_lines

        return corpus_metadata

    def _update_training_results(self, training_metrics):
        """Update training results in metadata"""
        self._training_metadata["training_results"] = {
            "initial_vocab_size": training_metrics.get("initial_vocab_size"),
            "final_vocab_size": len(self.global_lprobs) if hasattr(self, 'global_lprobs') else None,
            "tokens_pruned_per_iteration": training_metrics.get("pruned_per_iteration", []),
            "convergence_metrics": training_metrics.get("convergence", {}),
            "training_time_seconds": training_metrics.get("training_time"),
            "final_loss": training_metrics.get("final_loss")
        }
