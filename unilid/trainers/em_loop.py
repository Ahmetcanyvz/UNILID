from __future__ import annotations

import copy
import math
import os
import re
import logging
import tempfile
import subprocess
import shutil
import atexit
import multiprocessing
import random
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

import numpy as np

try:
    import sentencepiece as spm
except Exception:
    spm = None

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer

from unilid import constants
from unilid.algorithms.accumulate import _accumulate_usage
from unilid.algorithms.loss import _compute_alternatives_direct
from unilid.vocab_io import _process_file_worker, _write_sp_seed_vocab_file
from unilid.encoding import get_baseline_characters, get_baseline_bytes
from unilid.vocab_io import _load_custom_vocab_from_file

logger = logging.getLogger(__name__)


@dataclass
class EMState:
    """Encapsulates state needed during EM training iterations."""
    corpus_files: List[str]
    per_lang: bool
    lang_codes: List[str]
    all_tokens: Set[str]  # Mutable - tokens removed during pruning
    pinned: Set[str]
    global_lprobs: Dict[str, float]
    per_lang_lprobs: Optional[Dict[str, Dict[str, float]]]
    k_total: int  # Mutable - decremented during pruning
    k_step: List[int]


class EMLoopMixin:
    """EM iteration loop, state initialization, SentencePiece integration."""

    def _run_em_sp(
        self,
        corpus_files: Optional[List[str]],
        per_lang: bool,
        corpus_info: Optional[Dict[str, str]] = None,
        logging_path: str = None
    ) -> None:
        if self.use_sp_em or self.use_sp_seed_vocab:
            if per_lang:
                self.dataset_size_normalization = True
            self._temp_dir = tempfile.mkdtemp()
            processed_file_paths = None
            atexit.register(shutil.rmtree, self._temp_dir, ignore_errors=True)
            text_files = corpus_files if not per_lang else list(corpus_info.values())

            num_workers = min(min(os.cpu_count() or 1, 20), len(text_files))
            logger.info(f"Using {num_workers} worker processes.")

            worker_args = [(path, self._temp_dir, self._get_pretokenizer(), i) for i, path in enumerate(text_files)]

            with multiprocessing.Pool(processes=num_workers) as pool:
                processed_file_paths = pool.map(_process_file_worker, worker_args)
                processed_mapping = {orig: processed for orig, processed in zip(text_files, processed_file_paths)}

            state = self._initialize_em_state(corpus_files, per_lang, corpus_info, processed_mapping)
        else:
            state = self._initialize_em_state(corpus_files, per_lang, corpus_info)

        if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
            self._training_metadata["training_metrics"]["initial_vocab_size"] = len(state.all_tokens)

        num_sub_iterations = 2
        shrinking_factor = 0.75
        target_vocab_size = self.vocab_size
        current_vocab_size = len(state.all_tokens)

        pruning_iteration = 0

        while current_vocab_size > target_vocab_size:
            pruning_iteration += 1

            logger.info("SP Pruning Iteration %d – Vocab=%d | Target=%d",
                       pruning_iteration, current_vocab_size, target_vocab_size)

            prev_lprobs, prev_lang_lprobs = None, None

            for sub_it in range(num_sub_iterations):
                logger.info("  EM sub-iteration %d/%d", sub_it + 1, num_sub_iterations)

                prev_lprobs, prev_lang_lprobs = copy.deepcopy(state.global_lprobs), copy.deepcopy(state.per_lang_lprobs)

                if self.use_sp_em:
                    logger.info("Using SentencePiece for EM computation...")
                    if state.per_lang:
                        for lc, f in zip(state.lang_codes, state.corpus_files):
                            state.per_lang_lprobs[lc] = self._sp_em(state.per_lang_lprobs[lc], f)
                        state.global_lprobs = {
                            tk: np.logaddexp.reduce([self.log_mixture_weights[lc] + state.per_lang_lprobs[lc].get(tk, -np.inf)
                                    for lc in state.lang_codes], axis=-1)
                            for tk in state.all_tokens
                        }
                        observed_tokens = set(state.all_tokens)
                        lang_counts = {lc: {tk: np.exp(lp) for tk, lp in state.per_lang_lprobs[lc].items()} for lc in state.lang_codes}
                        unk_count = sum(lang_counts[lc].get(self.unk_token, 0.0) for lc in state.lang_codes)
                        total_counts = 1.0
                    else:
                        state.global_lprobs = self._sp_em(state.global_lprobs, ','.join(state.corpus_files))
                        counts = {tk: np.exp(lp) for tk, lp in state.global_lprobs.items()}
                        observed_tokens = set(state.all_tokens)
                        unk_count = counts.get(self.unk_token, 0.0)
                        total_counts = 1.0
                else:
                    logger.info("Using custom implementation for EM computation...")

                    if state.per_lang:
                        lang_counts = {lc: {tk: 0.0 for tk in state.all_tokens} for lc in state.lang_codes}
                        unk_count = 0
                        total_counts = 0
                        for lc, file in zip(state.lang_codes, state.corpus_files):
                            c_dict, _ = _accumulate_usage(file,
                                                          state.per_lang_lprobs[lc],
                                                          self.unk_token,
                                                          mode=self.em_mode,
                                                          whitespace_boundaries=self.whitespace_token_boundaries,
                                                          pretokenizer=self._get_pretokenizer())
                            for tk, v in c_dict.items():
                                lang_counts[lc][tk] += v
                            unk_count += lang_counts[lc][self.unk_token]
                            total_counts += sum(lang_counts[lc].values())
                        observed_tokens = set([k for counts in lang_counts.values() for k in counts])
                        for lc in state.lang_codes:
                            state.per_lang_lprobs[lc] = self._counts_to_log_probs(lang_counts[lc])
                        state.global_lprobs = {tk: np.logaddexp.reduce(
                            [self.log_mixture_weights[lc] + state.per_lang_lprobs[lc].get(tk, -np.inf) for lc in state.lang_codes]
                            ) for tk in state.all_tokens}
                    else:
                        counts = {tk: 0.0 for tk in state.all_tokens}
                        for file in state.corpus_files:
                            c_dict, _ = _accumulate_usage(file, state.global_lprobs, self.unk_token,
                                                          mode=self.em_mode,
                                                          whitespace_boundaries=self.whitespace_token_boundaries,
                                                          pretokenizer=self._get_pretokenizer())
                            for tk, v in c_dict.items():
                                counts[tk] += v
                        unk_count = counts[self.unk_token]
                        total_counts = sum(counts.values())
                        observed_tokens = counts.keys()
                        state.global_lprobs = self._counts_to_log_probs(counts)

                synthetic_step = (pruning_iteration - 1) * num_sub_iterations + sub_it
                self._log_iteration_diagnostics(
                    synthetic_step,
                    state,
                    prev_lprobs,
                    prev_lang_lprobs,
                    observed_tokens,
                    unk_count,
                    total_counts
                )

            # --- Pruning step ---
            num_candidates = current_vocab_size - len(state.pinned)
            num_to_keep = int(num_candidates * shrinking_factor) + len(state.pinned)
            new_target_size = max(target_vocab_size, num_to_keep)

            k_drop = current_vocab_size - new_target_size

            if k_drop <= 0:
                logger.info("Pruning finished, vocab size %d reached target %d",
                           current_vocab_size, target_vocab_size)
                break

            logger.info("SP Pruning Iteration %d – Pruning %d tokens (from %d to %d)",
                       pruning_iteration, k_drop, current_vocab_size, new_target_size)

            logging_dict = {}
            if state.per_lang:
                counts_for_logging = {tk: sum(lang_counts[lc].get(tk, 0.0) for lc in state.lang_codes) for tk in state.all_tokens}
            else:
                counts_for_logging = counts

            for tk in state.all_tokens:
                logging_dict[tk] = {
                    "pre-pruning-lprob": state.global_lprobs[tk],
                    "status": "pinned" if tk in state.pinned else "kept",
                    "loss": "NA",
                    "post-pruning-prob": "NA",
                    "expected-counts": counts_for_logging.get(tk, 0.0)
                }
            if state.per_lang:
                lang_alternative_sequences = {}
                if self.pruning_criterion == "exact_loss":
                    for lc, file in zip(state.lang_codes, state.corpus_files):
                        lang_alternative_sequences[lc] = _compute_alternatives_direct(state.per_lang_lprobs[lc], self.unk_token, byte_level=self.byte_level)
            else:
                alternative_sequences = _compute_alternatives_direct(state.global_lprobs, self.unk_token, byte_level=self.byte_level) if self.pruning_criterion != "probability" else None

            before = len(state.all_tokens)
            under_consideration = state.all_tokens - state.pinned
            if state.per_lang:
                per_lang_losses = self._compute_per_language_losses(
                    state.per_lang_lprobs, lang_counts, under_consideration, lang_alternative_sequences
                )
                if self.use_intersection_pruning:
                    tokens_to_remove = self._compute_intersection_pruning_tokens(
                        under_consideration, per_lang_losses, k_drop
                    )
                else:
                    dataset_sizes = {}
                    if self.dataset_size_normalization and not self.use_sp_em:
                        dataset_sizes = {lc: sum(lang_counts[lc].values()) for lc in state.lang_codes}
                        logger.info("Dataset sizes for normalization: %s",
                                   {lc: f"{size:.0f}" for lc, size in dataset_sizes.items()})
                    loss = self._compute_multigram_pruning_scores(
                        state, per_lang_losses, dataset_sizes
                    )

                    for tk, total_loss in loss:
                        logging_dict[tk]["loss"] = total_loss

                    tokens_to_remove = self._tokens_to_remove(loss, k_drop)

                for tk in tokens_to_remove:
                    logging_dict[tk]["status"] = "pruned"
            else:
                loss = self._compute_standard_pruning_scores(
                    state, counts, alternative_sequences
                )

                for tk, tk_loss in loss:
                    logging_dict[tk]["loss"] = tk_loss

                tokens_to_remove = self._tokens_to_remove(loss, k_drop)
                for tk in tokens_to_remove:
                    logging_dict[tk]["status"] = "pruned"

            state.all_tokens -= tokens_to_remove
            pruned = before - len(state.all_tokens)
            logger.info("   tokens_pruned=%d | new_vocab=%d", pruned, len(state.all_tokens))

            if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
                self._training_metadata["training_metrics"]["pruned_per_iteration"].append(pruned)

            if state.per_lang:
                for lc in state.lang_codes:
                    state.per_lang_lprobs[lc] = {tk: state.per_lang_lprobs[lc].get(tk, -np.inf) for tk in state.all_tokens}
                    state.per_lang_lprobs[lc] = self._log_normalize(state.per_lang_lprobs[lc])

            state.global_lprobs = {tk: state.global_lprobs[tk] for tk in state.all_tokens}
            state.global_lprobs = self._log_normalize(state.global_lprobs)

            for tk, new_lprob in state.global_lprobs.items():
                if tk in logging_dict:
                    logging_dict[tk]["post-pruning-prob"] = np.exp(new_lprob)

            if logging_path is not None:
                logging_file = f"{logging_path}_prune_{pruning_iteration}.pkl"
                with open(logging_file, 'wb') as handle:
                    pickle.dump(logging_dict, handle)

            current_vocab_size = len(state.all_tokens)

            assert all(x in state.global_lprobs for x in state.pinned)

        # --- Final Pruning Step ---
        k_drop_final = len(state.all_tokens) - target_vocab_size
        if k_drop_final > 0:
            logger.info("Final pruning step: removing %d tokens to reach exact size %d",
                       k_drop_final, target_vocab_size)

            logger.info("Running E-step to get accurate losses for final prune...")
            prev_lprobs, prev_lang_lprobs = copy.deepcopy(state.global_lprobs), copy.deepcopy(state.per_lang_lprobs)
            if self.use_sp_em:
                if state.per_lang:
                    for lc, f in zip(state.lang_codes, state.corpus_files):
                        state.per_lang_lprobs[lc] = self._sp_em(state.per_lang_lprobs[lc], f)
                    lang_counts = {lc: {tk: np.exp(lp) for tk, lp in state.per_lang_lprobs[lc].items()} for lc in state.lang_codes}
                else:
                    state.global_lprobs = self._sp_em(state.global_lprobs, ','.join(state.corpus_files))
                    counts = {tk: np.exp(lp) for tk, lp in state.global_lprobs.items()}
            else:
                if state.per_lang:
                    lang_counts = {lc: {tk: 0.0 for tk in state.all_tokens} for lc in state.lang_codes}
                    for lc, file in zip(state.lang_codes, state.corpus_files):
                        c_dict, _ = _accumulate_usage(file, state.per_lang_lprobs[lc], self.unk_token, mode=self.em_mode, whitespace_boundaries=self.whitespace_token_boundaries, pretokenizer=self._get_pretokenizer())
                        for tk, v in c_dict.items():
                            lang_counts[lc][tk] += v
                else:
                    counts = {tk: 0.0 for tk in state.all_tokens}
                    for file in state.corpus_files:
                        c_dict, _ = _accumulate_usage(file, state.global_lprobs, self.unk_token, mode=self.em_mode, whitespace_boundaries=self.whitespace_token_boundaries, pretokenizer=self._get_pretokenizer())
                        for tk, v in c_dict.items():
                            counts[tk] += v

            logger.info("Calculating final losses...")
            under_consideration = state.all_tokens - state.pinned
            if state.per_lang:
                lang_alternative_sequences = {}
                if self.pruning_criterion == "exact_loss":
                    for lc, file in zip(state.lang_codes, state.corpus_files):
                        lang_alternative_sequences[lc] = _compute_alternatives_direct(state.per_lang_lprobs[lc], self.unk_token, byte_level=self.byte_level)
                per_lang_losses = self._compute_per_language_losses(state.per_lang_lprobs, lang_counts, under_consideration, lang_alternative_sequences)
                if self.use_intersection_pruning:
                    tokens_to_remove = self._compute_intersection_pruning_tokens(under_consideration, per_lang_losses, k_drop_final)
                else:
                    dataset_sizes = {}
                    if self.dataset_size_normalization and not self.use_sp_em:
                         dataset_sizes = {lc: sum(lang_counts[lc].values()) for lc in state.lang_codes}
                    loss = self._compute_multigram_pruning_scores(state, per_lang_losses, dataset_sizes)
                    tokens_to_remove = self._tokens_to_remove(loss, k_drop_final)
            else:
                alternative_sequences = _compute_alternatives_direct(state.global_lprobs, self.unk_token, byte_level=self.byte_level) if self.pruning_criterion != "probability" else None
                loss = self._compute_standard_pruning_scores(state, counts, alternative_sequences)
                tokens_to_remove = self._tokens_to_remove(loss, k_drop_final)

            state.all_tokens -= tokens_to_remove
            logger.info("   final_tokens_pruned=%d | final_vocab=%d", len(tokens_to_remove), len(state.all_tokens))

        # --- Final EM run ---
        logger.info("Running final EM pass on the final vocabulary...")
        prev_lprobs, prev_lang_lprobs = copy.deepcopy(state.global_lprobs), copy.deepcopy(state.per_lang_lprobs)
        if self.use_sp_em:
            if state.per_lang:
                for lc, f in zip(state.lang_codes, state.corpus_files):
                    state.per_lang_lprobs[lc] = self._sp_em(state.per_lang_lprobs[lc], f)
                state.global_lprobs = {
                    tk: np.logaddexp.reduce([self.log_mixture_weights[lc] + state.per_lang_lprobs[lc].get(tk, -np.inf)
                            for lc in state.lang_codes], axis=-1)
                    for tk in state.all_tokens
                }
            else:
                state.global_lprobs = self._sp_em(state.global_lprobs, ','.join(state.corpus_files))
        else:
            if state.per_lang:
                lang_counts = {lc: {tk: 0.0 for tk in state.all_tokens} for lc in state.lang_codes}
                for lc, file in zip(state.lang_codes, state.corpus_files):
                    c_dict, _ = _accumulate_usage(file, state.per_lang_lprobs[lc], self.unk_token, mode=self.em_mode, whitespace_boundaries=self.whitespace_token_boundaries, pretokenizer=self._get_pretokenizer())
                    for tk, v in c_dict.items():
                        lang_counts[lc][tk] += v
                for lc in state.lang_codes:
                    state.per_lang_lprobs[lc] = self._counts_to_log_probs(lang_counts[lc])
                state.global_lprobs = {tk: np.logaddexp.reduce(
                    [self.log_mixture_weights[lc] + state.per_lang_lprobs[lc].get(tk, -np.inf) for lc in state.lang_codes]
                    ) for tk in state.all_tokens}
            else:
                counts = {tk: 0.0 for tk in state.all_tokens}
                for file in state.corpus_files:
                    c_dict, _ = _accumulate_usage(file, state.global_lprobs, self.unk_token, mode=self.em_mode, whitespace_boundaries=self.whitespace_token_boundaries, pretokenizer=self._get_pretokenizer())
                    for tk, v in c_dict.items():
                        counts[tk] += v
                state.global_lprobs = self._counts_to_log_probs(counts)

        # Final M-Step (Normalization)
        if state.per_lang:
             for lc in state.lang_codes:
                 state.per_lang_lprobs[lc] = self._log_normalize(state.per_lang_lprobs[lc])
        state.global_lprobs = self._log_normalize(state.global_lprobs)

        logger.info(f"Final EM pass complete. Final vocab size: {len(state.all_tokens)}")

        return state.global_lprobs

    def _run_em(
        self,
        corpus_files: Optional[List[str]],
        per_lang: bool,
        corpus_info: Optional[Dict[str, str]] = None,
        logging_path: str = None
    ) -> None:
        if self.use_sp_em or self.use_sp_seed_vocab:
            if per_lang:
                self.dataset_size_normalization = True
            self._temp_dir = tempfile.mkdtemp()
            processed_file_paths = None
            atexit.register(shutil.rmtree, self._temp_dir, ignore_errors=True)
            text_files = corpus_files if not per_lang else list(corpus_info.values())

            num_workers = min(min(os.cpu_count() or 1, 20), len(text_files))
            logger.info(f"Using {num_workers} worker processes.")

            worker_args = [(path, self._temp_dir, self._get_pretokenizer(), i) for i, path in enumerate(text_files)]

            with multiprocessing.Pool(processes=num_workers) as pool:
                processed_file_paths = pool.map(_process_file_worker, worker_args)
                processed_mapping = {orig: processed for orig, processed in zip(text_files, processed_file_paths)}

            state = self._initialize_em_state(corpus_files, per_lang, corpus_info, processed_mapping)
        else:
            state = self._initialize_em_state(corpus_files, per_lang, corpus_info)

        if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
            self._training_metadata["training_metrics"]["initial_vocab_size"] = len(state.all_tokens)

        prev_lprobs, prev_lang_lprobs = None, None
        for it in range(self.num_iterations):
            logger.info("EM %d/%d – starting | vocab=%d | to trim=%d | Pruning criterion: %s",
                       it + 1, self.num_iterations, len(state.all_tokens), state.k_total, self.pruning_criterion)
            prev_lprobs, prev_lang_lprobs = copy.deepcopy(state.global_lprobs), copy.deepcopy(state.per_lang_lprobs)
            if self.use_sp_em:
                logger.info("Using SentencePiece for EM computation...")
                if state.per_lang:
                    for lc, f in zip(state.lang_codes, state.corpus_files):
                        state.per_lang_lprobs[lc] = self._sp_em(state.per_lang_lprobs[lc], f)
                    state.global_lprobs = {
                        tk: np.logaddexp.reduce([self.log_mixture_weights[lc] + state.per_lang_lprobs[lc].get(tk, -np.inf)
                                for lc in state.lang_codes], axis=-1)
                        for tk in state.all_tokens
                    }
                    observed_tokens = set(state.all_tokens)
                    lang_counts = {lc: {tk: np.exp(lp) for tk, lp in state.per_lang_lprobs[lc].items()} for lc in state.lang_codes}
                    unk_count = sum(lang_counts[lc].get(self.unk_token, 0.0) for lc in state.lang_codes)
                    total_counts = 1.0
                else:
                    state.global_lprobs = self._sp_em(state.global_lprobs, ','.join(state.corpus_files))
                    counts = {tk: np.exp(lp) for tk, lp in state.global_lprobs.items()}
                    observed_tokens = set(state.all_tokens)
                    unk_count = counts.get(self.unk_token, 0.0)
                    total_counts = 1.0
            else:
                logger.info("Using custom implementation for EM computation...")

                if state.per_lang:
                    lang_counts = {lc: {tk: 0.0 for tk in state.all_tokens} for lc in state.lang_codes}
                    unk_count = 0
                    total_counts = 0
                    for lc, file in zip(state.lang_codes, state.corpus_files):
                        c_dict, _ = _accumulate_usage(file,
                                                      state.per_lang_lprobs[lc],
                                                      self.unk_token,
                                                      mode=self.em_mode,
                                                      whitespace_boundaries=self.whitespace_token_boundaries,
                                                      pretokenizer=self._get_pretokenizer())
                        for tk, v in c_dict.items():
                            lang_counts[lc][tk] += v
                        unk_count += lang_counts[lc][self.unk_token]
                        total_counts += sum(lang_counts[lc].values())
                    observed_tokens = set([k for counts in lang_counts.values() for k in counts])
                    for lc in state.lang_codes:
                        state.per_lang_lprobs[lc] = self._counts_to_log_probs(lang_counts[lc])
                    state.global_lprobs = {tk: np.logaddexp.reduce(
                        [self.log_mixture_weights[lc] + state.per_lang_lprobs[lc].get(tk, -np.inf) for lc in state.lang_codes]
                        ) for tk in state.all_tokens}
                else:
                    counts = {tk: 0.0 for tk in state.all_tokens}
                    for file in state.corpus_files:
                        c_dict, _ = _accumulate_usage(file, state.global_lprobs, self.unk_token,
                                                      mode=self.em_mode,
                                                      whitespace_boundaries=self.whitespace_token_boundaries,
                                                      pretokenizer=self._get_pretokenizer())
                        for tk, v in c_dict.items():
                            counts[tk] += v
                    unk_count = counts[self.unk_token]
                    total_counts = sum(counts.values())
                    observed_tokens = counts.keys()
                    state.global_lprobs = self._counts_to_log_probs(counts)

            self._log_iteration_diagnostics(it,
                                            state,
                                            prev_lprobs,
                                            prev_lang_lprobs,
                                            observed_tokens,
                                            unk_count,
                                            total_counts)

            # -------------------- pruning ----------------------------------
            k_drop = state.k_step[it]
            k_drop = min(k_drop, len(state.all_tokens) - self.vocab_size)
            if k_drop > 0:
                logging_dict = {}
                for tk in state.all_tokens:
                    logging_dict[tk] = {
                        "pre-pruning-lprob": state.global_lprobs[tk],
                        "status": "pinned" if tk in state.pinned else "kept",
                        "loss": "NA",
                        "post-pruning-prob": "NA",
                        "expected-counts": sum([lang_counts[lc][tk] for lc in state.lang_codes]) if state.per_lang else counts[tk]
                    }
                if state.per_lang:
                    lang_alternative_sequences = {}
                    if self.pruning_criterion == "exact_loss":
                        for lc, file in zip(state.lang_codes, state.corpus_files):
                            lang_alternative_sequences[lc] = _compute_alternatives_direct(state.per_lang_lprobs[lc], self.unk_token, byte_level=self.byte_level)
                else:
                    alternative_sequences = _compute_alternatives_direct(state.global_lprobs, self.unk_token, byte_level=self.byte_level) if self.pruning_criterion != "probability" else None

                before = len(state.all_tokens)
                under_consideration = state.all_tokens - state.pinned
                if state.per_lang:
                    per_lang_losses = self._compute_per_language_losses(
                        state.per_lang_lprobs, lang_counts, under_consideration, lang_alternative_sequences
                    )
                    if self.use_intersection_pruning:
                        tokens_to_remove = self._compute_intersection_pruning_tokens(
                            under_consideration, per_lang_losses, k_drop
                        )
                    else:
                        dataset_sizes = {}
                        if self.dataset_size_normalization and not self.use_sp_em:
                            dataset_sizes = {lc: sum(lang_counts[lc].values()) for lc in state.lang_codes}
                            logger.info("Dataset sizes for normalization: %s",
                                       {lc: f"{size:.0f}" for lc, size in dataset_sizes.items()})
                        loss = self._compute_multigram_pruning_scores(
                            state, per_lang_losses, dataset_sizes
                        )

                        for tk, total_loss in loss:
                            logging_dict[tk]["loss"] = total_loss

                        tokens_to_remove = self._tokens_to_remove(loss, k_drop)

                    for tk in tokens_to_remove:
                        logging_dict[tk]["status"] = "pruned"
                else:
                    loss = self._compute_standard_pruning_scores(
                        state, counts, alternative_sequences
                    )

                    for tk, tk_loss in loss:
                        logging_dict[tk]["loss"] = tk_loss

                    tokens_to_remove = self._tokens_to_remove(loss, k_drop)
                    for tk in tokens_to_remove:
                        logging_dict[tk]["status"] = "pruned"

                state.all_tokens -= tokens_to_remove
                pruned = before - len(state.all_tokens)
                state.k_total -= pruned
                logger.info("   tokens_pruned=%d | new_vocab=%d", pruned, len(state.all_tokens))

                if hasattr(self, '_training_metadata') and "training_metrics" in self._training_metadata:
                    self._training_metadata["training_metrics"]["pruned_per_iteration"].append(pruned)

                if state.per_lang:
                    for lc in state.lang_codes:
                        state.per_lang_lprobs[lc] = {tk: state.per_lang_lprobs[lc].get(tk, -np.inf) for tk in state.all_tokens}
                        state.per_lang_lprobs[lc] = self._log_normalize(state.per_lang_lprobs[lc])

                state.global_lprobs = {tk: state.global_lprobs[tk] for tk in state.all_tokens}
                state.global_lprobs = self._log_normalize(state.global_lprobs)
                for tk, new_lprob in state.global_lprobs.items():
                    logging_dict[tk]["post-pruning-prob"] = np.exp(new_lprob)
                if logging_path is not None:
                    logging_file = f"{logging_path}_em_{it}.pkl"
                    with open(logging_file, 'wb') as handle:
                        pickle.dump(logging_dict, handle)

            logger.info(f"Total probability mass: {sum([np.exp(x) for x in state.global_lprobs.values()])}")
            assert all(x in state.global_lprobs for x in state.pinned)
        return state.global_lprobs

    def _initialize_em_state(
        self,
        corpus_files: Optional[List[str]],
        per_lang: bool,
        corpus_info: Optional[Dict[str, str]] = None,
        processed_mapping: Optional[Dict[str, str]] = None
    ) -> EMState:
        """Initialize EM training state and data structures."""
        if per_lang and not corpus_info:
            raise ValueError("corpus_info required when multigram=True")

        lang_codes = []
        if per_lang:
            lang_codes = list(corpus_info.keys())
            corpus_files = [corpus_info[lc] for lc in lang_codes]
            self.lang_codes = lang_codes

        if processed_mapping:
            processed_corpus_files = [processed_mapping[f] for f in corpus_files]

        files_for_init_vocab = processed_corpus_files if self.use_sp_seed_vocab else corpus_files
        all_tokens = self._initial_vocab(files_for_init_vocab)

        baseline = self.baseline_tokens(corpus_files, self.byte_level, self.min_char_occurence_threshold)
        pinned = set(self.special_tokens) | baseline | {self.unk_token}

        all_tokens.update(pinned)

        uniform_lprob = np.log(1.0 / len(all_tokens))
        if per_lang:
            per_lang_lprobs = {lc: {tk: uniform_lprob for tk in all_tokens} for lc in lang_codes}
            self.mixture_weights = self._sanitize_mixture_weights(lang_codes, self.mixture_weights)
            self.log_mixture_weights = {lc: np.log(w) for lc, w in self.mixture_weights.items()}
        else:
            per_lang_lprobs = {}

        global_lprobs = {tk: uniform_lprob for tk in all_tokens}

        k_total = len(all_tokens) - self.vocab_size
        k_step = [math.floor(k_total / constants.NUM_TRIMMING_ITERATIONS)] * constants.NUM_TRIMMING_ITERATIONS + [0] * (self.num_iterations - constants.NUM_TRIMMING_ITERATIONS)

        return EMState(
            corpus_files=corpus_files if not self.use_sp_em else processed_corpus_files,
            per_lang=per_lang,
            lang_codes=lang_codes,
            all_tokens=all_tokens,
            pinned=pinned,
            global_lprobs=global_lprobs,
            per_lang_lprobs=per_lang_lprobs,
            k_total=k_total,
            k_step=k_step
        )

    def _sp_seed_vocab(self, train_files: List[str]) -> Set[str]:
        """
        Use patched SP (num_sub_iterations=0, no seed file) to extract a seed vocabulary
        via suffix-array seeding.
        """
        if spm is None:
            raise RuntimeError("sentencepiece is not installed but use_sp_seed_vocab=True")

        seed_vocab_size = self.vocab_size * constants.INITIAL_VOCAB_MULT_FACTOR

        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "model")

            command = ["spm_train"] + constants.SP_DEFAULT_ARGS + [
                f"--input={','.join(train_files)}",
                f"--model_prefix={out_prefix}",

                f"--seed_sentencepiece_size={seed_vocab_size}",
                f"--max_sentencepiece_length={constants.MAX_BYTE_TOKEN_LEN if self.byte_level else constants.MAX_CHAR_TOKEN_LEN}",
                f"--split_by_whitespace={'true' if self.whitespace_token_boundaries else 'false'}",

                "--num_sub_iterations=0",
                f"--unk_piece={self.unk_token}",
                f"--num_threads={min(20, multiprocessing.cpu_count()-1)}",
                f"--vocab_size={seed_vocab_size}"
            ]

            logger.info(f"Running SentencePiece CLI command:\n{' '.join(command)}")
            try:
                output = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            except subprocess.CalledProcessError as e:
                match = re.search(r'<=\s*(\d+)', e.stderr)
                if match:
                    max_vocab = int(match.group(1))
                    logger.warning(f"Failed to create seed vocabulary with size {seed_vocab_size}. Maximum allowed vocabulary size: {max_vocab}. Trying again...")

                    command[-1] = f"--vocab_size={max_vocab}"
                    output = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
                else:
                    raise e

            logger.info("SentencePiece seed creation successful.")
            logger.info(f"SentencePiece STDOUT: {output.stdout}")
            logger.info(f"SentencePiece STDERR (logs + errors): : {output.stderr}")

            sp = spm.SentencePieceProcessor()
            sp.load(out_prefix + ".model")

            vocab = set()
            for i in range(sp.get_piece_size()):
                p = sp.id_to_piece(i)
                hf_p = p
                exceeds_len = (self.byte_level and len(hf_p.encode("utf-8")) > constants.MAX_BYTE_TOKEN_LEN) or (not self.byte_level and len(hf_p) > constants.MAX_CHAR_TOKEN_LEN)
                if not exceeds_len:
                    vocab.add(hf_p if p not in self.special_tokens else p)

            return vocab

    def _sp_em(
            self,
            vocab_dict: Dict[str, int],
            text_paths: str
        ) -> Dict[str, float]:
            if spm is None:
                raise RuntimeError("spm_train executable not found. Make sure sentencepiece is installed and in your PATH.")

            tmpdir = self._temp_dir
            out_prefix = os.path.join(tmpdir, "model")
            vocab_path = os.path.join(tmpdir, "vocab.tsv")
            token_count = _write_sp_seed_vocab_file(list(vocab_dict.items()), vocab_path, self.unk_token)

            expected_vocab_size = token_count + 3

            command = ["spm_train"] + constants.SP_DEFAULT_ARGS + [
                f"--input={text_paths}",
                f"--model_prefix={out_prefix}",

                f"--seed_sentencepieces_file={vocab_path}",
                f"--vocab_size={expected_vocab_size}",

                f"--num_sub_iterations=1",
                f"--unk_piece={self.unk_token}",
                f"--num_threads={min(20, multiprocessing.cpu_count()-1)}"
            ]

            logger.info(f"Running SentencePiece CLI command:\n{' '.join(command)}")
            output = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            logger.info("SentencePiece training successful.")
            logger.info(f"SentencePiece STDOUT: {output.stdout}")
            logger.info(f"SentencePiece STDERR (logs + errors): : {output.stderr}")

            sp = spm.SentencePieceProcessor()
            sp.load(out_prefix + ".model")

            token_logps = {}
            additional_tokens = set()
            for i in range(sp.get_piece_size()):
                p = sp.id_to_piece(i)
                p_hf = p
                if p_hf not in vocab_dict:
                    additional_tokens.add(p_hf)
                if p_hf in set(constants.SPECIAL_TOKENS.values()) | {self.unk_token}:
                    logp = float(vocab_dict[p_hf])
                else:
                    logp = float(sp.get_score(i))
                token_logps[p_hf] = logp
            if additional_tokens:
                logger.warning(f"unknown tokens: {additional_tokens}")
            missing_tokens = set()
            for t in vocab_dict.keys():
                if t not in token_logps:
                    missing_tokens.add(t)
                    token_logps[t] = float(vocab_dict[t])
            if missing_tokens:
                logger.warning(f"Tokens {missing_tokens} not in sentencepiece model; defaulting to base "
                               f"tokenizer log-prob {vocab_dict[t]:.5f}")
            token_logps = self._log_normalize(token_logps)
            return token_logps
