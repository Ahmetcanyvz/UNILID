from __future__ import annotations

from pathlib import Path
import math
import os
import json
import logging
import random
import time
from typing import Dict, List, Tuple, Set, Optional, Union

import numpy as np

from tokenizers import Tokenizer, pre_tokenizers, decoders
from tokenizers.models import Unigram, BPE
from tokenizers.trainers import UnigramTrainer, BpeTrainer

from unilid import constants
from unilid.encoding import get_baseline_characters, get_baseline_bytes
from unilid.pruning import compute_intersection_pruning_tokens, tokmix_merge
from unilid.tokenizer_builder import (
    load_tokenizer_with_mode,
    _build_unigramlm_hf_tokenizer_from_lprobs,
)
from unilid.vocab_io import _load_custom_vocab_from_file
from unilid.trainers.em_loop import EMLoopMixin, EMState
from unilid.trainers.pruning_strategy import PruningStrategyMixin

logger = logging.getLogger(__name__)


class StandardUnigramLMTokenizer(EMLoopMixin, PruningStrategyMixin):
    """Unified implementation supporting HF, EM and Multigram."""

    def __init__(
        self,
        vocab_size: int = 8_000,
        unk_token: str = constants.SPECIAL_TOKENS["unk_token"],
        special_tokens: Optional[List[str]] = None,
        em_mode: Optional[str] = None,
        num_iterations: int = 10,
        multigram: bool = False,
        mixture_weights: Optional[Dict[str, float]] = None,
        byte_level: bool = True,
        whitespace_token_boundaries: bool = False,
        min_char_occurence_threshold: Optional[int] = None,
        dataset_size_normalization: bool = True,
        use_intersection_pruning: bool = True,
        initial_vocab_tokens: Optional[str] = None,
        pruning_criterion: str = "exact_loss",
        use_sp_seed_vocab: bool = True,
        use_sp_em: bool = True,
    ) -> None:
        """
        Initialize StandardUnigramLMTokenizer.

        All training paths are first-class configurable options:
        - em_mode: "hf", "bpe", "hard", "soft" — all valid, selected by this param
        - multigram: True/False — selected by this param
        - use_sp_em: True/False — selected by this param
        - use_sp_seed_vocab: True/False — selected by this param
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.special_tokens = special_tokens or list(constants.SPECIAL_TOKENS.values())
        self.byte_level = byte_level
        self.whitespace_token_boundaries = whitespace_token_boundaries
        self.min_char_occurence_threshold = min_char_occurence_threshold
        if not self.min_char_occurence_threshold:
            self.min_char_occurence_threshold = constants.MIN_CHAR_OCCURENCE_THRESHOLD

        self.base_tokenizer: Optional[Tokenizer] = None
        self.em_mode = em_mode
        self.num_iterations = num_iterations
        self.multigram = multigram
        self.mixture_weights = mixture_weights or {}
        self.dataset_size_normalization = dataset_size_normalization
        self.use_intersection_pruning = use_intersection_pruning
        self.initial_vocab_tokens = initial_vocab_tokens
        self.pruning_criterion = pruning_criterion
        self.use_sp_seed_vocab = use_sp_seed_vocab
        self.use_sp_em = use_sp_em

        # Validate parameters
        if self.initial_vocab_tokens and not os.path.exists(self.initial_vocab_tokens):
            raise FileNotFoundError(f"Initial vocabulary file not found: {self.initial_vocab_tokens}")

        if pruning_criterion not in ["exact_loss", "approx_loss", "probability"]:
            raise ValueError(f"Invalid pruning_criterion: {pruning_criterion}. Must be one of: exact_loss, approx_loss, probability")

        # components extracted from --initial-vocab source tokenizer
        self._source_pretokenizer = None
        self._source_normalizer = None
        self._source_decoder = None

        # multigram-specific
        self.lang_codes: List[str] = []
        self.global_lprobs: Dict[str, float] = {}

        # metadata collection
        self._training_metadata = {
            "start_time": None,
            "training_metrics": {}
        }

    # ------------------------- convenience helpers -----------------------
    def __getattr__(self, name):
        if self.base_tokenizer and hasattr(self.base_tokenizer, name):
            return getattr(self.base_tokenizer, name)
        raise AttributeError(name)

    def __call__(self, texts, *, add_special_tokens: bool = False, **kwargs):
        if self.base_tokenizer is None:
            raise ValueError("Tokenizer not yet loaded or trained.")
        if isinstance(texts, list):
            return self.encode_batch(texts, add_special_tokens=add_special_tokens)
        return self.base_tokenizer.encode(texts, add_special_tokens=add_special_tokens)

    _padding_side_default = "left"

    @property
    def padding_side(self):
        return getattr(self, "_padding_side", self._padding_side_default)

    @padding_side.setter
    def padding_side(self, value: str):
        if value not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        self._padding_side = value

    def save_pretrained(self, output_dir: str, **kwargs):
        if self.base_tokenizer is None:
            raise ValueError("Tokenizer not initialised / trained.")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        tok_json_path = out / "tokenizer.json"
        tok_json_path.write_text(self.base_tokenizer.to_str(), encoding="utf-8")

        special_map = {
            "unk_token": self.unk_token,
            "pad_token": getattr(self, "pad_token", None),
            "bos_token": getattr(self, "bos_token", None),
            "eos_token": getattr(self, "eos_token", None),
        }
        special_map = {k: v for k, v in special_map.items() if v is not None}
        (out / "special_tokens_map.json").write_text(
            json.dumps(special_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(out)

    @classmethod
    def from_pretrained(cls, pretrained_dir: str, **kwargs):
        tok_file = Path(pretrained_dir) / "tokenizer.json"
        if not tok_file.is_file():
            raise FileNotFoundError(tok_file)
        obj = cls(**kwargs)
        obj.base_tokenizer = Tokenizer.from_file(str(tok_file))
        sp_map = Path(pretrained_dir) / "special_tokens_map.json"
        if sp_map.is_file():
            specials = json.loads(sp_map.read_text(encoding="utf-8"))
            for attr, val in specials.items():
                setattr(obj, attr, val)
        return obj

    def pad(
        self,
        encoded_inputs,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
        **kwargs,
    ):
        # lazy: torch is in the [train] extra and is used nowhere else in this
        # module, so a prediction-only or [dev]-only install must not need it to
        # import the trainer.
        import torch

        if not isinstance(encoded_inputs, (list, tuple)):
            encoded_inputs = [encoded_inputs]

        pad_id = getattr(self.base_tokenizer, "pad_token_id", self.get_vocab()[constants.SPECIAL_TOKENS["pad_token"]])
        longest = max(len(f["input_ids"]) for f in encoded_inputs)
        if max_length is not None:
            longest = max_length
        if pad_to_multiple_of:
            longest = (longest + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

        batch = {}
        for field in encoded_inputs[0].keys():
            values = [f[field] for f in encoded_inputs]
            if all(v is None for v in values):
                continue
            if len(values) > 0 and isinstance(values[0], torch.Tensor):
                batch[field] = torch.stack(values)
                continue

            if len(values) > 0 and isinstance(values[0], (list, np.ndarray)):
                if field == "input_ids":
                    padded = [v[:longest] + [pad_id] * (longest - len(v[:longest])) for v in values]
                elif field == "attention_mask":
                    padded = [v[:longest] + [0] * (longest - len(v[:longest])) for v in values]
                elif field == "token_type_ids":
                    padded = []
                    for v in values:
                        v_truncated = v[:longest]
                        last_value = v_truncated[-1] if v_truncated else 0
                        padded.append(v_truncated + [last_value] * (longest - len(v_truncated)))
                elif field == "labels":
                    padded = [v[:longest] + [-100] * (longest - len(v[:longest])) for v in values]
                else:
                    logger.info(f"Unknown field type {field}; padding with 0s")
                    padded = [v[:longest] + [0] * (longest - len(v[:longest])) for v in values]

                batch[field] = torch.tensor(padded, dtype=torch.long) if return_tensors == "pt" else padded

            else:
                if len(values) > 0:
                    first_val = values[0]
                    if isinstance(first_val, (int, float)):
                        try:
                            batch[field] = torch.tensor(values) if return_tensors == "pt" else values
                        except:
                            batch[field] = values
                    else:
                        batch[field] = values

        if "attention_mask" not in batch:
            batch["attention_mask"] = (
                torch.ones_like(batch["input_ids"]) if return_tensors == "pt"
                else [[1]*len(f["input_ids"]) + [0]*(longest-len(f["input_ids"]))
                      for f in encoded_inputs]
            )

        return batch

    def load(self, base_path):
        self.base_tokenizer, self.byte_level = load_tokenizer_with_mode(base_path)
        self.vocab_size = len(self.base_tokenizer.get_vocab())
        with open(base_path, encoding="utf-8") as fh:
            data = json.load(fh)

        if data.get("model", {}).get("type") == "Unigram":
            vocab_pairs = data["model"]["vocab"]
            self.global_lprobs = self._log_normalize({tk: lp for tk, lp in vocab_pairs})
        else:
            logger.warning(f"No global probs for tokenizer at {base_path} found")

    def encode(self, text, add_special_tokens=False):
        if not self.base_tokenizer:
            raise ValueError("No base tokenizer loaded or trained.")
        return self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_batch(self, texts, *, add_special_tokens: bool = False):
        if self.base_tokenizer is None:
            raise ValueError("Tokenizer not yet loaded or trained.")
        return self.base_tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )

    def decode(self, ids):
        if not self.base_tokenizer:
            raise ValueError("No base_tokenizer loaded or trained.")
        return self.base_tokenizer.decode(ids)

    def get_vocab(self):
        if not self.base_tokenizer:
            raise ValueError("No base_tokenizer.")
        return self.base_tokenizer.get_vocab()

    def get_vocab_size(self):
        if not self.base_tokenizer:
            return 0
        return len(self.base_tokenizer.get_vocab())

    def _get_tokenizer_type(self):
        if self.em_mode == "bpe":
            return "bpe"
        elif self.multigram:
            return "multigram"
        else:
            return "standard"

    def _get_pretokenizer(self):
        if self.base_tokenizer:
            logger.info("Using pretokenizer from base tokenizer")
            return self.base_tokenizer.pre_tokenizer
        if self._source_pretokenizer:
            logger.info("Using pretokenizer from source tokenizer (--initial-vocab)")
            return self._source_pretokenizer
        if self.byte_level:
            logger.info("Using byte-level pretokenizer")
            return pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=self.whitespace_token_boundaries)
        elif self.whitespace_token_boundaries:
            logger.info("Using whitespace pretokenizer")
            return pre_tokenizers.Whitespace()
        else:
            logger.info("No pretokenizer found")
            return None

    def _get_normalizer(self):
        if self.base_tokenizer:
            return self.base_tokenizer.normalizer
        if self._source_normalizer:
            return self._source_normalizer
        return None

    def _get_decoder(self):
        if self.base_tokenizer:
            return self.base_tokenizer.decoder
        if self._source_decoder:
            return self._source_decoder
        if self.byte_level:
            return decoders.ByteLevel()
        elif self.whitespace_token_boundaries:
            raise NotImplementedError
        else:
            return None

    # ------------------------------------------------------------------
    # thin wrappers that call _run_em then build the HF tokenizer
    # ------------------------------------------------------------------
    def _train_custom_em(self, corpus_files: List[str], output_path=None):
        logging_path = output_path.split('.')[0] if output_path else None
        self.global_lprobs = self._run_em(corpus_files, per_lang=False, logging_path=logging_path)
        return self._build_tokenizer(self.global_lprobs, output_path)

    def _train_multigram(self, corpus_info: Dict[str, str], output_path=None):
        logging_path = output_path.split('.')[0] if output_path else None
        self.global_lprobs = self._run_em(None, per_lang=True, corpus_info=corpus_info, logging_path=logging_path)
        return self._build_tokenizer(self.global_lprobs, output_path)

    def _train_hf(self, corpus_files, output_path=None):
        logger.info("Using HuggingFace UnigramTrainer for standard Unigram.")
        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = self._get_pretokenizer()
        tok.decoder = self._get_decoder()

        trainer = UnigramTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            unk_token=self.unk_token
        )
        tok.train(corpus_files, trainer)

        self.base_tokenizer = tok

        if output_path:
            self.base_tokenizer.save(output_path)
            self._add_metadata_to_saved_tokenizer(output_path)
            logger.info(f"Saved HF-based standard tokenizer with metadata to {output_path}")
        return self.base_tokenizer

    def _train_bpe(self, corpus_files: List[str], output_path=None):
        logger.info("Using HuggingFace BPETrainer.")
        tok = Tokenizer(BPE(unk_token=self.unk_token))
        tok.pre_tokenizer = self._get_pretokenizer()
        tok.decoder = self._get_decoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if self.byte_level else [],
        )
        tok.train(corpus_files, trainer)
        self.base_tokenizer = tok
        if output_path:
            tok.save(output_path)
            self._add_metadata_to_saved_tokenizer(output_path)
            logger.info("Saved BPE tokenizer with metadata to %s", output_path)
        return tok

    def train(self, corpus_files, output_path=None, load_if_exists=False, corpus_info=None):
        """
        Train a Unigram tokenizer using the specified approach.
        Respects user-chosen em_mode and multigram parameters (no hardcoded overrides).
        """
        if load_if_exists and output_path and os.path.isfile(output_path):
            logger.info(f"Loading existing tokenizer from {output_path}")
            self.base_tokenizer = Tokenizer.from_file(output_path)
            return self.base_tokenizer

        self._training_metadata.update(
            self._collect_training_metadata(corpus_files, corpus_info)
        )
        self._training_metadata["start_time"] = time.time()

        self._training_metadata["training_metrics"] = {
            "initial_vocab_size": None,
            "pruned_per_iteration": [],
            "convergence": {"l1_deltas": [], "unk_probs": []},
            "training_time": None,
            "final_loss": None,
            "pruning_criterion": self.pruning_criterion
        }

        token_level = "byte" if self.byte_level else "character"
        logger.info(f"Training new tokenizer with EM mode: {self.em_mode} at {token_level}-level")

        # Dispatch based on user-configured parameters (no overrides)
        if self.em_mode == "bpe":
            result = self._train_bpe(corpus_files, output_path)

        elif self.em_mode == "hf" and not self.multigram:
            result = self._train_hf(corpus_files, output_path)

        elif self.multigram:
            if not corpus_info:
                raise ValueError("corpus_info is required for multigram training")
            result = self._train_multigram(corpus_info, output_path)

        elif self.em_mode in ["hard", "soft"]:
            result = self._train_custom_em(corpus_files, output_path)
        else:
            raise ValueError("Invalid settings for StandardUnigramLMTokenizer: no EM mode specified")

        if self._training_metadata["start_time"]:
            self._training_metadata["training_metrics"]["training_time"] = time.time() - self._training_metadata["start_time"]

        self._update_training_results(self._training_metadata["training_metrics"])

        return result

    @staticmethod
    def baseline_tokens(files: List[str],
                    byte_level: bool,
                    min_char_occurence_threshold: int = constants.MIN_CHAR_OCCURENCE_THRESHOLD
                    ) -> Set[str]:
        return (
            get_baseline_bytes()[0]
            if byte_level
            else set(get_baseline_characters(files, min_occurence_threshold=min_char_occurence_threshold)[0])
        )

    def _initial_vocab(self, train_files: List[str]) -> Set[str]:
        """Bootstrap an initial token set."""
        if self.initial_vocab_tokens:
            logger.info(f"Loading custom initial vocabulary from {self.initial_vocab_tokens}")
            custom_tokens = _load_custom_vocab_from_file(self.initial_vocab_tokens)

            # If source is an HF tokenizer, also extract pretokenizer/normalizer/decoder
            from unilid.vocab_io import _detect_vocab_file_format
            if _detect_vocab_file_format(self.initial_vocab_tokens) == 'hf_tokenizer':
                source_tok = Tokenizer.from_file(self.initial_vocab_tokens)
                if source_tok.pre_tokenizer:
                    self._source_pretokenizer = source_tok.pre_tokenizer
                    logger.info(f"Extracted pretokenizer from source: {source_tok.pre_tokenizer}")
                if source_tok.normalizer:
                    self._source_normalizer = source_tok.normalizer
                    logger.info(f"Extracted normalizer from source: {source_tok.normalizer}")
                if source_tok.decoder:
                    self._source_decoder = source_tok.decoder
                    logger.info(f"Extracted decoder from source: {source_tok.decoder}")

            vocab = set(custom_tokens)
            vocab.update(self.special_tokens)
            logger.info(f"Custom vocabulary prepared: {len(vocab)} tokens")

        elif self.use_sp_seed_vocab:
            logger.info("Using SentencePiece suffix-array seeding for initial vocabulary...")
            vocab = self._sp_seed_vocab(train_files)
            vocab.update(self.special_tokens)
            logger.info(f"SP-seeded initial vocabulary size: {len(vocab)}")
        else:
            vocab = self._initial_vocab_hf(train_files)
            logger.info(f"HF-seeded initial vocabulary size: {len(vocab)}")

        ws_count, ws_in_word_count = 0, 0
        for tk in vocab:
            if constants._HF_SPACE in tk:
                ws_count += 1
                if constants._HF_SPACE in tk[1:-1]:
                    ws_in_word_count += 1
        logger.info(f"{ws_count} tokens with whitespace characters ({ws_in_word_count} where whitespace is within token)")
        logger.info(f"Seed vocab sample: {random.sample(list(vocab), min(50, len(vocab)))}")

        return vocab

    def _initial_vocab_hf(self, train_files: List[str]) -> Set[str]:
        """Original HF trainer to bootstrap an initial token set."""
        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = self._get_pretokenizer()
        logger.info("--- Pre-tokenizer Sample Output ---: " + str(tok.pre_tokenizer.pre_tokenize_str("This is a test")))
        seed_vocab_size = self.vocab_size * constants.INITIAL_VOCAB_MULT_FACTOR
        logger.info(f"Attempting to find seed vocab of size {seed_vocab_size} with HuggingFace library...")
        trainer = UnigramTrainer(
            vocab_size=seed_vocab_size,
            special_tokens=self.special_tokens,
            unk_token=self.unk_token,
            max_piece_length=constants.MAX_BYTE_TOKEN_LEN if self.byte_level else constants.MAX_CHAR_TOKEN_LEN,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if self.byte_level else [],
            n_sub_iterations=0
        )

        def file_iterator(files: List[str]):
            for file_path in files:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        yield line

        corpus_iterator = file_iterator(train_files)
        tok.train_from_iterator(corpus_iterator, trainer)
        vocab = tok.get_vocab()

        return set(vocab.keys())

    def _build_final_vocab(
        self, lprobs: Dict[str, float]
    ) -> Tuple[List[Tuple[str, float]], int]:
        normalized_lprobs = self._log_normalize(lprobs, use_floor=True)
        sorted_items = sorted(normalized_lprobs.items(), key=lambda x: x[1], reverse=True)
        vocab_list = []
        special = []
        for tk, lp in sorted_items:
            if tk in self.special_tokens:
                special.append((tk, lp))
            else:
                vocab_list.append((tk, lp))
        vocab_list = special + vocab_list
        unk_id = next(i for i, (tk, _) in enumerate(vocab_list) if tk == self.unk_token)
        return vocab_list, unk_id

    def _build_tokenizer(self, lprobs: Dict[str, float], output_path):
        vocab_list, unk_id = self._build_final_vocab(lprobs)
        tok = _build_unigramlm_hf_tokenizer_from_lprobs(
            vocab_list,
            unk_id,
            pretokenizer=self._get_pretokenizer(),
            decoder=self._get_decoder()
        )
        self.base_tokenizer = tok
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            tok.save(output_path)
            self._add_metadata_to_saved_tokenizer(output_path)
            logger.info("Tokenizer with metadata saved to %s", output_path)
        return tok

    # --------- utilities ---------
    @staticmethod
    def _sanitize_mixture_weights(lang_codes: List[str], user_weights: Dict[str, float]) -> Dict[str, float]:
        if not lang_codes:
            return {}
        if not user_weights:
            uniform = 1.0 / len(lang_codes)
            return {lc: uniform for lc in lang_codes}
        else:
            weights = {lc: user_weights.get(lc, 0.0) for lc in lang_codes}
            s = sum(weights.values())
            if s <= 0:
                raise ValueError("Language mixture weights cannot sum to <= 0")
            return {lc: w / s for lc, w in weights.items()}

    @staticmethod
    def _normalize(values: Dict[str, float]) -> Dict[str, float]:
        total = sum(values.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(values))
            return {t: uniform for t in values}
        return {t: values[t] / total for t in values}

    @staticmethod
    def _counts_to_log_probs(counts: Dict[str, float]) -> Dict[str, float]:
        total = sum(counts.values())
        log_total = math.log(total)
        return {k: math.log(v) - log_total if v > 0 else constants.MIN_TOKEN_LOG_PROB for k, v in counts.items()}

    @staticmethod
    def _log_normalize(logvals: Dict[str, float], use_floor=False) -> Dict[str, float]:
        if use_floor:
            logvals = {tk: max(lp, constants.MIN_TOKEN_LOG_PROB) for tk, lp in logvals.items()}
        m = max(logvals.values())
        if not np.isfinite(m):
            logger.warning("Token log-probabilities sum to non-finite value; defaulting to uniform distribution")
            uniform = -np.log(len(logvals))
            token_logps = {k: uniform for k in logvals.keys()}
        else:
            logZ = m + np.log(sum(np.exp(v - m) for v in logvals.values()))
            token_logps = {k: (v - logZ) for k, v in logvals.items()}

        return token_logps
