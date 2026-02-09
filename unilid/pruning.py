import math
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional
from pathlib import Path

from tokenizers import Tokenizer

from unilid import constants
from unilid.tokenizer_builder import _build_unigramlm_hf_tokenizer_from_lprobs

import logging
logger = logging.getLogger(__name__)


def compute_intersection_pruning_tokens(
    per_lang_losses: Dict[str, List[Tuple[str, float]]],
    k_drop: int,
    lang_codes: List[str],
    under_consideration: Optional[Set[str]] = None
) -> Set[str]:
    """
    Find tokens to prune using rank-based intersection approach.
    """
    if k_drop <= 0:
        return set()

    if not lang_codes or len(lang_codes) < 2:
        logger.warning("Intersection pruning requires at least 2 languages")
        return set()

    if under_consideration is None:
        under_consideration = set()
        for lang_losses in per_lang_losses.values():
            under_consideration.update(token for token, _ in lang_losses)

    logger.info(f"Computing intersection pruning for {len(under_consideration)} tokens across {len(lang_codes)} languages")

    per_lang_ranks = {}

    for lc in lang_codes:
        lang_loss_tuples = per_lang_losses[lc].copy()
        lang_loss_tuples.sort(key=lambda x: x[1])

        token_to_rank = {}
        for rank, (tk, loss) in enumerate(lang_loss_tuples):
            if tk in under_consideration:
                token_to_rank[tk] = rank + 1

        per_lang_ranks[lc] = token_to_rank
        total_under_consideration_in_lang = len(token_to_rank)
        missing_tokens = len(under_consideration) - total_under_consideration_in_lang
        logger.info(f"Language {lc}: ranked {total_under_consideration_in_lang} tokens, {missing_tokens} missing (assigned rank 0)")

    token_max_ranks = []

    for tk in under_consideration:
        token_ranks = []
        for lc in lang_codes:
            if tk in per_lang_ranks[lc]:
                token_ranks.append(per_lang_ranks[lc][tk])
            else:
                token_ranks.append(0)

        max_rank = max(token_ranks)
        token_max_ranks.append((tk, max_rank))

    token_max_ranks.sort(key=lambda x: x[1])

    if len(token_max_ranks) < k_drop:
        logger.warning(f"Only {len(token_max_ranks)} tokens available, but {k_drop} requested")
        selected_tokens = set([tk for tk, _ in token_max_ranks])
    else:
        selected_tokens = set([tk for tk, _ in token_max_ranks[:k_drop]])

    if selected_tokens:
        selected_ranks = [max_rank for tk, max_rank in token_max_ranks[:len(selected_tokens)]]
        logger.info(f"Intersection pruning: selected {len(selected_tokens)} tokens with max-ranks from {min(selected_ranks)} to {max(selected_ranks)}")

    return selected_tokens


def proportional_sample(train_files_by_lang: Dict[str, str],
                        out_root: str,
                        total_lines: int) -> Dict[str, str]:
    """
    For each language copy <= n lines into <out_root>/<lang>/sample.txt,
    where n is proportional to the byte count recorded in meta.json.
    """
    byte_tot = {lg: json.load(open(Path(path).parent/"meta.json"))["bytes_written"]
                for lg, path in train_files_by_lang.items()}
    total = sum(byte_tot.values())

    out_path = Path(out_root)
    out_path.mkdir(parents=True, exist_ok=True)
    sampled_files = {}
    for lg, path in train_files_by_lang.items():
        k = max(constants.MIN_TOKENIZER_LANG_TRAINING_LINES, int(total_lines * byte_tot[lg] / total))
        dest_dir = out_path / lg
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "sample.txt"
        sampled_files[lg] = str(dest)

        if dest.exists():
            continue

        with open(path, encoding="utf-8", errors="ignore") as src, \
             open(dest, "w", encoding="utf-8") as tgt:
            lines_written = 0
            for line in src:
                if lines_written == k:
                    break
                if constants._RNG.random() > 0.2:
                    continue
                tgt.write(line)
                lines_written += 1
    return sampled_files


def _sample_lines_from_file(file_path: str, sample_proportion: float, seed: int) -> List[str]:
    """Efficiently sample lines from a file using reservoir sampling."""
    import random
    random.seed(seed)

    sample_size = None
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        first_lines = []
        i = 0
        for i, line in enumerate(f):
            if i >= 1000:
                break
            if line.strip():
                first_lines.append(line.strip())

        if i < 1000:
            return first_lines

        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        avg_line_size = sum(len(line) for line in first_lines) / len(first_lines)
        estimated_lines = int(file_size / avg_line_size)
        sample_size = max(1000, int(estimated_lines * sample_proportion))

    reservoir = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line

    return reservoir


def tokmix_merge(
    per_lang_tokenizers: Dict[str, Dict[str, float]],
    k_final: int,
    unk_token: str,
    special: List[str],
    pretokenizer: Any,
    decoder: Any = None,
    pruning_criterion: str = "probability",
    use_intersection_pruning: bool = False,
    byte_level: bool = False
) -> Tokenizer:
    """
    Unified TokMix merge function supporting all combinations of pruning criteria and strategies.
    """
    from unilid.algorithms.loss import _compute_alternatives_direct, _compute_loss_with_alternative_sequences

    lang_codes = list(per_lang_tokenizers.keys())
    n_langs = len(lang_codes)
    pinned = set(special)

    all_tokens = set()
    for token_probs in per_lang_tokenizers.values():
        all_tokens.update(token_probs.keys())
    candidate_tokens = all_tokens - pinned

    logger.info(f"TokMix merge: {len(all_tokens)} total tokens, {len(candidate_tokens)} candidates, "
               f"criterion={pruning_criterion}, intersection={use_intersection_pruning}")

    # Step 1: Compute per-language values based on pruning criterion
    per_lang_values = {}

    if pruning_criterion == "probability":
        for lang, token_probs in per_lang_tokenizers.items():
            per_lang_values[lang] = [(token, prob) for token, prob in token_probs.items()
                                   if token in candidate_tokens]

    elif pruning_criterion in ["approx_loss", "exact_loss"]:
        if pruning_criterion == "exact_loss":
            logger.info("Pre-computing alternative sequences...")
            per_lang_alternatives = {}
            for lang, token_probs in per_lang_tokenizers.items():
                alternatives = _compute_alternatives_direct(token_probs, unk_token, byte_level=byte_level)
                per_lang_alternatives[lang] = alternatives

        logger.info(f"Computing {pruning_criterion} values...")
        for lang, token_probs in per_lang_tokenizers.items():
            lang_values = []
            for token in candidate_tokens:
                if token not in token_probs:
                    continue

                if pruning_criterion == "approx_loss":
                    loss = token_probs[token] * math.log(max(token_probs[token], 1e-30))
                else:  # exact_loss
                    loss = _compute_loss_with_alternative_sequences(
                        token, token_probs, per_lang_alternatives[lang],
                        token_probs, unk_token
                    )
                lang_values.append((token, loss))
            per_lang_values[lang] = lang_values

    # Step 2: Token selection based on strategy
    if use_intersection_pruning:
        tokens_to_keep_count = k_final - len(pinned)
        tokens_to_drop = len(candidate_tokens) - tokens_to_keep_count

        if tokens_to_drop > 0:
            selected_to_remove = compute_intersection_pruning_tokens(
                per_lang_values, tokens_to_drop, lang_codes, candidate_tokens
            )
            kept_candidates = candidate_tokens - selected_to_remove
        else:
            kept_candidates = candidate_tokens

        kept_tokens = list(special) + list(kept_candidates)

    else:
        if pruning_criterion == "probability":
            avg_values = defaultdict(float)
            for lang_values in per_lang_values.values():
                for token, prob in lang_values:
                    avg_values[token] += prob
            for token in avg_values:
                avg_values[token] /= n_langs

            sorted_tokens = sorted(
                avg_values.items(),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            avg_values = defaultdict(float)
            for lang_values in per_lang_values.values():
                for token, loss in lang_values:
                    avg_values[token] += loss
            for token in avg_values:
                avg_values[token] /= n_langs

            sorted_tokens = sorted(
                avg_values.items(),
                key=lambda x: x[1],
                reverse=True
            )

        num_regular_tokens = max(0, k_final - len(special))
        kept_regular = [token for token, _ in sorted_tokens[:num_regular_tokens]]
        kept_tokens = list(special) + kept_regular

    logger.info(f"Selected {len(kept_tokens)} tokens for final vocabulary")

    # Step 3: Build final tokenizer with probability averaging
    avg_prob = defaultdict(float)

    for token in kept_tokens:
        for lang_probs in per_lang_tokenizers.values():
            avg_prob[token] += lang_probs.get(token, 0.0)
        avg_prob[token] /= n_langs

    Z = sum(avg_prob[t] for t in kept_tokens if t not in pinned)
    lprobs_tuples = []
    for token in kept_tokens:
        if token in pinned:
            p = max(avg_prob[token], constants.MIN_TOKEN_PROB)
        else:
            p = avg_prob[token] / Z if Z > 0 else constants.MIN_TOKEN_PROB
        lprobs_tuples.append((token, math.log(max(p, constants.MIN_TOKEN_PROB))))

    unk_id = kept_tokens.index(unk_token)
    return _build_unigramlm_hf_tokenizer_from_lprobs(lprobs_tuples, unk_id, pretokenizer=pretokenizer, decoder=decoder)
