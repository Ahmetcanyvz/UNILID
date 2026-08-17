import os
import json
import math
from typing import List, Tuple, Any

from unilid import constants
from unilid.token_encoding import _apply_pretok_mapping_hf

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Special tokens and probability mass
# ---------------------------------------------------------------------------
def special_token_set(extra_unk_token: str = None) -> set:
    """The tokens that carry no language evidence and must hold no mass."""
    tokens = set(constants.SPECIAL_TOKENS.values())
    if extra_unk_token:
        tokens.add(extra_unk_token)
    return tokens


def renormalize_over_real_tokens(token_logps: dict,
                                 special_tokens: set = None) -> dict:
    """Move every unit of probability mass onto tokens that can affect a score.

    No special token's stored weight is ever read when scoring. The Rust
    scorer takes its unknown-token score from a single global constant
    (``min_score - K_UNK_PENALTY``, model.rs), not from the per-language row,
    and ``<s>``/``</s>``/``<pad>`` are reachable only by text containing those
    literal substrings. Verified by perturbation: setting all four entries of
    every row to -500 changes predicted scores by 0.000000.

    Mass parked on them is therefore mass taken away from the tokens that do
    contribute, which lowers every real token by a constant. That constant
    differs per training method, which is what makes rows trained by different
    methods incomparable inside one model. So the mass is normalized over the
    real tokens only and the special tokens are parked at the floor.
    """
    special = special_tokens if special_tokens is not None else special_token_set()
    real = {tk: lp for tk, lp in token_logps.items() if tk not in special}
    if not real:
        raise ValueError("no non-special tokens to normalize over; the "
                         "per-language distribution would be empty")

    m = max(real.values())
    if not math.isfinite(m):
        raise ValueError(f"non-finite maximum log-probability {m} over the "
                         f"real tokens; refusing to write a degenerate row")
    log_z = m + math.log(sum(math.exp(v - m) for v in real.values()))
    out = {tk: lp - log_z for tk, lp in real.items()}
    for tk in token_logps:
        if tk in special:
            out[tk] = constants.MIN_TOKEN_LOG_PROB
    return out


# ---------------------------------------------------------------------------
# HF corpus writing
# ---------------------------------------------------------------------------
def write_hf_bytelevel_corpus(in_path: str, out_path: str, pretok: Any, normalizer: Any = None):
    """Convert raw UTF-8 text to HF-style text (e.g., with 'Ġ' markers)."""
    with open(in_path, "r", encoding="utf-8", errors="strict") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            line = line.rstrip("\n")
            pieces = _apply_pretok_mapping_hf(line, pretok, return_individual_pieces=True, normalizer=normalizer)
            fout.write("".join(pieces) + "\n")


def write_hf_bytelevel_corpus_optimized(in_path: str, out_path: str, pretok: Any, chunk_size_bytes: int = 1024 * 1024, normalizer: Any = None):
    """
    An optimized version of the corpus writing function that reads and writes
    in larger chunks to reduce I/O overhead.
    """
    logger.info(f"Processing {os.path.basename(in_path)} -> {os.path.basename(out_path)}")
    with open(in_path, "r", encoding="utf-8", errors="strict") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        while True:
            lines = fin.readlines(chunk_size_bytes)
            if not lines:
                break
            processed_lines_generator = (
                "".join(_apply_pretok_mapping_hf(line.rstrip("\n"), pretok, return_individual_pieces=True, normalizer=normalizer)) + "\n"
                for line in lines
            )
            fout.writelines(processed_lines_generator)
    logger.info(f"Finished processing {os.path.basename(in_path)}")


def _process_file_worker(args: Tuple[str, str, Any]) -> str:
    """
    Helper function for multiprocessing.Pool. It unpacks arguments to call
    the optimized processing function.
    """
    train_file, output_dir, pretok, i = args
    base_name = os.path.basename(train_file)
    output_file_path = os.path.join(output_dir, f"processed_{i}_{base_name}")
    write_hf_bytelevel_corpus_optimized(train_file, output_file_path, pretok)
    return output_file_path


# ---------------------------------------------------------------------------
# SentencePiece seed vocab writing
# ---------------------------------------------------------------------------
def _write_sp_seed_vocab_file(vocab_with_scores: list[tuple[str, float]], out_path: str, unk_token: str):
    """One token per line: token<TAB>log_score (uniform log prob)."""
    specials = set(constants.SPECIAL_TOKENS.values()) | {unk_token}
    count = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for t, score in vocab_with_scores:
            if t in specials:
                continue
            if ("\t" in t) or ("\n" in t) or ("\r" in t):
                raise ValueError(f"Token contains tab/newline: {repr(t)}")
            f.write(f"{t}\t{score}\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# HF tokenizer vocab loading
# ---------------------------------------------------------------------------
def _load_vocab_from_hf_tokenizer(file_path: str) -> List[str]:
    """Load vocabulary from HuggingFace tokenizer JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'model' not in data or 'vocab' not in data['model']:
        raise ValueError(f"File does not appear to be a HuggingFace tokenizer: {file_path}")

    vocab_data = data['model']['vocab']
    if isinstance(vocab_data, list):
        tokens = [item[0] for item in vocab_data if isinstance(item, list) and len(item) >= 1]
    elif isinstance(vocab_data, dict):
        tokens = list(vocab_data.keys())
    else:
        raise ValueError(f"Unsupported vocab format in HF tokenizer file: {file_path}")

    return tokens


# ---------------------------------------------------------------------------
# Generic vocab file loading (text, JSON, HF)
# ---------------------------------------------------------------------------
def _detect_vocab_file_format(file_path: str) -> str:
    """Automatically detect the format of a vocabulary file."""
    if file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'model' in data and 'vocab' in data.get('model', {}):
                    return 'hf_tokenizer'
                elif isinstance(data, list):
                    return 'json_list'
                else:
                    raise ValueError(f"Unsupported JSON format in {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {file_path}")
    else:
        return 'text_file'


def _load_vocab_from_text_file(file_path: str) -> List[str]:
    """Load vocabulary from plain text file (one token per line)."""
    tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            token = line.rstrip('\n\r')
            if token:
                tokens.append(token)
            elif line.strip() == '':
                logger.warning(f"Empty line {line_num} in vocab file {file_path}")
    return tokens


def _load_vocab_from_json_file(file_path: str) -> List[str]:
    """Load vocabulary from JSON list file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of strings: {file_path}")

    tokens = []
    for i, token in enumerate(data):
        if not isinstance(token, str):
            logger.warning(f"Non-string token at index {i} in {file_path}: {token}")
            continue
        if token:
            tokens.append(token)

    return tokens


def _load_custom_vocab_from_file(file_path: str) -> List[str]:
    """Main method to load vocabulary from file, with automatic format detection."""
    format_type = _detect_vocab_file_format(file_path)

    if format_type == 'text_file':
        tokens = _load_vocab_from_text_file(file_path)
    elif format_type == 'json_list':
        tokens = _load_vocab_from_json_file(file_path)
    elif format_type == 'hf_tokenizer':
        tokens = _load_vocab_from_hf_tokenizer(file_path)
    else:
        raise ValueError(f"Unsupported vocabulary file format: {format_type}")

    if not tokens:
        raise ValueError(f"No valid tokens found in vocabulary file: {file_path}")

    logger.info(f"Loaded {len(tokens)} tokens from {file_path} (format: {format_type})")
    return tokens
