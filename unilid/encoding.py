from __future__ import annotations

import multiprocessing
import hashlib
from collections import Counter
from typing import Dict, List, Tuple, Set, Any, Optional, Union

from tokenizers.pre_tokenizers import ByteLevel

from unilid import constants

import logging
logger = logging.getLogger(__name__)


def get_baseline_bytes() -> Tuple[Set[str], Dict[str, int]]:
    alphabet = ByteLevel.alphabet()
    return set(alphabet), {c: 1 for c in alphabet}


def get_baseline_characters(training_data_files, character_coverage=0.9995, min_occurence_threshold=constants.MIN_CHAR_OCCURENCE_THRESHOLD):
    """
    Get baseline characters from training data files, matching SentencePiece's approach.
    """
    num_processes = min(multiprocessing.cpu_count() or 1, len(training_data_files), 20)

    chars_by_file = {}
    total_char_counts = Counter()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_count_worker, training_data_files)

        for file_path, counts in results:
            chars_by_file[file_path] = counts
            total_char_counts.update(counts)

    total_chars = sum(total_char_counts.values())

    char_freqs = {char: count/total_chars for char, count in total_char_counts.items()}

    sorted_chars = sorted(total_char_counts.items(), key=lambda x: x[1], reverse=True)

    coverage_chars = set()
    cumulative_coverage = 0.0
    target_coverage = total_chars * character_coverage

    for char, count in sorted_chars:
        coverage_chars.add(char)
        cumulative_coverage += count
        if cumulative_coverage >= target_coverage:
            break

    frequency_chars = {char for char, count in total_char_counts.items()
                      if count >= min_occurence_threshold}

    chars_to_keep = list(coverage_chars.union(frequency_chars))

    covered_count = sum(total_char_counts[char] for char in coverage_chars)
    actual_coverage = covered_count / total_chars * 100.0

    freq_only_chars = frequency_chars - coverage_chars
    freq_only_count = len(freq_only_chars)

    logger.info(f"Character coverage: {len(coverage_chars)}/{len(total_char_counts)} "
                f"characters ({actual_coverage:.4f}%)")
    logger.info(f"Additional characters from frequency threshold: {freq_only_count}")
    logger.info(f"Total characters kept: {len(chars_to_keep)}")
    logger.info(f"Characters not included: {len(total_char_counts) - len(chars_to_keep)}")

    return chars_to_keep, char_freqs


def _count_worker(file_path):
    """Worker function to count character/byte frequencies in a file."""
    try:
        token_counts = Counter()
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                token_counts.update(line)
        return file_path, token_counts
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return file_path, Counter()


def _as_batch(txt):
    """Return (list version, was_single_flag)."""
    if isinstance(txt, (str, tuple)):
        return [txt], True
    if isinstance(txt, list):
        return txt, False
    raise TypeError(type(txt))


def _apply_trunc_pad(
    ids_batch, toks_batch, offs_batch,
    *,
    truncation: bool, max_length: int | None,
    padding: str | None, pad_id: int, pad_tok: str,
    return_attention_mask: bool,
):
    """
    Apply truncation and padding to batched encoding results.
    """
    out_ids, out_toks, out_offs = [], [], []
    masks = [] if return_attention_mask else None

    for ids, toks, offs in zip(ids_batch, toks_batch,
                               offs_batch or [None] * len(ids_batch)):
        if truncation and max_length:
            ids, toks = ids[:max_length], toks[:max_length]
            if offs is not None:
                offs = offs[:max_length]

        if padding == "max_length" and max_length:
            pad_len = max_length - len(ids)
            ids  += [pad_id] * pad_len
            toks += [pad_tok] * pad_len
            if offs is not None:
                offs += [(0, 0)] * pad_len
        else:
            pad_len = 0

        if masks is not None:
            real_len = len(ids) - pad_len
            masks.append([1] * real_len + [0] * pad_len)

        out_ids.append(ids)
        out_toks.append(toks)
        if offs is not None:
            out_offs.append(offs)

    return (
        out_ids,
        out_toks,
        masks,
        out_offs if offs_batch is not None else None,
    )


def mk_id(raw: str) -> str:
    """8-byte / 16-char hex digest, utf-8, replace errors."""
    digest = hashlib.blake2s(
        raw.encode("utf-8", "replace"),
        digest_size=8
    ).hexdigest()
    return digest


def encode_text(
    tokenizer: Any,
    text: Union[str, List[str], Tuple[str, str], List[Tuple[str, str]]],
    *,
    add_special_tokens: bool = False,
    padding: Optional[str] = None,
    truncation: Union[bool, str] = False,
    max_length: Optional[int] = None,
    stride: Optional[int] = None,
    return_offsets_mapping: bool = False,
    return_attention_mask: bool = False,
    return_special_tokens_mask: bool = False,
    return_token_type_ids: bool = False,
    return_tensors: Optional[str] = None,
    return_word_ids: bool = False,
    language: Optional[str] = None,
    use_best_language: bool = True,
) -> Dict[str, Any]:

    batch_txt, was_single = _as_batch(text)
    pad_token = tokenizer.pad_token if hasattr(tokenizer, "pad_token") else constants.SPECIAL_TOKENS["pad_token"]
    pad_id = tokenizer.get_vocab()[pad_token]

    # ---- Language-specific UnigramLM path ----
    from unilid.trainers.language_specific_trainer import LanguageSpecificUnigramLMTokenizer
    if isinstance(tokenizer, LanguageSpecificUnigramLMTokenizer):
        if use_best_language:
            best_langs, enc_objects = tokenizer.best_language_encode_batch(batch_txt)
            ids_batch  = [enc.ids    for enc in enc_objects]
            toks_batch = [enc.tokens for enc in enc_objects]
            offsets_b  = [enc.offsets for enc in enc_objects]
        else:
            if language is None:
                raise ValueError("`language` must be provided when use_best_language=False")
            enc_objects = tokenizer.language_info[language]["tokenizer"].encode_batch(batch_txt)
            best_langs  = None
            ids_batch   = [enc.ids    for enc in enc_objects]
            toks_batch  = [enc.tokens for enc in enc_objects]
            offsets_b   = [enc.offsets for enc in enc_objects]

        ids_batch, toks_batch, masks, offsets_b = _apply_trunc_pad(
            ids_batch, toks_batch, offsets_b,
            truncation=truncation, max_length=max_length,
            padding=padding, pad_id=pad_id,
            pad_tok=pad_token,
            return_attention_mask=return_attention_mask,
        )

        result = {"input_ids": ids_batch, "tokens": toks_batch}
        if best_langs is not None:
            result["best_lang"] = best_langs
        if return_attention_mask:
            result["attention_mask"] = masks
        if return_offsets_mapping:
            result["offset_mapping"] = offsets_b
        if was_single:
            for k, v in result.items():
                if isinstance(v, list) and len(v) == 1:
                    result[k] = v[0]
        return result

    # ---- Generic fast tokenizer path ----
    if hasattr(tokenizer, "encode_batch"):
        encodings = tokenizer.encode_batch(batch_txt, add_special_tokens=add_special_tokens)

        ids_batch   = [enc.ids    for enc in encodings]
        toks_batch  = [enc.tokens for enc in encodings]
        word_ids_b  = [enc.word_ids() for enc in encodings] if return_word_ids else None
        offsets_b  = [enc.offsets for enc in encodings] if return_offsets_mapping else None

        ids_batch, toks_batch, masks, offsets_b = _apply_trunc_pad(
            ids_batch, toks_batch, offsets_b,
            truncation=truncation, max_length=max_length,
            padding=padding, pad_id=pad_id,
            pad_tok=pad_token,
            return_attention_mask=return_attention_mask,
        )

        result = {"input_ids": ids_batch, "tokens": toks_batch}
        if masks is not None:
            result["attention_mask"] = masks
        if word_ids_b is not None:
            result["word_ids"] = word_ids_b
        if return_offsets_mapping:
            result["offset_mapping"] = offsets_b

    else:
        # ---- Fallback to high-level HF __call__ ----
        hf_kwargs = dict(
            add_special_tokens   = add_special_tokens,
            padding              = padding,
            truncation           = truncation,
            max_length           = max_length,
            stride               = stride,
            return_offsets_mapping    = return_offsets_mapping,
            return_attention_mask     = return_attention_mask,
            return_special_tokens_mask= return_special_tokens_mask,
            return_token_type_ids     = return_token_type_ids,
            return_tensors            = return_tensors,
        )
        enc = tokenizer(batch_txt, **hf_kwargs)
        result = enc.to_dict() if hasattr(enc, "to_dict") else dict(enc)
        if "tokens" not in result and hasattr(tokenizer, "convert_ids_to_tokens"):
            result["tokens"] = [tokenizer.convert_ids_to_tokens(ids) for ids in result["input_ids"]]

    if was_single:
        for k, v in result.items():
            if isinstance(v, list) and len(v) == 1:
                result[k] = v[0]
    return result
