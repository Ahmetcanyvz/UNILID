import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from unilid import constants
from unilid.metadata import _save_tokenizer_metadata, _create_base_metadata

logger = logging.getLogger(__name__)


def load_tokenizer_from_config(config):
    tokenizer_class = config.get('class', 'standard')
    byte_level = config.get('byte_level', False)

    if tokenizer_class == 'standard':
        from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer
        tokenizer = StandardUnigramLMTokenizer(
            byte_level=byte_level,
            multigram=False
        )
        tokenizer.load(config['path'])

    elif tokenizer_class == 'langspec':
        from unilid.trainers.language_specific_trainer import LanguageSpecificUnigramLMTokenizer
        tokenizer = LanguageSpecificUnigramLMTokenizer(byte_level=byte_level)
        tokenizer.load(config['base_path'], config.get('language_paths'))

    else:
        try:
            from transformers import AutoTokenizer  # lazy: optional dep, only for non-standard tokenizers
            tokenizer = AutoTokenizer.from_pretrained(config['path'])
        except Exception as e:
            logger.info(f"Tried to load tokenizer via default method and could not {e}")
            raise
    return tokenizer


def train_standard_tokenizer(corpus_info,
                             vocab_size=8000,
                             em_mode='soft',
                             multigram=False,
                             byte_level=False,
                             output_path=None,
                             dataset_size_normalization=True,
                             use_intersection_pruning=False,
                             pruning_criterion="exact_loss",
                             initial_vocab_tokens=None,
                             use_sp_seed_vocab=True,
                             use_sp_em=True
                             ):
    """Train a standard UnigramLM tokenizer on all languages."""
    from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer

    train_files = []
    for lang_code, train_file in corpus_info.items():
        train_files.append(train_file)

    if not train_files:
        logger.error("No training files found")
        return None

    if not output_path:
        output_path = f"standard_unigramlm_{em_mode}.json" if not multigram else f"multigramlm_{em_mode}.json"

    tokenizer = StandardUnigramLMTokenizer(
        vocab_size=vocab_size,
        em_mode=em_mode,
        multigram=multigram,
        byte_level=byte_level,
        dataset_size_normalization=dataset_size_normalization,
        use_intersection_pruning=use_intersection_pruning,
        pruning_criterion=pruning_criterion,
        initial_vocab_tokens=initial_vocab_tokens,
        use_sp_seed_vocab=use_sp_seed_vocab,
        use_sp_em=use_sp_em
    )
    tokenizer.train(train_files, output_path, corpus_info=corpus_info)

    return output_path


def train_lang_tokenizers(corpus_info: Dict[str, str],
                          out_dir: str,
                          vocab_size=8000,
                          byte_level=False,
                          use_per_lang_if_exists=True,
                          shared_seed_vocab=False
                          ) -> Dict[str, str]:
    """Train per-language tokenizers, optionally from a shared seed vocab."""
    from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer

    paths = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if shared_seed_vocab:
        seed_outfile = os.path.join(out_dir, "tokmix_seed_tokenizer.json")
        if not os.path.exists(seed_outfile):
            logger.info("Training seed tokenizer.")
            tok = StandardUnigramLMTokenizer(
                vocab_size   = vocab_size*constants.INITIAL_VOCAB_MULT_FACTOR,
                em_mode      = "soft",
                byte_level   = byte_level,
                multigram    = False
            )
            tok.train(corpus_info.values(), output_path=seed_outfile)
    else:
        seed_outfile = None

    for lg, sample_path in corpus_info.items():
        outfile = os.path.join(out_dir, f"tokmix_{lg}_tokenizer{'_shared_seed' if shared_seed_vocab else ''}.json")
        if not os.path.exists(outfile) or not use_per_lang_if_exists:

            logger.info("Training per-language tokenizer for %s ...", lg)
            tok = StandardUnigramLMTokenizer(
                vocab_size   = vocab_size,
                em_mode      = "soft",
                byte_level   = byte_level,
                multigram    = False,
                initial_vocab_tokens = seed_outfile
            )

            tok.train([sample_path], output_path=outfile)
        paths[lg] = outfile

    return paths


def train_tokmix(per_lang_paths: Dict[str, str],
                 k_final: int,
                 output_path: str,
                 byte_level: bool = False,
                 pruning_criterion: str = "probability",
                 use_intersection_pruning: bool = False,
                 baseline_tokens: set = None):
    """Combine language-specific tokenizers trained earlier using TokMix approach."""
    from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer
    from unilid.pruning import tokmix_merge

    source_metadata = {}
    for lang, path in per_lang_paths.items():
        try:
            with open(path, 'r') as f:
                tok_data = json.load(f)
                if "tokenizer_metadata" in tok_data:
                    source_metadata[lang] = {
                        "path": path,
                        "em_mode": tok_data["tokenizer_metadata"]["training_config"].get("em_mode"),
                        "vocab_size": tok_data["tokenizer_metadata"]["training_config"].get("vocab_size")
                    }
                else:
                    source_metadata[lang] = {"path": path, "em_mode": "unknown", "vocab_size": "unknown"}
        except:
            source_metadata[lang] = {"path": path, "em_mode": "unknown", "vocab_size": "unknown"}

    tkzs = {lg: StandardUnigramLMTokenizer() for lg in per_lang_paths}
    pretokenizer, decoder = None, None
    for lg, path in per_lang_paths.items():
        tkzs[lg].load(path)
        pretokenizer = tkzs[lg]._get_pretokenizer()
        decoder = tkzs[lg]._get_decoder()

    strategy_desc = f"intersection-{pruning_criterion}" if use_intersection_pruning else f"average-{pruning_criterion}"
    logger.info(f"Using TokMix strategy: {strategy_desc}")

    tokens_to_include = baseline_tokens if baseline_tokens else set()
    tokens_to_include.update(constants.SPECIAL_TOKENS.values())
    merged = tokmix_merge(
        {lg: tkz.global_lprobs for lg, tkz in tkzs.items()},
        k_final,
        constants.SPECIAL_TOKENS["unk_token"],
        tokens_to_include,
        pretokenizer,
        decoder,
        pruning_criterion=pruning_criterion,
        use_intersection_pruning=use_intersection_pruning,
        byte_level=byte_level
    )

    merged.save(output_path)

    tokmix_metadata = _create_base_metadata("TokMixTokenizer", "tokmix")
    tokmix_metadata["training_config"] = {
        "vocab_size": k_final,
        "source_tokenizer_configs": source_metadata,
        "pruning_criterion": pruning_criterion,
        "use_intersection_pruning": use_intersection_pruning,
        "merging_strategy": strategy_desc,
        "byte_level": byte_level
    }

    metadata_path = _save_tokenizer_metadata(output_path, tokmix_metadata)

    if metadata_path:
        logger.info("Saved TokMix tokenizer to %s with metadata to %s", output_path, metadata_path)
    else:
        logger.info("Saved TokMix tokenizer to %s", output_path)
    return output_path


def train_language_specific_tokenizer(corpus_info, vocab_size=8000,
                                      reestimation_em_mode='soft',
                                      output_dir="",
                                      byte_level=True,
                                      base_tokenizer_path=None,
                                      tokenizer_path_format=None,
                                      load_base_if_exists=True,
                                      load_tokenizers_if_exists=False,
                                      use_sentencepiece=True,
                                      base_em_mode=None,
                                      use_sp_seed_vocab=True,
                                      use_sp_em=True):
    """Train a shared base tokenizer plus one token distribution per language.

    ``use_sentencepiece`` selects the per-language re-estimation method only.
    The base tokenizer is trained separately, under ``base_em_mode`` (default:
    ``reestimation_em_mode``); when that resolves to "soft" or "hard" the base
    step runs the custom EM loop, which uses the spm_train binary unless both
    ``use_sp_seed_vocab`` and ``use_sp_em`` are False.
    """
    from unilid.trainers.language_specific_trainer import (
        DEFAULT_BASE_NAME,
        LanguageSpecificUnigramLMTokenizer,
    )

    tok = LanguageSpecificUnigramLMTokenizer(vocab_size=vocab_size, reestimation_em_mode=reestimation_em_mode, byte_level=byte_level,
                                             base_em_mode=base_em_mode, use_sp_seed_vocab=use_sp_seed_vocab, use_sp_em=use_sp_em)
    if not base_tokenizer_path:
        base_tokenizer_path = os.path.join(output_dir, DEFAULT_BASE_NAME)

    if not tokenizer_path_format:
        from unilid.trainers.language_specific_trainer import DEFAULT_LANGTOKENIZER_NAME
        # The trainer formats this with lang_code only, so em_mode is filled in
        # here (the raw template would raise KeyError('em_mode') downstream).
        tokenizer_path_format = os.path.join(
            output_dir,
            DEFAULT_LANGTOKENIZER_NAME.format(em_mode=reestimation_em_mode,
                                              lang_code="{lang_code}"))

    tokenizer_paths = tok.train(
        corpus_info,
        base_tokenizer_path=base_tokenizer_path,
        tokenizer_path_format=tokenizer_path_format,
        load_base_if_exists=load_base_if_exists,
        load_tokenizers_if_exists=load_tokenizers_if_exists,
        use_sentencepiece=use_sentencepiece
    )
    if not tokenizer_paths:
        logger.error("Failed to train or load language-specific tokenizer.")
        return None

    return tokenizer_paths
