from __future__ import annotations

import os, time, itertools
import logging
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List, Tuple, Optional

from pathlib import Path
import ujson
from unilid.encoding import encode_text, mk_id
from unilid.api import load_tokenizer_from_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- helper: disable inner Rust tokenizer thread-pools ----------
_WORKER_TOKENIZER = None
_WORKER_USE_BEST  = False

def _init_worker(tokenizer, use_best):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    global _WORKER_TOKENIZER, _WORKER_USE_BEST
    _WORKER_TOKENIZER = tokenizer
    _WORKER_USE_BEST  = use_best

def _iter_batches(fin, batch_size: int):
    """Yield successive `batch_size`-sized lists of lines from an open file"""
    while True:
        batch = [ln.rstrip("\n") for ln in itertools.islice(fin, batch_size) if ln]
        if not batch:
            break
        yield batch

def _process_batch_lines(lines):
    enc = encode_text(
        _WORKER_TOKENIZER,
        lines,
        use_best_language=_WORKER_USE_BEST,
    )

    unk_tok = getattr(_WORKER_TOKENIZER, "unk_token")
    unk_id = _WORKER_TOKENIZER.get_vocab().get(unk_tok)
    unk_total = 0
    json_parts = []

    best = enc.get("best_lang")
    for i, (txt, ids) in enumerate(zip(lines, enc["input_ids"])):
        if not txt or txt.isspace():
            continue
        unk_total += ids.count(unk_id)
        rec = {"text": txt, "token_ids": ids, 'sample_id': mk_id(txt)}
        if best is not None:
            rec["best_lang"] = best[i]
        json_parts.append(ujson.dumps(rec))

    return "\n".join(json_parts) + "\n", unk_total

def _tokenize_file(
        lang: str, split: str,
        in_path: str, out_path: str,
        tokenizer, use_best: bool,
        batch_size: int = 4096, n_workers: int | None = None):

    n_workers = n_workers or min(20, cpu_count() - 1)
    line_cnt = unk_cnt = 0

    with open(in_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8") as fout, \
         Pool(n_workers,
              initializer=_init_worker,
              initargs=(tokenizer, use_best)) as pool:

        for jsonl, unks in tqdm(
                pool.imap_unordered(_process_batch_lines,
                                    _iter_batches(fin, batch_size),
                                    chunksize=1),
                desc=f"{lang}-{split}"):
            fout.write(jsonl)
            line_cnt += jsonl.count("\n")
            unk_cnt  += unks

    return line_cnt, unk_cnt


class CorpusTokenizer:
    """A class for tokenizing corpora with various tokenizers."""

    def __init__(self,
                 corpus_info: Dict[str, Dict[str, Any]],
                 tokenizers: Dict[str, Dict[str, Any]],
                 output_dir: str = "",
                 max_lines_per_file: Optional[int] = 10000,
                 skip_if_exists: bool = True):
        self.corpus_info = corpus_info
        self.tokenizer_configs = tokenizers
        self.output_dir = output_dir
        self.max_lines_per_file = max_lines_per_file
        self.skip_if_exists = skip_if_exists

        self.tokenizers = self._load_tokenizers()

        self.tokenized_files = {name: {} for name in self.tokenizers.keys()}

    def _load_tokenizers(self) -> Dict[str, Any]:
        loaded_tokenizers = {}

        for name, config in self.tokenizer_configs.items():
            try:
                tokenizer = load_tokenizer_from_config(config)
                loaded_tokenizers[name] = {
                    'tokenizer': tokenizer,
                    'config': config
                }
                logger.info(f"Loaded tokenizer '{name}'")

            except Exception as e:
                logger.error(f"Failed to load tokenizer '{name}' with config {config}: {str(e)}")
        return loaded_tokenizers

    def tokenize_all(self):

        total_jobs = 0
        for _tname in self.tokenizers:
            for _lang, _info in self.corpus_info.items():
                total_jobs += len(_info.get("splits", {}))

        job_idx = 0
        for tname, tdata in self.tokenizers.items():
            tok      = tdata["tokenizer"]
            cfg      = tdata["config"]
            use_best = cfg.get("use_best_lang", True)
            bs       = cfg.get("batch_size", 4096)
            workers  = cfg.get("workers")

            unk_tok = getattr(tok, "unk_token")
            unk_id = tok.get_vocab().get(unk_tok)
            logger.info(f"Unknown token: {unk_tok} with ID {unk_id}")

            base_out = Path(cfg["tokenized_data_path"])
            base_out.mkdir(parents=True, exist_ok=True)
            self.tokenized_files[tname] = {}

            for lang, info in self.corpus_info.items():
                if "splits" not in info:
                    continue
                self.tokenized_files[tname][lang] = {}
                lang_out = base_out / lang
                lang_out.mkdir(exist_ok=True)

                for split, in_path in info["splits"].items():
                    job_idx += 1
                    tag = f"[{job_idx}/{total_jobs}] [{tname}] {lang}-{split}"

                    if not os.path.isfile(in_path):
                        logger.warning(f"{tag}: input file {in_path} is missing – skipped")
                        continue

                    out_path = lang_out / f"{split}.jsonl"
                    if self.skip_if_exists and out_path.is_file():
                        logger.info(f"{tag}: already tokenised – skipped")
                        self.tokenized_files[tname][lang][split] = str(out_path)
                        continue

                    logger.info(f"{tag}: starting tokenisation "
                                f"(batch={bs}, workers={workers or 'auto'})")
                    t0 = time.perf_counter()

                    n_lines, n_unk = _tokenize_file(
                        lang       = lang,
                        split      = split,
                        in_path    = in_path,
                        out_path   = str(out_path),
                        tokenizer  = tok,
                        use_best   = use_best,
                        batch_size = bs,
                        n_workers  = workers,
                    )

                    dt = time.perf_counter() - t0
                    if n_lines:
                        self.tokenized_files[tname][lang][split] = str(out_path)

                    logger.info(f"{tag}: done in {dt:,.1f}s – "
                                f"{n_lines:,} lines, {n_unk:,} UNK → {out_path}")

        return self.tokenized_files


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8', errors="replace") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Tokenize corpora with multiple tokenizers')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    parser.add_argument('--output_dir', type=str, default="", help='Base output directory')
    parser.add_argument('--max_lines', type=int, default=10000, help='Maximum lines per file')
    parser.add_argument('--no_skip', action='store_true', help='Do not skip existing files')

    args = parser.parse_args()

    config = load_config(args.config)

    corpus_tokenizer = CorpusTokenizer(
        corpus_info=config['corpus_info'],
        tokenizers=config['tokenizers'],
        output_dir=args.output_dir,
        max_lines_per_file=args.max_lines,
        skip_if_exists=not args.no_skip
    )

    results = corpus_tokenizer.tokenize_all()

    logger.info("\n===== Tokenization Summary =====")
    for tokenizer_name, lang_data in results.items():
        logger.info(f"Tokenizer: {tokenizer_name}")
        for lang, split_data in lang_data.items():
            logger.info(f"  Language: {lang}")
            for split, path in split_data.items():
                logger.info(f"    {split}: {path}")


if __name__ == "__main__":
    main()
