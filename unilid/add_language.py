"""Add one language to an existing calibrated .unilid model.

Workflow (the paper's incremental-customization recipe):
1. Train the new language's distribution over the model's EXISTING shared
   vocabulary (per-language fixed-vocabulary re-estimation, 20 EM rounds by
   default).
2. Append the new weight row to the container at the last index; every existing
   row and the language order are preserved byte-identically.
3. Update the calibration with only the new language's data: its training-line
   count N; a re-examination threshold estimated by the recipe of record when
   N < head_n (no threshold otherwise); the unseen-token constant applies to the
   new row at load time like every other row. The new language becomes a
   replacement candidate if and only if N >= replacement_min_n.

Two documented approximations (stated, not silent): existing languages'
thresholds are kept although a new language changes the margin distribution they
were estimated on, and a new language with N >= replacement_min_n becomes a
replacement candidate for every gated language without recalibration of those
languages. High-entropy-group membership is NOT recomputed here (it needs
cross-language statistics and a validation scoring pass; see the paper's
development-protocol appendix).
"""
from __future__ import annotations

import argparse
import datetime
import gc
import shutil
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Optional

import numpy as np

from .calibration import (
    CAUSE_LOW_CALIBRATION,
    Calibration,
    TauRow,
    UnilidCalibrationError,
    estimate_tau,
)
from .constants import (
    MISSING_TOKEN_FILL_DETECTION_BOUND,
    MISSING_TOKEN_FILL_LOG_PROB,
)
from .vocab_io import special_token_set
from .model_io import (
    UnilidModel,
    _get_vocab_with_scores,
    load_unilid_raw,
    read_calibration,
    write_unilid,
)


def _count_training_lines(train_file: Path) -> int:
    """N for the new language: the number of non-empty lines in its training
    file (the same non-empty-line convention the threshold-estimation recipe
    uses when reading calibration lines)."""
    n = 0
    with open(train_file, encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n"):
                n += 1
    return n


def _train_new_language(base_tok_json: str, lang: str, train_file: Path,
                        workdir: Path, method: str, em_rounds: int) -> Path:
    """Train the new language's distribution over the existing vocabulary and
    return the path of the written per-language tokenizer JSON."""
    from .trainers.language_specific_trainer import (
        LanguageSpecificUnigramLMTokenizer,
    )

    if method not in ("sp", "em"):
        raise ValueError(f"method must be 'sp' or 'em', got {method!r}")
    if method == "sp" and not shutil.which("spm_train"):
        raise RuntimeError(
            "method='sp' (the release-verified training path) needs the "
            "spm_train executable from the bundled sentencepiece fork on PATH; "
            "build it (see README) or pass method='em' (pure Python, "
            "not verified against the release's end-to-end chain)")

    workdir.mkdir(parents=True, exist_ok=True)
    base_path = workdir / "langspec_base_tokenizer.json"
    base_path.write_text(base_tok_json, encoding="utf-8")

    trainer = LanguageSpecificUnigramLMTokenizer(num_iterations=em_rounds)
    result = trainer.train(
        corpus_info={lang: str(train_file)},
        base_tokenizer_path=str(base_path),
        output_dir=str(workdir),
        load_base_if_exists=True,
        load_tokenizers_if_exists=False,
        use_sentencepiece=(method == "sp"),
        num_reestimation_iterations=em_rounds,
    )
    if not result or lang not in result.get("language_paths", {}):
        raise RuntimeError(
            f"training produced no tokenizer for {lang!r} (trainer result: "
            f"{result!r})")
    matches = sorted(workdir.glob(f"langspec_*_{lang}.tokenizer.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one trained tokenizer file for {lang!r} in "
            f"{workdir}, found {[m.name for m in matches]}")
    return matches[0]


def _extract_row(tok_path: Path, ref_vocab: dict, vocab_size: int) -> np.ndarray:
    """Build the language's dense weight row exactly as save_unilid does."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))
    scores = {token: score for token, score in _get_vocab_with_scores(tok)}
    row = np.full(vocab_size, np.float32(MISSING_TOKEN_FILL_LOG_PROB),
                  dtype=np.float32)
    for token, rid in ref_vocab.items():
        lp = scores.get(token)
        if lp is not None:
            row[rid] = np.float32(lp)
    return row


# Above this ratio between the largest and smallest real-token mass, the model's
# rows are reported as spread out. Diagnostic only, never fatal: 2.0 is well
# above the variation one training method produces and well below the factor of
# five that separates a pre-fix sp row from a corrected one.
SCALE_SPREAD_REPORT_RATIO = 2.0


def _real_token_mass(rows: np.ndarray, real_idx: np.ndarray,
                     block: int = 64) -> np.ndarray:
    """Probability mass each row places on the tokens that affect a score.

    Blocked so that a released-scale matrix (1,940 x 100,000) never becomes a
    float64 temporary of its full size.
    """
    rows = np.asarray(rows)
    if rows.ndim == 1:
        rows = rows[None, :]
    out = np.empty(rows.shape[0], dtype=np.float64)
    for start in range(0, rows.shape[0], block):
        chunk = np.asarray(rows[start:start + block])[:, real_idx]
        out[start:start + block] = np.exp(chunk.astype(np.float64)).sum(axis=1)
    return out


def _match_real_token_scale(new_row: np.ndarray, weights, ref_vocab: dict,
                            lang: str) -> np.ndarray:
    """Put the new row on the same scale as the model's existing rows.

    Only the tokens that can affect a score matter, and only their mass
    relative to the other rows'. A row carrying more of it scores higher by a
    constant per token against every existing language, an advantage that has
    nothing to do with the language.

    Models written before special tokens were excluded from the distribution
    hold part of their mass on those tokens, so their real-token mass is below
    one; the released 1,940-language model sits at exactly 0.2. Rescaling the
    new row to the model's own figure keeps it comparable with what is already
    there, whichever training method produced either.
    """
    specials = special_token_set()
    real_idx = np.array(sorted(rid for token, rid in ref_vocab.items()
                               if token not in specials), dtype=np.int64)
    if real_idx.size == 0:
        raise RuntimeError("the base vocabulary has no non-special tokens")

    existing = _real_token_mass(np.asarray(weights), real_idx)
    lo, hi = float(existing.min()), float(existing.max())
    if lo <= 0.0 or not np.isfinite(hi):
        raise RuntimeError(
            f"the model's existing rows put no usable probability mass on real "
            f"tokens (min {lo:.6g}, max {hi:.6g}); refusing to add a language "
            f"to an artifact whose rows cannot be compared")
    # Some spread is normal: before special tokens were excluded, a row's real
    # mass was 1 - p(unk), which differs per language. A wide spread is worth
    # saying out loud, but it is not an error, and the median is still the best
    # available target.
    if hi / lo > SCALE_SPREAD_REPORT_RATIO:
        print(f"Note: the model's rows put between {lo:.6g} and {hi:.6g} of "
              f"their probability mass on real tokens across "
              f"{len(existing):,} languages. {lang!r} is matched to the "
              f"median; if those rows came from different training methods "
              f"they were already not comparable with each other")

    target = float(np.median(existing))
    new_mass = float(_real_token_mass(new_row, real_idx)[0])
    shift = np.float32(np.log(target) - np.log(new_mass))
    if abs(float(shift)) < 1e-6:
        print(f"Row scale: real-token mass {new_mass:.6g} already matches the "
              f"model's {target:.6g}; no rescaling needed")
        return new_row
    out = new_row.copy()
    out[real_idx] += shift   # specials stay at the floor; they hold no mass
    print(f"Row scale: real-token mass {new_mass:.6g} rescaled to the model's "
          f"{target:.6g} ({float(shift):+.4f} nats per token), so {lang!r} is "
          f"scored on the same footing as the existing languages")
    return out


def add_language(model_path, lang: str, train_file, output_path, *,
                 method: str = "sp", em_rounds: int = 20,
                 calibration: Optional[str] = None,
                 workdir: Optional[Path] = None) -> dict:
    """Add ``lang`` to the model at ``model_path`` and write the extended,
    recalibrated container to ``output_path``. Returns a summary dict.

    ``model_path`` must carry a calibration (a version-2 .unilid file, or a
    version-1 file plus ``calibration=`` pointing at its calibration JSON): the
    threshold recipe is defined on the calibrated model.
    """
    model_path = Path(model_path)
    train_file = Path(train_file)
    output_path = Path(output_path).with_suffix(".unilid")
    if not train_file.is_file():
        raise FileNotFoundError(f"training file not found: {train_file}")
    if output_path.resolve() == model_path.resolve():
        raise ValueError(
            "output_path must differ from model_path (the input container is "
            "never rewritten in place)")

    cal = read_calibration(model_path)
    if cal is not None and calibration is not None:
        raise UnilidCalibrationError(
            f"{model_path} already bundles a calibration; drop the "
            f"calibration= argument")
    if cal is None and calibration is not None:
        cal = Calibration.from_json_file(calibration)
    if cal is None:
        raise UnilidCalibrationError(
            f"{model_path} carries no calibration artifact, and add_language "
            f"needs one (thresholds are defined on the calibrated model). "
            f"Pass calibration=<path> or use a version-2 model file.")

    base_tok_bytes, weights, langs = load_unilid_raw(model_path)
    cal.runtime_for(langs)  # validate the INPUT artifact before any training
    if lang in langs:
        raise ValueError(f"language {lang!r} is already in the model")
    if lang in cal.train_counts:
        raise UnilidCalibrationError(
            f"language {lang!r} is absent from the model but present in the "
            f"calibration train_counts; the artifact is inconsistent")
    vocab_size = weights.shape[1]
    from tokenizers import Tokenizer
    base_tok = Tokenizer.from_str(base_tok_bytes.decode("utf-8"))
    ref_vocab = base_tok.get_vocab()
    base_tok_json = base_tok_bytes.decode("utf-8")

    if workdir is None:
        workdir = output_path.parent / (output_path.stem + "_workdir")
    workdir = Path(workdir)

    n_l = _count_training_lines(train_file)
    if n_l == 0:
        raise ValueError(f"training file {train_file} has no non-empty lines")

    print(f"Training {lang!r} over the existing vocabulary "
          f"({n_l:,} lines, method={method}, {em_rounds} EM rounds)...")
    tok_path = _train_new_language(base_tok_json, lang, train_file, workdir,
                                   method, em_rounds)
    new_row = _extract_row(tok_path, ref_vocab, vocab_size)

    # Post-training guards (loud): full-vocabulary coverage and finiteness.
    if not np.isfinite(new_row).all():
        raise RuntimeError(
            f"trained row for {lang!r} contains non-finite values")
    n_fill = int((new_row <= MISSING_TOKEN_FILL_DETECTION_BOUND).sum())
    if n_fill:
        raise RuntimeError(
            f"trained tokenizer for {lang!r} is missing {n_fill} of "
            f"{vocab_size} base-vocabulary tokens (assembly fill detected); "
            f"the trainer must produce a score for every base-vocab token")

    new_row = _match_real_token_scale(new_row, weights, ref_vocab, lang)

    # Clamp outcome report (the clamp itself is applied at load time, to this
    # row like every other): one-sided, plateau-only.
    c = cal.unseen_token_constant
    row_min = float(new_row.min())
    plateau = int((new_row == new_row.min()).sum())
    if row_min > c:
        clamp_outcome = (f"plateau of {plateau:,} entries at {row_min:.6g} "
                         f"will be lowered to c={c} at load")
    else:
        clamp_outcome = (f"row minimum {row_min:.6g} is at or below c={c}; "
                         f"the row is left unchanged by the unseen-token "
                         f"constant (reference semantics for such rows)")
    print(f"Unseen-token constant: {clamp_outcome}")

    new_langs = list(langs) + [lang]
    new_weights = np.empty((len(new_langs), vocab_size), dtype=np.float32)
    new_weights[:-1] = weights
    new_weights[-1] = new_row
    del weights
    gc.collect()

    counts = dict(cal.train_counts)
    counts[lang] = n_l
    provenance = dict(cal.provenance)
    provenance.setdefault("added_languages", [])
    provenance["added_languages"] = list(provenance["added_languages"]) + [{
        "lang": lang, "n": n_l, "method": method, "em_rounds": em_rounds,
        "date": datetime.date.today().isoformat(),
    }]

    tau_row = None
    if n_l < cal.head_n:
        # The threshold needs a scoring pass against the FULL new model
        # (including the new language) on the clamped matrix. Write a
        # temporary container whose calibration carries a placeholder row for
        # the new language (excluded, low_calibration, zero lines processed:
        # literally true at this point), load it calibrated, estimate, then
        # write the final container with the estimated row.
        placeholder = TauRow(tau=float("-inf"), excluded=True,
                             cause=CAUSE_LOW_CALIBRATION,
                             n_scoreable=0, n_self_won=0)
        provisional = dataclass_replace(
            cal, group_a={**cal.group_a, lang: placeholder},
            train_counts=counts, provenance=provenance)
        tmp_path = output_path.with_name(output_path.stem + ".tmp.unilid")
        if tmp_path.resolve() == model_path.resolve():
            raise ValueError(
                f"temporary path {tmp_path} would overwrite the input model; "
                f"choose a different output_path")
        try:
            write_unilid(tmp_path, base_tok_bytes, new_langs,
                         new_weights, provisional)
            print(f"Estimating the re-examination threshold for {lang!r} "
                  f"(N={n_l:,} < head_n={cal.head_n:,})...")
            tmp_model = UnilidModel(tmp_path, calibrated=True)
            with open(train_file, encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
            tau_row = estimate_tau(tmp_model, lang, lines, n_l)
            del tmp_model
            gc.collect()
        finally:
            tmp_path.unlink(missing_ok=True)
        final_cal = dataclass_replace(
            cal, group_a={**cal.group_a, lang: tau_row},
            train_counts=counts, provenance=provenance)
        if tau_row.excluded:
            print(f"Threshold: EXCLUDED (cause {tau_row.cause}; "
                  f"{tau_row.n_self_won:,} own-won lines of "
                  f"{tau_row.n_scoreable:,} scoreable); {lang!r} will never "
                  f"be re-examined")
        else:
            print(f"Threshold: tau={tau_row.tau:.6g} "
                  f"({tau_row.n_self_won:,} own-won lines of "
                  f"{tau_row.n_scoreable:,} scoreable)")
    else:
        print(f"N={n_l:,} >= head_n={cal.head_n:,}: no re-examination "
              f"threshold (the language's predictions are not re-examined)")
        final_cal = dataclass_replace(
            cal, train_counts=counts, provenance=provenance)

    final_cal.runtime_for(new_langs)  # validate before writing the output
    write_unilid(output_path, base_tok_bytes, new_langs,
                 new_weights, final_cal)
    is_candidate = n_l >= cal.replacement_min_n
    print(f"Wrote {output_path} ({len(new_langs):,} languages). "
          f"{lang!r} is{'' if is_candidate else ' NOT'} a replacement "
          f"candidate (N={n_l:,}, requirement {cal.replacement_min_n:,}).")
    return {"lang": lang, "n": n_l, "clamp": clamp_outcome,
            "tau_row": tau_row, "replacement_candidate": is_candidate,
            "output": str(output_path), "tokenizer": str(tok_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Add a language to an existing calibrated .unilid model")
    parser.add_argument("model", type=Path, help="Input .unilid model")
    parser.add_argument("lang", help="Language code for the new language")
    parser.add_argument("train_file", type=Path,
                        help="Training text file, one line per sample")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output .unilid path (must differ from the input)")
    parser.add_argument("--method", choices=["sp", "em"], default="sp",
                        help="Per-language training path (default sp, the "
                             "release-verified one)")
    parser.add_argument("--em-rounds", type=int, default=20)
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Calibration JSON for a version-1 model file")
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args(argv)
    add_language(args.model, args.lang, args.train_file, args.output,
                 method=args.method, em_rounds=args.em_rounds,
                 calibration=args.calibration, workdir=args.workdir)


if __name__ == "__main__":
    main()
