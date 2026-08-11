"""Calibration artifact utilities: export, bundle, and per-language threshold
re-estimation for .unilid models.

Deriving a calibration from scratch for a new model (including high-entropy
group identification) is a separate, non-incremental procedure described in the
paper's development-protocol appendix; this CLI manages an existing artifact.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np

from .calibration import Calibration, UnilidCalibrationError, estimate_tau
from .model_io import (
    UnilidModel,
    load_unilid_raw,
    read_calibration,
    subset_rows,
    write_unilid,
)


def _require_bundled(model_path: Path) -> Calibration:
    cal = read_calibration(model_path)
    if cal is None:
        raise UnilidCalibrationError(
            f"{model_path} is a version-1 file with no bundled calibration")
    return cal


def cmd_export(args):
    cal = _require_bundled(args.model)
    cal.to_json_file(args.output)
    print(f"Wrote {args.output}")


def cmd_bundle(args):
    if read_calibration(args.model) is not None:
        raise UnilidCalibrationError(
            f"{args.model} already bundles a calibration; export/edit/re-bundle "
            f"from the version-1 base file instead")
    cal = Calibration.from_json_file(args.calibration)
    base_tok_bytes, weights, langs = load_unilid_raw(args.model)
    cal.runtime_for(langs)  # full consistency validation before writing
    out = Path(args.output).with_suffix(".unilid")
    if out.resolve() == Path(args.model).resolve():
        raise ValueError("output must differ from the input model file")
    write_unilid(out, base_tok_bytes, langs,
                 np.array(weights, dtype=np.float32), cal)
    print(f"Wrote {out} (version 2, calibration bundled, "
          f"{len(langs):,} languages)")


def cmd_estimate(args):
    model_path = Path(args.model)
    cal = _require_bundled(model_path)
    lang = args.lang
    if lang not in cal.train_counts:
        raise UnilidCalibrationError(
            f"language {lang!r} is not in the calibration train_counts")
    n_l = cal.train_counts[lang]
    if lang in cal.group_b:
        raise UnilidCalibrationError(
            f"{lang!r} is in the high-entropy group (group B); its threshold "
            f"uses the fixed group_b_percentile, and this command implements "
            f"only the size-adaptive group A recipe. Group B thresholds are "
            f"re-derived with the analysis pipeline, not here.")
    if n_l >= cal.head_n:
        raise UnilidCalibrationError(
            f"{lang!r} has N={n_l:,} >= head_n={cal.head_n:,} and is not in "
            f"the high-entropy group: no re-examination threshold applies "
            f"to it")
    out = Path(args.output).with_suffix(".unilid")
    if out.resolve() == model_path.resolve():
        raise ValueError("output must differ from the input model file")

    model = UnilidModel(model_path, calibrated=True)
    with open(args.train_file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
    row = estimate_tau(model, lang, lines, n_l)
    print(f"{lang}: tau={row.tau} excluded={row.excluded} cause={row.cause!r} "
          f"n_scoreable={row.n_scoreable} n_self_won={row.n_self_won}")
    new_cal = dataclass_replace(cal, group_a={**cal.group_a, lang: row})
    del model
    gc.collect()

    base_tok_bytes, weights, langs = load_unilid_raw(model_path)
    new_cal.runtime_for(langs)
    write_unilid(out, base_tok_bytes, langs,
                 np.array(weights, dtype=np.float32), new_cal)
    print(f"Wrote {out}")


def cmd_subset(args):
    model_path = Path(args.model)
    out = Path(args.output).with_suffix(".unilid")
    if out.resolve() == model_path.resolve():
        raise ValueError("output must differ from the input model file")

    if args.langs:
        languages = [l.strip() for l in args.langs.split(",") if l.strip()]
    else:
        languages = [l.strip() for l in
                     Path(args.langs_file).read_text().splitlines()
                     if l.strip()]

    base_tok_bytes, weights, langs = load_unilid_raw(model_path)
    cal = read_calibration(model_path)
    sub_weights, sub_langs = subset_rows(weights, langs, languages)
    print(f"Keeping {len(sub_langs)} of {len(langs)} languages")

    if cal is None:
        if args.recalibrate:
            raise UnilidCalibrationError(
                f"{model_path} is a version-1 file with no calibration; there "
                f"are no thresholds to re-estimate (bundle a calibration "
                f"first)")
        write_unilid(out, base_tok_bytes, sub_langs, sub_weights, None)
        print(f"Wrote {out} (version 1, no calibration)")
        return

    sub_cal = cal.subset_for(sub_langs)
    if args.recalibrate:
        corpus_dir = Path(args.recalibrate)
        group_a_langs = sorted(sub_cal.group_a)
        print(f"Re-estimating {len(group_a_langs)} group A threshold(s) "
              f"against the subset model...")
        model = UnilidModel(model_path, calibrated=True, languages=sub_langs)
        rows = {}
        for lang in group_a_langs:
            train_file = corpus_dir / f"{lang}_train.txt"
            if not train_file.is_file():
                raise FileNotFoundError(
                    f"training file for {lang!r} not found: {train_file} "
                    f"(--recalibrate expects <corpus_dir>/<lang>_train.txt)")
            with open(train_file, encoding="utf-8") as f:
                lines = [l.rstrip("\n") for l in f if l.rstrip("\n")]
            rows[lang] = estimate_tau(model, lang, lines,
                                      sub_cal.train_counts[lang])
            print(f"  {lang}: tau={rows[lang].tau} "
                  f"excluded={rows[lang].excluded} cause={rows[lang].cause!r}")
        del model
        gc.collect()
        provenance = dict(sub_cal.provenance)
        provenance["subset"] = {
            "n_languages": len(sub_langs),
            "thresholds": "re-estimated against the subset model "
                          "(--recalibrate)",
        }
        sub_cal = dataclass_replace(sub_cal, group_a=rows,
                                    provenance=provenance)
    sub_cal.runtime_for(sub_langs)
    write_unilid(out, base_tok_bytes, sub_langs, sub_weights, sub_cal)
    which = "re-estimated" if args.recalibrate else "carried over"
    print(f"Wrote {out} (version 2, {len(sub_langs)} languages, "
          f"thresholds {which})")
    if not args.recalibrate:
        print("Carried thresholds make re-examination fire at most as often "
              "as calibrated (margins against a smaller candidate set are at "
              "least as large); pass --recalibrate <corpus_dir> to "
              "re-estimate them")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Calibration artifact utilities for .unilid models")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="Write a model's bundled calibration "
                                      "to a standalone JSON file")
    p.add_argument("model", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("bundle", help="Attach a calibration JSON to a "
                                      "version-1 model, writing a version-2 "
                                      "container")
    p.add_argument("model", type=Path)
    p.add_argument("calibration", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.set_defaults(func=cmd_bundle)

    p = sub.add_parser("estimate", help="Re-estimate one language's "
                                        "re-examination threshold from its "
                                        "training file")
    p.add_argument("model", type=Path)
    p.add_argument("lang")
    p.add_argument("train_file", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("subset", help="Write a model restricted to a subset "
                                      "of its languages (scoring cost is "
                                      "linear in the number of languages)")
    p.add_argument("model", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    langs_group = p.add_mutually_exclusive_group(required=True)
    langs_group.add_argument("--langs",
                             help="Comma-separated language codes to keep")
    langs_group.add_argument("--langs-file", type=Path,
                             help="File with one language code per line")
    p.add_argument("--recalibrate", type=Path, default=None,
                   metavar="CORPUS_DIR",
                   help="Optionally re-estimate each retained group A "
                        "language's threshold from "
                        "CORPUS_DIR/<lang>_train.txt against the subset "
                        "model; without this, thresholds are carried over "
                        "from the full model (re-examination then fires at "
                        "most as often as calibrated)")
    p.set_defaults(func=cmd_subset)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
