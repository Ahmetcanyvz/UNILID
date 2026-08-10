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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
