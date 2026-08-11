"""Calibrated inference for UNILID.

Implements the calibration mechanism of the UniLID paper ("Calibrated UniLID",
section 4 and the development-protocol appendix): one shared unseen-token constant
c applied as a one-sided clamp to every language's unseen-token log-probabilities,
and an inference-time re-examination of low-margin predictions into two disjoint
groups of languages:

- group A: languages with fewer than ``head_n`` training samples, each with its own
  threshold tau at the size-adaptive percentile
  q_L = margin_q * (1 - min(N_L, head_n) / head_n) of the margins on its own
  training lines;
- group B: the high-entropy group (membership is part of the calibration artifact;
  the identification criteria are in the paper), each with its own threshold at the
  fixed ``group_b_percentile``.

A re-examined prediction moves to the highest-ranked of the candidates ranked
2..topk with at least ``replacement_min_n`` training samples and a score within
``proximity_bound`` natural-log units of the top candidate; if no candidate
qualifies, the prediction is unchanged.

All constants live in the calibration artifact, not in this module. The reference
implementation this module ports is in the analysis repository:
``analysis/floor_equalization.py`` (clamp), ``analysis/solo_gates.py`` (threshold
estimation), and ``analysis/external_bench_eval.py`` / ``analysis/gate_variants.py``
(gate + replacement walk).

Precision policy (deliberate, matches the reference arithmetic):
- weights, clamp, candidate scores, and the proximity comparison are float32;
- the margin (gap) used by the gate is a float32 subtraction, then widened to
  float64 for the comparison against tau (the reference compares a float32 gap
  array against a float64 tau array, which evaluates in float64);
- threshold estimation computes its margins in float64 (the reference builds them
  from Python floats returned by the scorer before any float32 materialization).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CALIBRATION_FORMAT_VERSION = 1

CAUSE_NONE = ""
CAUSE_LOW_CALIBRATION = "low_calibration"
CAUSE_ZERO_STRENGTH = "zero_strength"
_VALID_CAUSES = (CAUSE_NONE, CAUSE_LOW_CALIBRATION, CAUSE_ZERO_STRENGTH)


class UnilidCalibrationError(ValueError):
    """A calibration artifact is missing, malformed, or inconsistent with a model."""


@dataclass(frozen=True)
class TauRow:
    """One language's re-examination threshold record.

    ``n_self_won`` counts the calibration lines on which the language was
    top-scoring BEFORE the finite-margin filter (the reference CSVs record the
    same quantity), so ``n_self_won >= min_calib_lines`` does not by itself imply
    the language was not excluded.
    """

    tau: float  # -inf when excluded
    excluded: bool
    cause: str
    n_scoreable: int
    n_self_won: int

    def __post_init__(self):
        if self.cause not in _VALID_CAUSES:
            raise UnilidCalibrationError(
                f"invalid cause {self.cause!r}; expected one of {_VALID_CAUSES}")
        if self.excluded and self.cause == CAUSE_NONE:
            raise UnilidCalibrationError(
                "excluded threshold row must carry a cause")
        if self.excluded:
            if self.tau != float("-inf"):
                raise UnilidCalibrationError(
                    f"excluded threshold row must have tau=-inf, got "
                    f"{self.tau!r}")
        elif not math.isfinite(self.tau):
            raise UnilidCalibrationError(
                f"non-excluded threshold row must have a finite tau, got "
                f"{self.tau!r}")


@dataclass(frozen=True)
class Calibration:
    """The complete calibration artifact for one model."""

    unseen_token_constant: float
    head_n: int
    replacement_min_n: int
    proximity_bound: float
    topk: int
    margin_q: float
    group_b_percentile: float
    calib_max: int
    min_calib_lines: int
    calib_seed: int
    group_a: Dict[str, TauRow]
    group_b: Dict[str, TauRow]
    train_counts: Dict[str, int]
    provenance: dict

    # ------------------------------------------------------------------ JSON I/O

    @staticmethod
    def _row_from_json(lang: str, obj: dict) -> TauRow:
        required = {"tau", "excluded", "n_scoreable", "n_self_won"}
        missing = required - set(obj)
        if missing:
            raise UnilidCalibrationError(
                f"threshold row for {lang!r} is missing fields {sorted(missing)}")
        excluded = obj["excluded"]
        if not isinstance(excluded, bool):
            raise UnilidCalibrationError(
                f"threshold row for {lang!r}: excluded must be a JSON boolean, "
                f"got {type(excluded).__name__}")
        tau = obj["tau"]
        if tau is None:
            if not excluded:
                raise UnilidCalibrationError(
                    f"threshold row for {lang!r}: tau is null but excluded is "
                    f"false; a non-excluded row needs a finite tau")
            tau = float("-inf")
        else:
            tau = float(tau)
            if not math.isfinite(tau):
                raise UnilidCalibrationError(
                    f"threshold row for {lang!r}: non-finite tau {tau!r} in JSON "
                    f"(excluded rows store null)")
            if excluded:
                raise UnilidCalibrationError(
                    f"threshold row for {lang!r}: excluded is true but tau is "
                    f"{tau!r}; excluded rows store tau as null")
        return TauRow(tau=tau, excluded=excluded, cause=obj.get("cause", ""),
                      n_scoreable=int(obj["n_scoreable"]),
                      n_self_won=int(obj["n_self_won"]))

    @staticmethod
    def _row_to_json(row: TauRow) -> dict:
        return {"tau": None if row.excluded else row.tau,
                "excluded": row.excluded, "cause": row.cause,
                "n_scoreable": row.n_scoreable, "n_self_won": row.n_self_won}

    @classmethod
    def from_json_dict(cls, data: dict) -> "Calibration":
        version = data.get("format_version")
        if version != CALIBRATION_FORMAT_VERSION:
            raise UnilidCalibrationError(
                f"calibration format_version {version!r} not supported (reader "
                f"supports {CALIBRATION_FORMAT_VERSION})")
        try:
            consts = data["constants"]
            group_a_obj = data["group_a"]["thresholds"]
            group_b_obj = data["group_b"]["thresholds"]
            train_counts = data["train_counts"]
        except KeyError as e:
            raise UnilidCalibrationError(
                f"calibration JSON is missing required section {e}") from None
        const_names = ["unseen_token_constant", "head_n", "replacement_min_n",
                       "proximity_bound", "topk", "margin_q",
                       "group_b_percentile", "calib_max", "min_calib_lines",
                       "calib_seed"]
        missing = [c for c in const_names if c not in consts]
        if missing:
            raise UnilidCalibrationError(
                f"calibration constants block is missing {missing}")
        group_a = {lang: cls._row_from_json(lang, row)
                   for lang, row in group_a_obj.items()}
        group_b = {lang: cls._row_from_json(lang, row)
                   for lang, row in group_b_obj.items()}
        overlap = set(group_a) & set(group_b)
        if overlap:
            raise UnilidCalibrationError(
                f"group A and group B overlap: {sorted(overlap)}")
        counts = {}
        for lang, n in train_counts.items():
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise UnilidCalibrationError(
                    f"train_counts[{lang!r}] must be a non-negative integer, "
                    f"got {n!r}")
            counts[lang] = n
        topk = int(consts["topk"])
        if topk < 2:
            raise UnilidCalibrationError(
                f"constants.topk must be at least 2 (the margin needs a "
                f"runner-up), got {topk}")
        head_n = int(consts["head_n"])
        if head_n <= 0:
            raise UnilidCalibrationError(
                f"constants.head_n must be positive, got {head_n}")
        return cls(
            unseen_token_constant=float(consts["unseen_token_constant"]),
            head_n=int(consts["head_n"]),
            replacement_min_n=int(consts["replacement_min_n"]),
            proximity_bound=float(consts["proximity_bound"]),
            topk=int(consts["topk"]),
            margin_q=float(consts["margin_q"]),
            group_b_percentile=float(consts["group_b_percentile"]),
            calib_max=int(consts["calib_max"]),
            min_calib_lines=int(consts["min_calib_lines"]),
            calib_seed=int(consts["calib_seed"]),
            group_a=group_a, group_b=group_b, train_counts=counts,
            provenance=dict(data.get("provenance", {})))

    def to_json_dict(self) -> dict:
        return {
            "format_version": CALIBRATION_FORMAT_VERSION,
            "constants": {
                "unseen_token_constant": self.unseen_token_constant,
                "head_n": self.head_n,
                "replacement_min_n": self.replacement_min_n,
                "proximity_bound": self.proximity_bound,
                "topk": self.topk,
                "margin_q": self.margin_q,
                "group_b_percentile": self.group_b_percentile,
                "calib_max": self.calib_max,
                "min_calib_lines": self.min_calib_lines,
                "calib_seed": self.calib_seed,
            },
            "group_a": {
                "criterion": ("predicted language has fewer than head_n "
                              "training samples"),
                "thresholds": {lang: self._row_to_json(row)
                               for lang, row in sorted(self.group_a.items())},
            },
            "group_b": {
                "criterion": ("the high-entropy group; identification criteria "
                              "in the paper's development-protocol appendix"),
                "thresholds": {lang: self._row_to_json(row)
                               for lang, row in sorted(self.group_b.items())},
            },
            "train_counts": {lang: n
                             for lang, n in sorted(self.train_counts.items())},
            "provenance": self.provenance,
        }

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> "Calibration":
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise UnilidCalibrationError(
                f"calibration payload is not valid UTF-8 JSON: {e}") from None
        return cls.from_json_dict(data)

    def to_json_bytes(self) -> bytes:
        # repr-level float precision: json.dumps uses repr(float), which
        # round-trips float64 exactly; tau values must never be rounded.
        return json.dumps(self.to_json_dict(), ensure_ascii=False).encode("utf-8")

    @classmethod
    def from_json_file(cls, path) -> "Calibration":
        path = Path(path)
        if not path.is_file():
            raise UnilidCalibrationError(
                f"calibration file not found: {path}")
        return cls.from_json_bytes(path.read_bytes())

    def to_json_file(self, path) -> Path:
        path = Path(path)
        path.write_bytes(self.to_json_bytes())
        return path

    # ------------------------------------------------------- model consistency

    def runtime_for(self, langs: Sequence[str]) -> "CalibrationRuntime":
        """Build per-model arrays, enforcing every consistency requirement of the
        reference loader (analysis/external_bench_eval._load_gate_thresholds)."""
        lang_to_pos = {}
        for i, lang in enumerate(langs):
            if lang in lang_to_pos:
                raise UnilidCalibrationError(
                    f"model language list contains {lang!r} twice")
            lang_to_pos[lang] = i
        n_lang = len(langs)

        missing_counts = [l for l in langs if l not in self.train_counts]
        if missing_counts:
            raise UnilidCalibrationError(
                f"{len(missing_counts)} model language(s) missing from "
                f"train_counts, first: {missing_counts[:5]}")
        N = np.array([self.train_counts[l] for l in langs], dtype=np.int64)

        expected_a = {l for i, l in enumerate(langs) if N[i] < self.head_n}
        if expected_a != set(self.group_a):
            extra = sorted(set(self.group_a) - expected_a)[:5]
            absent = sorted(expected_a - set(self.group_a))[:5]
            raise UnilidCalibrationError(
                f"group A must equal the set of model languages with fewer than "
                f"{self.head_n:,} training samples: {len(self.group_a)} in the "
                f"artifact vs {len(expected_a)} expected "
                f"(unexpected: {extra}, missing: {absent})")

        for lang in self.group_b:
            if lang not in lang_to_pos:
                raise UnilidCalibrationError(
                    f"group B language {lang!r} is not in the model")
            n_l = int(N[lang_to_pos[lang]])
            if n_l < self.head_n:
                raise UnilidCalibrationError(
                    f"group B language {lang!r} has N={n_l:,} < head_n "
                    f"({self.head_n:,}); the two groups must be disjoint by "
                    f"construction")

        # tau arrays: float64, -inf default; excluded rows are forced to -inf by
        # TauRow's invariant (mirrors _load_tau_csv).
        tau_a = np.full(n_lang, -np.inf, dtype=np.float64)
        for lang, row in self.group_a.items():
            tau_a[lang_to_pos[lang]] = row.tau
        tau_b = np.full(n_lang, -np.inf, dtype=np.float64)
        for lang, row in self.group_b.items():
            tau_b[lang_to_pos[lang]] = row.tau

        in_a = np.zeros(n_lang, dtype=bool)
        in_a[[lang_to_pos[l] for l in self.group_a]] = True
        in_b = np.zeros(n_lang, dtype=bool)
        in_b[[lang_to_pos[l] for l in self.group_b]] = True
        if (in_a & in_b).any():
            raise UnilidCalibrationError("group A and group B overlap")

        return CalibrationRuntime(calibration=self, N=N, in_a=in_a, in_b=in_b,
                                  tau_a=tau_a, tau_b=tau_b)


@dataclass(frozen=True)
class CalibrationRuntime:
    """Per-model arrays derived from a Calibration and a language order."""

    calibration: Calibration
    N: np.ndarray      # int64 [n_lang]
    in_a: np.ndarray   # bool  [n_lang]
    in_b: np.ndarray   # bool  [n_lang]
    tau_a: np.ndarray  # float64 [n_lang], -inf outside group A
    tau_b: np.ndarray  # float64 [n_lang], -inf outside group B


# ------------------------------------------------------------------- the clamp

def apply_unseen_token_constant(W: np.ndarray, target: float
                                ) -> Tuple[np.ndarray, int]:
    """One-sided clamp of each language's unseen-token log-probabilities to
    ``target``. Port of analysis/floor_equalization.build_equalized_weights:
    a row's unseen tokens are its exact minimum-value plateau; the plateau is
    lowered to ``target`` only when it lies above it; rows whose minimum is at or
    below ``target`` are left entirely untouched. No renormalization (the scorer
    compares unnormalized log-weight sums)."""
    out = np.array(W, dtype=np.float32)
    n_mod = 0
    for i in range(out.shape[0]):
        row = out[i]
        floor = row.min()
        if floor > target:
            row[row == floor] = np.float32(target)
            n_mod += 1
    if not np.isfinite(out).all():
        raise UnilidCalibrationError(
            "non-finite weights after applying the unseen-token constant")
    return out, n_mod


# ---------------------------------------------------------- gate and walk port

def _walk_replacement(prox_limit: float, replacement_min_n: int,
                      ids_row: np.ndarray, scores_row: np.ndarray,
                      N: np.ndarray) -> Tuple[str, Optional[int]]:
    """Port of analysis/gate_variants._walk_replacement. ``scores_row`` must be
    float32 (the proximity comparison is float32 by the precision policy)."""
    top1_score = scores_row[0]
    saw_min_n = False
    for j in range(1, len(ids_row)):
        cid = int(ids_row[j])
        if cid < 0:
            continue  # unfilled candidate slot
        if N[cid] < replacement_min_n:
            continue
        saw_min_n = True
        if (top1_score - scores_row[j]) > prox_limit:
            continue
        return "moved", cid
    if saw_min_n:
        return "blocked_by_proximity", None
    return "no_cand", None


def re_examine(topk_ids: np.ndarray, topk_scores: np.ndarray,
               runtime: CalibrationRuntime
               ) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    """Gate + replacement walk over a batch of top-k rows. Port of
    analysis/external_bench_eval._gate_walk_and_merge with the base prediction
    defined as each row's own rank-1 candidate (the run_eval convention), which
    makes the reference's agree_mask trivially all-true and its fresh-copy
    two-step apply equivalent to this single per-row pass (the two groups are
    disjoint and no group member can be a walk target, since acceptance requires
    N >= replacement_min_n).

    ``topk_ids``: int [n, k]; ``topk_scores``: float32 [n, k]; unfilled slots are
    id -1. Returns (final language index per row, per-group stats).
    """
    if topk_scores.dtype != np.float32:
        raise UnilidCalibrationError(
            f"topk_scores must be float32 (the reference arithmetic), got "
            f"{topk_scores.dtype}")
    if topk_ids.shape != topk_scores.shape:
        raise UnilidCalibrationError(
            f"topk_ids shape {topk_ids.shape} != topk_scores shape "
            f"{topk_scores.shape}")
    cal = runtime.calibration
    n_rows = topk_ids.shape[0]
    base_pred = topk_ids[:, 0].astype(np.int64)
    if (base_pred < 0).any():
        raise UnilidCalibrationError(
            "a top-k row has no rank-1 candidate; empty inputs must be filtered "
            "before re-examination")

    # float32 gap, widened to float64 by the comparison against float64 tau
    # (matches the reference's array comparison dtype).
    gap = topk_scores[:, 0] - topk_scores[:, 1]

    in_set_a = runtime.in_a[base_pred]
    in_set_b = runtime.in_b[base_pred]
    if (in_set_a & in_set_b).any():
        raise UnilidCalibrationError(
            "a row's prediction falls in both groups; expected disjoint")

    below_a = in_set_a & (gap < runtime.tau_a[base_pred])
    below_b = in_set_b & (gap < runtime.tau_b[base_pred])
    if (below_a & below_b).any():
        raise UnilidCalibrationError(
            "a row is flagged for re-examination under both groups; expected "
            "disjoint")

    final = base_pred.copy()
    stats = {}
    for name, mask in (("group_a", below_a), ("group_b", below_b)):
        n_moved = n_blocked = n_no_cand = 0
        for r in np.where(mask)[0].tolist():
            outcome, cid = _walk_replacement(
                cal.proximity_bound, cal.replacement_min_n,
                topk_ids[r], topk_scores[r], runtime.N)
            if outcome == "moved":
                final[r] = cid
                n_moved += 1
            elif outcome == "blocked_by_proximity":
                n_blocked += 1
            else:
                n_no_cand += 1
        n_examined = int(mask.sum())
        if n_examined != n_moved + n_blocked + n_no_cand:
            raise UnilidCalibrationError(
                f"re-examination bookkeeping mismatch in {name}: "
                f"{n_examined} examined != {n_moved} moved + {n_blocked} "
                f"blocked_by_proximity + {n_no_cand} no_cand")
        stats[name] = {"n_examined": n_examined, "n_moved": n_moved,
                       "n_blocked_by_proximity": n_blocked,
                       "n_no_cand": n_no_cand}
    return final, stats


# --------------------------------------------------------- threshold estimation

def estimate_tau(model, lang: str, lines: List[str], n_l: int) -> TauRow:
    """Estimate one language's re-examination threshold from its own training
    lines. Port of the recipe of record (analysis/solo_gates.py, the run that
    produced tau_floor21_gate.csv), including both exclusion branches.

    ``model`` must be a calibrated-loaded UnilidModel (margins are defined on the
    distributions with the unseen-token constant applied); ``lines`` are the
    language's training lines; ``n_l`` its training-line count.

    Sampling note: this uses a fresh np.random.default_rng(calib_seed) for the
    single language, which reproduces the reference batch run only for the first
    language in its loop (the reference shares one generator sequentially across
    languages). Re-deriving a full historical tau table therefore requires the
    analysis-repo batch script, not repeated calls to this function.
    """
    if not getattr(model, "calibrated", False):
        raise UnilidCalibrationError(
            "estimate_tau requires a calibrated-loaded model (the margin is "
            "defined on the clamped matrix); load with calibrated=True")
    cal: Calibration = model.calibration
    if lang not in model._lang_to_idx:
        raise UnilidCalibrationError(f"language {lang!r} is not in the model")
    li = model._lang_to_idx[lang]

    lines = [l for l in lines if l]
    if len(lines) > cal.calib_max:
        rng = np.random.default_rng(cal.calib_seed)
        keep = sorted(rng.choice(len(lines), cal.calib_max, replace=False))
        lines = [lines[k] for k in keep]

    preprocessed = []
    for line in lines:
        pt = model.preprocess(line)
        if pt:
            preprocessed.append(pt)
    topk = model.model.top_k_of_cached_weight_sets_batch(preprocessed, cal.topk)

    wins = [c for c in topk if c and c[0][0] == li]
    # float64 margins from the scorer's Python floats (reference: _gap on the
    # raw candidate tuples, before any float32 materialization).
    gaps = np.array([c[0][1] - c[1][1] if len(c) >= 2 else float("inf")
                     for c in wins])
    gaps = gaps[np.isfinite(gaps)]

    low_calib = len(gaps) < cal.min_calib_lines
    q_l = cal.margin_q * (1.0 - min(float(n_l), float(cal.head_n)) / cal.head_n)
    zero_strength = q_l <= 0.0
    excluded = low_calib or zero_strength
    cause = (CAUSE_LOW_CALIBRATION if low_calib
             else CAUSE_ZERO_STRENGTH if zero_strength else CAUSE_NONE)
    tau = float("-inf") if excluded else float(np.percentile(gaps, q_l))
    return TauRow(tau=tau, excluded=excluded, cause=cause,
                  n_scoreable=len(topk), n_self_won=len(wins))
