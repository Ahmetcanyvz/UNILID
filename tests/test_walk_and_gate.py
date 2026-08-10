"""Unit tests for unilid.calibration._walk_replacement and re_examine, the
ports of analysis/gate_variants.py::_walk_replacement (~825-868) and
analysis/external_bench_eval.py::_gate_walk_and_merge (~596-689). Pure
numpy/Python; no Rust extension needed.
"""
import numpy as np
import pytest

from unilid.calibration import (
    Calibration,
    CalibrationRuntime,
    UnilidCalibrationError,
    _walk_replacement,
    re_examine,
)

PROX = 21.0
MIN_N = 100000


# --------------------------------------------------------- _walk_replacement

def test_walk_skips_below_min_n_candidate_and_walks_rank_order():
    # rank2 (id=10) has insufficient N and must be skipped without accepting;
    # rank3 (id=11) qualifies on N and proximity and is accepted.
    ids_row = np.array([0, 10, 11, 12])
    scores_row = np.array([0.0, -5.0, -3.0, -1.0], dtype=np.float32)
    N = np.zeros(20, dtype=np.int64)
    N[10] = 50          # below MIN_N
    N[11] = 200000       # qualifies -> should be the accepted candidate
    N[12] = 200000       # also qualifies, but rank order means 11 wins first

    outcome, cid = _walk_replacement(PROX, MIN_N, ids_row, scores_row, N)
    assert (outcome, cid) == ("moved", 11)


def test_walk_skips_unfilled_slot_id_minus_one():
    ids_row = np.array([0, -1, 11])
    scores_row = np.array([0.0, -100.0, -3.0], dtype=np.float32)
    N = np.zeros(20, dtype=np.int64)
    N[11] = 200000

    outcome, cid = _walk_replacement(PROX, MIN_N, ids_row, scores_row, N)
    assert (outcome, cid) == ("moved", 11)


def test_walk_no_candidate_meets_min_n_is_no_cand():
    ids_row = np.array([0, 10, 11])
    scores_row = np.array([0.0, -1.0, -2.0], dtype=np.float32)
    N = np.zeros(20, dtype=np.int64)  # nobody meets MIN_N

    outcome, cid = _walk_replacement(PROX, MIN_N, ids_row, scores_row, N)
    assert (outcome, cid) == ("no_cand", None)


def test_walk_blocked_by_proximity_when_qualifying_candidates_are_too_far():
    ids_row = np.array([0, 10, 11])
    scores_row = np.array([0.0, -50.0, -60.0], dtype=np.float32)  # both gaps > 21
    N = np.zeros(20, dtype=np.int64)
    N[10] = 200000
    N[11] = 200000

    outcome, cid = _walk_replacement(PROX, MIN_N, ids_row, scores_row, N)
    assert (outcome, cid) == ("blocked_by_proximity", None)


def test_walk_proximity_boundary_exact_equal_is_accepted_strict_greater_rejects():
    ids_row = np.array([0, 10])
    N = np.zeros(20, dtype=np.int64)
    N[10] = MIN_N  # exactly at replacement_min_n -> N[cid] < MIN_N is False -> qualifies

    # Exactly at the proximity bound (gap == 21.0 exactly, representable in
    # float32): accepted, since the check is `> prox_limit`, not `>=`.
    scores_at_bound = np.array([0.0, -21.0], dtype=np.float32)
    outcome, cid = _walk_replacement(PROX, MIN_N, ids_row, scores_at_bound, N)
    assert (outcome, cid) == ("moved", 10)

    # The next representable float32 value past the bound: gap is now
    # (by a hair) > 21.0 -> rejected. Only one candidate slot, and it meets
    # N but fails proximity, so the outcome is blocked_by_proximity.
    just_past = np.nextafter(np.float32(-21.0), np.float32(-1e9)).astype(np.float32)
    scores_past_bound = np.array([0.0, just_past], dtype=np.float32)
    assert (np.float32(0.0) - just_past) > np.float32(PROX)  # sanity: gap truly exceeds 21.0
    outcome2, cid2 = _walk_replacement(PROX, MIN_N, ids_row, scores_past_bound, N)
    assert (outcome2, cid2) == ("blocked_by_proximity", None)


# ------------------------------------------------------------- re_examine()

def _minimal_calibration(topk: int) -> Calibration:
    """A Calibration whose group_a/group_b/train_counts/provenance are not
    used by re_examine (only proximity_bound and replacement_min_n are read
    off `runtime.calibration`); the rest are placeholders."""
    return Calibration(
        unseen_token_constant=-21.0, head_n=18000,
        replacement_min_n=MIN_N, proximity_bound=PROX, topk=topk,
        margin_q=5.0, group_b_percentile=5.0, calib_max=2000,
        min_calib_lines=200, calib_seed=0,
        group_a={}, group_b={}, train_counts={}, provenance={})


def _build_batch():
    """7 hand-constructed rows exercising every re_examine code path in one
    batch (languages 0-5; language 4 is the well-resourced walk target,
    language 5 is under-resourced):

    row0: group A, gap < tau -> moved to lang4
    row1: group A, gap == tau exactly -> NOT examined (strict <)
    row2: group B, gap < tau -> moved to lang4
    row3: neither group, huge gap -> never touched
    row4: group A but excluded (tau=-inf) -> never re-examined despite a
          small gap
    row5: group A, gap < tau, but no candidate meets replacement_min_n ->
          no_cand, unchanged
    row6: group A, gap < tau, one candidate meets N but fails proximity,
          the other candidate is under-resourced -> blocked_by_proximity,
          unchanged
    """
    N = np.array([500, 25000, 100, 300, 200000, 50], dtype=np.int64)
    in_a = np.array([True, False, False, True, False, False])
    in_b = np.array([False, True, False, False, False, False])
    tau_a = np.array([2.0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    tau_b = np.array([-np.inf, 1.0, -np.inf, -np.inf, -np.inf, -np.inf])

    cal = _minimal_calibration(topk=3)
    runtime = CalibrationRuntime(calibration=cal, N=N, in_a=in_a, in_b=in_b,
                                 tau_a=tau_a, tau_b=tau_b)

    ids = np.array([
        [0, 4, 5],
        [0, 4, 5],
        [1, 4, 5],
        [2, 4, 5],
        [3, 4, 5],
        [0, 5, -1],
        [0, 5, 4],
    ], dtype=np.int64)
    scores = np.array([
        [10.0, 9.0, 5.0],
        [10.0, 8.0, 5.0],
        [6.0, 5.5, 3.0],
        [1.0, -50.0, -60.0],
        [2.0, 1.9, 1.0],
        [10.0, 9.5, -np.inf],
        [10.0, 9.0, -50.0],
    ], dtype=np.float32)
    return runtime, ids, scores


def test_re_examine_mixed_batch_outcomes_and_bookkeeping():
    runtime, ids, scores = _build_batch()
    final, stats = re_examine(ids, scores, runtime)

    expected_final = [4, 0, 4, 2, 3, 0, 0]
    assert final.tolist() == expected_final

    assert stats["group_a"] == {
        "n_examined": 3, "n_moved": 1,
        "n_blocked_by_proximity": 1, "n_no_cand": 1,
    }
    assert stats["group_b"] == {
        "n_examined": 1, "n_moved": 1,
        "n_blocked_by_proximity": 0, "n_no_cand": 0,
    }
    # bookkeeping identity
    for name, s in stats.items():
        assert s["n_examined"] == s["n_moved"] + s["n_blocked_by_proximity"] + s["n_no_cand"]


def test_re_examine_gap_equal_to_tau_is_not_examined():
    runtime, ids, scores = _build_batch()
    final, _stats = re_examine(ids, scores, runtime)
    # row1: gap == tau_a[0] == 2.0 exactly -> unchanged (strict < required).
    assert final[1] == 0


def test_re_examine_predictions_outside_both_groups_never_touched():
    runtime, ids, scores = _build_batch()
    final, _stats = re_examine(ids, scores, runtime)
    assert final[3] == 2  # row3: lang2 is in neither group


def test_re_examine_excluded_language_never_re_examined():
    runtime, ids, scores = _build_batch()
    final, _stats = re_examine(ids, scores, runtime)
    assert final[4] == 3  # row4: lang3's tau is -inf despite a small gap


def test_re_examine_requires_float32_scores():
    runtime, ids, scores = _build_batch()
    with pytest.raises(UnilidCalibrationError, match="float32"):
        re_examine(ids, scores.astype(np.float64), runtime)


def test_re_examine_dtype_policy_float64_tau_comparison_wins_over_float32_rounding():
    """[R1] The gate compares the float32 gap against tau in float64 (numpy's
    elementwise comparison between a float32 array and a float64 array
    upcasts the float32 side losslessly, so the float64 tau value is used at
    full precision). Constructs tau just above the exact float32 gap value by
    less than half the float32 rounding unit at that magnitude, so that if
    the comparison were instead done by first rounding tau down to float32,
    it would round back to exactly the gap value and the strict '<' would
    evaluate False (not examined). The reference/implementation semantics
    (float64 comparison) must evaluate True (examined) and thus perform the
    walk. Values verified with numpy directly (IEEE-754 float32 spacing at
    magnitude 5.0 is 2**-21 ~= 4.7683716e-07; a quarter of that added to 5.0
    in float64 still rounds to exactly float32(5.0))."""
    gap_value = np.float32(5.0)
    spacing = float(np.spacing(np.float32(5.0)))
    tau64 = 5.0 + spacing / 4.0
    assert np.float32(tau64) == gap_value  # tau rounds back to the gap value in fp32
    # Note: comparing bare Python-float/np.float32 *scalars* follows NumPy's
    # weak-scalar promotion (NEP 50) and would downcast tau64 to float32
    # here, which is NOT what the implementation does (it compares a
    # float32 *array* against values indexed from an explicitly float64
    # *array*, which is strong promotion to float64). Wrap tau64 as an
    # explicit np.float64 to match that real comparison's semantics.
    assert gap_value < np.float64(tau64)  # strictly less in fp64

    N = np.array([0, 200000], dtype=np.int64)
    in_a = np.array([True, False])
    in_b = np.array([False, False])
    tau_a = np.array([tau64, -np.inf], dtype=np.float64)
    tau_b = np.array([-np.inf, -np.inf], dtype=np.float64)
    cal = _minimal_calibration(topk=2)
    runtime = CalibrationRuntime(calibration=cal, N=N, in_a=in_a, in_b=in_b,
                                 tau_a=tau_a, tau_b=tau_b)

    ids = np.array([[0, 1]], dtype=np.int64)
    # top1 - rank2 = 10.0 - 5.0 = 5.0 exactly (both exactly representable,
    # Sterbenz's lemma guarantees the float32 subtraction is exact).
    scores = np.array([[10.0, 5.0]], dtype=np.float32)

    final, stats = re_examine(ids, scores, runtime)
    assert final[0] == 1  # moved: proves the comparison ran in float64
    assert stats["group_a"]["n_moved"] == 1
