"""Unit tests for unilid.calibration.Calibration / TauRow: JSON round trip,
schema validation (section 4.2 of OPEN_SOURCE_DESIGN.md), and the
model-consistency checks in Calibration.runtime_for (the port of
analysis/external_bench_eval.py::_load_gate_thresholds).
"""
import copy

import numpy as np
import pytest

from unilid.calibration import Calibration, TauRow, UnilidCalibrationError

# The calibration artifact's constants block, at the values documented in
# OPEN_SOURCE_DESIGN.md section 4.2 (the released model's own constants).
CALIB_CONSTS = dict(
    unseen_token_constant=-21.0,
    head_n=18000,
    replacement_min_n=100000,
    proximity_bound=21.0,
    topk=5,
    margin_q=5.0,
    group_b_percentile=5.0,
    calib_max=2000,
    min_calib_lines=200,
    calib_seed=0,
)
CONST_NAMES = list(CALIB_CONSTS)


def _cal(group_a=None, group_b=None, train_counts=None, provenance=None,
        **const_overrides):
    consts = dict(CALIB_CONSTS, **const_overrides)
    return Calibration(
        group_a=group_a or {}, group_b=group_b or {},
        train_counts=train_counts or {}, provenance=provenance or {},
        **consts)


# --------------------------------------------------------------- round trip

def _sample_calibration() -> Calibration:
    group_a = {
        "aaa": TauRow(tau=7.310638256072998, excluded=False, cause="",
                      n_scoreable=1335, n_self_won=1316),
        "ddd": TauRow(tau=float("-inf"), excluded=True,
                      cause="low_calibration", n_scoreable=50, n_self_won=10),
    }
    group_b = {
        "bbb": TauRow(tau=0.5, excluded=False, cause="",
                      n_scoreable=2000, n_self_won=1900),
    }
    train_counts = {"aaa": 500, "bbb": 25000, "ccc": 500000, "ddd": 100}
    provenance = {"source_model_sha256": "deadbeef", "derived": "unit test"}
    return _cal(group_a, group_b, train_counts, provenance)


def test_json_round_trip_preserves_rows_counts_and_constants():
    cal = _sample_calibration()
    raw = cal.to_json_bytes()
    cal2 = Calibration.from_json_bytes(raw)

    assert cal2.group_a == cal.group_a
    assert cal2.group_b == cal.group_b
    assert cal2.train_counts == cal.train_counts
    assert cal2.provenance == cal.provenance
    for name in CONST_NAMES:
        assert getattr(cal2, name) == getattr(cal, name)


def test_excluded_row_serializes_tau_as_null_and_reparses_to_negative_inf():
    cal = _sample_calibration()
    d = cal.to_json_dict()
    ddd_json = d["group_a"]["thresholds"]["ddd"]
    assert ddd_json["tau"] is None
    assert ddd_json["excluded"] is True
    assert ddd_json["cause"] == "low_calibration"

    cal2 = Calibration.from_json_bytes(cal.to_json_bytes())
    row = cal2.group_a["ddd"]
    assert row.tau == float("-inf")
    assert row.excluded is True
    assert row.cause == "low_calibration"


# ------------------------------------------------------------- JSON rejects

def _mutate(d, path, value=None, delete=False):
    """Deep-copy ``d`` and set (or delete) the value at nested-key ``path``."""
    out = copy.deepcopy(d)
    node = out
    for key in path[:-1]:
        node = node[key]
    if delete:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


def test_reject_excluded_false_with_tau_null():
    cal = _sample_calibration()
    d = cal.to_json_dict()
    mutated = _mutate(d, ["group_a", "thresholds", "aaa", "tau"], None)
    assert mutated["group_a"]["thresholds"]["aaa"]["excluded"] is False
    with pytest.raises(UnilidCalibrationError, match="tau is null but excluded is false"):
        Calibration.from_json_dict(mutated)


def test_reject_excluded_true_with_numeric_tau():
    cal = _sample_calibration()
    d = cal.to_json_dict()
    mutated = _mutate(d, ["group_a", "thresholds", "ddd", "tau"], 3.0)
    assert mutated["group_a"]["thresholds"]["ddd"]["excluded"] is True
    with pytest.raises(UnilidCalibrationError, match="excluded rows store tau as null"):
        Calibration.from_json_dict(mutated)


def test_reject_non_bool_excluded():
    cal = _sample_calibration()
    d = cal.to_json_dict()
    mutated = _mutate(d, ["group_a", "thresholds", "aaa", "excluded"], 1)
    with pytest.raises(UnilidCalibrationError, match="must be a JSON boolean"):
        Calibration.from_json_dict(mutated)


@pytest.mark.parametrize("name", CONST_NAMES)
def test_reject_each_missing_constant_individually(name):
    cal = _sample_calibration()
    d = cal.to_json_dict()
    mutated = _mutate(d, ["constants", name], delete=True)
    with pytest.raises(UnilidCalibrationError) as exc:
        Calibration.from_json_dict(mutated)
    assert name in str(exc.value)


@pytest.mark.parametrize("field", ["tau", "excluded", "n_scoreable", "n_self_won"])
def test_reject_each_missing_row_field_individually(field):
    cal = _sample_calibration()
    d = cal.to_json_dict()
    mutated = _mutate(d, ["group_a", "thresholds", "aaa", field], delete=True)
    with pytest.raises(UnilidCalibrationError) as exc:
        Calibration.from_json_dict(mutated)
    assert field in str(exc.value)


def test_reject_format_version_mismatch():
    cal = _sample_calibration()
    d = cal.to_json_dict()
    mutated = _mutate(d, ["format_version"], 2)
    with pytest.raises(UnilidCalibrationError, match="format_version"):
        Calibration.from_json_dict(mutated)


def test_reject_group_a_group_b_overlap():
    cal = _sample_calibration()
    d = cal.to_json_dict()
    # Put the same language ("aaa") in both groups with valid rows.
    mutated = _mutate(
        d, ["group_b", "thresholds", "aaa"],
        {"tau": 1.0, "excluded": False, "cause": "", "n_scoreable": 10,
         "n_self_won": 5})
    with pytest.raises(UnilidCalibrationError, match="overlap"):
        Calibration.from_json_dict(mutated)


def test_reject_negative_train_count():
    cal = _sample_calibration()
    d = cal.to_json_dict()
    mutated = _mutate(d, ["train_counts", "aaa"], -5)
    with pytest.raises(UnilidCalibrationError, match="non-negative integer"):
        Calibration.from_json_dict(mutated)


def test_reject_non_int_train_count():
    cal = _sample_calibration()
    d = cal.to_json_dict()
    mutated = _mutate(d, ["train_counts", "aaa"], 500.5)
    with pytest.raises(UnilidCalibrationError, match="non-negative integer"):
        Calibration.from_json_dict(mutated)


# ------------------------------------------------------------ runtime_for()

HEAD_N = 1000


def _valid_langs_and_cal():
    langs = ["aaa", "bbb", "ccc"]
    train_counts = {"aaa": 500, "bbb": 2000, "ccc": 100}
    # Group A must equal {langs with N < head_n} = {"aaa", "ccc"}.
    group_a = {
        "aaa": TauRow(tau=1.0, excluded=False, cause="", n_scoreable=10, n_self_won=5),
        "ccc": TauRow(tau=1.0, excluded=False, cause="", n_scoreable=10, n_self_won=5),
    }
    cal = _cal(group_a=group_a, train_counts=train_counts, head_n=HEAD_N)
    return langs, cal


def test_runtime_for_valid_case_builds_expected_arrays():
    langs, cal = _valid_langs_and_cal()
    runtime = cal.runtime_for(langs)

    assert runtime.tau_a.dtype == np.float64
    assert runtime.tau_b.dtype == np.float64
    # aaa (idx 0) and ccc (idx 2) are group A -> tau_a set, tau_b default -inf.
    assert runtime.tau_a[0] == 1.0
    assert runtime.tau_a[2] == 1.0
    assert runtime.tau_a[1] == float("-inf")  # bbb: outside group A
    assert (runtime.tau_b == float("-inf")).all()  # no group B members
    assert runtime.in_a.tolist() == [True, False, True]
    assert runtime.in_b.tolist() == [False, False, False]


def test_runtime_for_group_a_set_mismatch_shows_both_directions():
    langs = ["aaa", "bbb", "ccc"]
    train_counts = {"aaa": 500, "bbb": 2000, "ccc": 100}
    # Expected group A is {"aaa", "ccc"}; this artifact wrongly includes
    # "bbb" (N=2000 >= head_n) and wrongly omits "ccc" (N=100 < head_n).
    group_a = {
        "aaa": TauRow(tau=1.0, excluded=False, cause="", n_scoreable=10, n_self_won=5),
        "bbb": TauRow(tau=1.0, excluded=False, cause="", n_scoreable=10, n_self_won=5),
    }
    cal = _cal(group_a=group_a, train_counts=train_counts, head_n=HEAD_N)
    with pytest.raises(UnilidCalibrationError) as exc:
        cal.runtime_for(langs)
    msg = str(exc.value)
    assert "bbb" in msg  # unexpected member
    assert "ccc" in msg  # missing member


def test_runtime_for_group_b_member_missing_from_model():
    langs, cal = _valid_langs_and_cal()
    group_b = {"zzz": TauRow(tau=1.0, excluded=False, cause="",
                             n_scoreable=10, n_self_won=5)}
    cal2 = _cal(group_a=cal.group_a, group_b=group_b,
               train_counts=cal.train_counts, head_n=HEAD_N)
    with pytest.raises(UnilidCalibrationError, match="not in the model"):
        cal2.runtime_for(langs)


def test_runtime_for_group_b_member_with_n_below_head_n():
    langs, cal = _valid_langs_and_cal()
    # "aaa" has N=500 < head_n=1000, so it cannot legally be a group B member.
    group_b = {"aaa": TauRow(tau=1.0, excluded=False, cause="",
                             n_scoreable=10, n_self_won=5)}
    cal2 = _cal(group_a=cal.group_a, group_b=group_b,
               train_counts=cal.train_counts, head_n=HEAD_N)
    with pytest.raises(UnilidCalibrationError, match="must be disjoint"):
        cal2.runtime_for(langs)


def test_runtime_for_model_lang_missing_from_train_counts():
    langs = ["aaa", "bbb", "ccc", "ddd"]  # "ddd" absent from train_counts
    _, cal = _valid_langs_and_cal()
    with pytest.raises(UnilidCalibrationError, match="missing from"):
        cal.runtime_for(langs)


def test_runtime_for_duplicate_model_lang():
    langs = ["aaa", "bbb", "aaa"]
    _, cal = _valid_langs_and_cal()
    with pytest.raises(UnilidCalibrationError, match="twice"):
        cal.runtime_for(langs)
