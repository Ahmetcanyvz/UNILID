"""Unit tests for unilid.calibration.estimate_tau, the port of the threshold
recipe of record in analysis/solo_gates.py (tau_floor21_gate.csv's own
generation code), against a tiny synthetic model scored with the real Rust
extension.

Vocab <unk>=0, a=1, b=2, ab=3; row[3] ("ab" as a single token) is far below
the clamp target -21.0 for every language, so the clamp is a no-op and the
DP always prefers "a"+"b" over the single "ab" token for the text "ab":

    A: [0.0, -2.0,  -8.0,  -100.0]
    B: [0.0, -4.0,  -9.0,  -100.0]
    C: [0.0, -50.0, -49.0, -100.0]

For the text "a" (a single character, only tokenizable as the one-token
segmentation ["a"]) the score is just row[1]: A=-2.0, B=-4.0, C=-50.0 -> A
wins with margin 2.0. For "ab" (DP picks "a"+"b" since row[3]=-100 always
loses): A=-10.0, B=-13.0, C=-99.0 -> A wins with margin 3.0. A wins both
texts on every repetition, so calibration lines built from these two texts
give a fully hand-computable, non-degenerate margin distribution.

head_n is set to 1000 here (not the production 18000) purely for clean
arithmetic; the Calibration is entirely synthetic to this test file.
"""
import numpy as np
import pytest

from unilid.calibration import (
    CAUSE_LOW_CALIBRATION,
    CAUSE_ZERO_STRENGTH,
    Calibration,
    TauRow,
    UnilidCalibrationError,
    estimate_tau,
)
from unilid.model_io import UnilidModel, write_unilid

LANGS = ["A", "B", "C"]
WEIGHTS = np.array([
    [0.0, -2.0, -8.0, -100.0],
    [0.0, -4.0, -9.0, -100.0],
    [0.0, -50.0, -49.0, -100.0],
], dtype=np.float32)
HEAD_N = 1000


def _calibration() -> Calibration:
    return Calibration(
        unseen_token_constant=-21.0, head_n=HEAD_N, replacement_min_n=100000,
        proximity_bound=21.0, topk=5, margin_q=5.0, group_b_percentile=5.0,
        calib_max=2000, min_calib_lines=200, calib_seed=0,
        group_a={"A": TauRow(tau=1.0, excluded=False, cause="",
                             n_scoreable=10, n_self_won=10)},  # placeholder
        group_b={},
        train_counts={"A": 300, "B": 50000, "C": 50000},
        provenance={})


@pytest.fixture
def v2_path(tmp_path, tiny_base_tok_json):
    path = tmp_path / "model.unilid"
    write_unilid(path, tiny_base_tok_json.encode("utf-8"), LANGS, WEIGHTS,
                calibration=_calibration())
    return path


@pytest.fixture
def calibrated_model(v2_path):
    return UnilidModel(v2_path)


def test_tau_matches_hand_computed_percentile_of_the_margins(calibrated_model):
    lines = ["a"] * 150 + ["ab"] * 150
    row = estimate_tau(calibrated_model, "A", lines, n_l=300)

    assert row.excluded is False
    assert row.cause == ""
    assert row.n_scoreable == 300
    assert row.n_self_won == 300
    # q_L = margin_q * (1 - min(300, 1000)/1000) = 5 * 0.7 = 3.5.
    # np.percentile(q=3.5) of the 300-element array [2.0]*150 + [3.0]*150:
    # linear-interpolation index = (3.5/100)*(300-1) ~= 10.465, which falls
    # inside the first 150 (sorted) entries, all equal to 2.0.
    assert row.tau == pytest.approx(2.0)


def test_fewer_than_min_calib_lines_wins_excludes_low_calibration(calibrated_model):
    lines = ["a"] * 50  # all winning, but only 50 < min_calib_lines=200
    row = estimate_tau(calibrated_model, "A", lines, n_l=300)

    assert row.excluded is True
    assert row.cause == CAUSE_LOW_CALIBRATION
    assert row.tau == float("-inf")
    assert row.n_scoreable == 50
    assert row.n_self_won == 50


def test_q_l_zero_at_n_equal_head_n_excludes_zero_strength(calibrated_model):
    lines = ["a"] * 150 + ["ab"] * 150  # 300 wins, well above min_calib_lines
    # N == head_n -> min(N, head_n)/head_n == 1 -> q_L == 0 exactly.
    row = estimate_tau(calibrated_model, "A", lines, n_l=HEAD_N)

    assert row.excluded is True
    assert row.cause == CAUSE_ZERO_STRENGTH
    assert row.tau == float("-inf")
    assert row.n_scoreable == 300
    assert row.n_self_won == 300


def test_low_calibration_takes_precedence_over_zero_strength(calibrated_model):
    lines = ["a"] * 50  # < min_calib_lines=200 -> low_calibration
    # N == head_n -> also zero_strength; low_calibration must win the cause.
    row = estimate_tau(calibrated_model, "A", lines, n_l=HEAD_N)

    assert row.excluded is True
    assert row.cause == CAUSE_LOW_CALIBRATION
    assert row.tau == float("-inf")


def test_estimate_tau_requires_a_calibrated_loaded_model(v2_path):
    base_model = UnilidModel(v2_path, calibrated=False)
    with pytest.raises(UnilidCalibrationError, match="calibrated"):
        estimate_tau(base_model, "A", [], 100)
