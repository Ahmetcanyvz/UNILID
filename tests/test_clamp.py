"""Unit tests for unilid.calibration.apply_unseen_token_constant, the port of
analysis/floor_equalization.py::build_equalized_weights.

Reference semantics (both the ported function and its reference read
identically):

    out = np.array(W, dtype=np.float32)
    for each row i:
        floor = row.min()
        if floor > target:
            row[row == floor] = target      # only the exact-min plateau moves
    raise if any output entry is non-finite

So the clamp only ever pushes a row's minimum-value plateau DOWN to `target`
when that plateau currently sits above `target` (i.e. target is more negative
than the plateau); rows whose minimum is already at or below `target` are
left completely untouched, and entries that are not part of the plateau are
never touched regardless of their own value.
"""
import numpy as np
import pytest

from unilid.calibration import UnilidCalibrationError, apply_unseen_token_constant

TARGET = -21.0


def test_one_sided_clamp_moves_only_the_min_plateau_and_counts_modified_rows():
    # Row 0: floor = -10.0 (single occurrence, index 2), which is > TARGET
    # (-10.0 > -21.0) -> triggers. Only that entry moves; others untouched.
    row0 = [-5.0, -3.0, -10.0, -2.0]
    # Row 1: floor = -25.0, which is <= TARGET (-25.0 > -21.0 is False) ->
    # left entirely untouched.
    row1 = [-25.0, -3.0, -1.0, -5.0]
    W = np.array([row0, row1], dtype=np.float32)

    out, n_mod = apply_unseen_token_constant(W, TARGET)

    assert n_mod == 1
    np.testing.assert_array_equal(
        out[0], np.array([-5.0, -3.0, -21.0, -2.0], dtype=np.float32))
    np.testing.assert_array_equal(out[1], np.array(row1, dtype=np.float32))


def test_row_with_min_equal_to_or_below_target_is_bit_identical():
    # Row 0: floor == TARGET exactly -> floor > target is False -> untouched.
    row0 = [-21.0, -5.0, -1.0, -3.0]
    # Row 1: floor < TARGET -> also untouched.
    row1 = [-30.0, -5.0, -1.0, -3.0]
    W = np.array([row0, row1], dtype=np.float32)

    out, n_mod = apply_unseen_token_constant(W, TARGET)

    assert n_mod == 0
    np.testing.assert_array_equal(out, W)


def test_multi_entry_plateau_all_move_and_a_nonplateau_entry_at_target_stays():
    # Row 0: three-way tied plateau at -10.0 (> TARGET) -> all three move to
    # TARGET; the fourth entry (-3.0), not part of the plateau, is untouched.
    row0 = [-10.0, -10.0, -3.0, -10.0]
    # Row 1: floor is -25.0 (<= TARGET) so the WHOLE row is untouched, even
    # though it contains a non-floor entry that already equals TARGET exactly
    # (index 1 == -21.0). This checks the mask is row[row == floor], not
    # row[row == target]: an entry equal to target that is not the row's own
    # floor must never be touched.
    row1 = [-25.0, -21.0, -2.0, -1.0]
    W = np.array([row0, row1], dtype=np.float32)

    out, n_mod = apply_unseen_token_constant(W, TARGET)

    assert n_mod == 1
    np.testing.assert_array_equal(
        out[0], np.array([-21.0, -21.0, -3.0, -21.0], dtype=np.float32))
    np.testing.assert_array_equal(out[1], np.array(row1, dtype=np.float32))


def test_output_is_fp32_and_input_is_not_mutated():
    W64 = np.array([[-5.0, -3.0, -10.0, -2.0]], dtype=np.float64)
    W64_copy = W64.copy()

    out, n_mod = apply_unseen_token_constant(W64, TARGET)

    assert out.dtype == np.float32
    assert n_mod == 1
    np.testing.assert_array_equal(
        out[0], np.array([-5.0, -3.0, -21.0, -2.0], dtype=np.float32))
    # Input array must be untouched (compare against the copy taken before
    # the call, not against a freshly-derived value).
    np.testing.assert_array_equal(W64, W64_copy)


def test_non_finite_input_raises():
    # A row's true minimum is -inf, so `floor > target` is always False for
    # any finite target (clamp never triggers on it); the -inf entry then
    # survives into the output and the post-check `isfinite(out).all()`
    # fails, which is the intended failure mode (not a mistaken clamp of
    # -inf to the target).
    W = np.array([[-np.inf, -1.0, -2.0, -1.5]], dtype=np.float32)
    with pytest.raises(UnilidCalibrationError):
        apply_unseen_token_constant(W, TARGET)
