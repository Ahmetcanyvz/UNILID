"""End-to-end tests of unilid.model_io.UnilidModel's calibrated inference
path, against a tiny synthetic model built with the real tokenizers Rust
extension (tokenizers.models.Unigram with set_weight_sets /
top_k_of_cached_weight_sets_batch / best_of_cached_weight_sets_batch /
tokens_of_cached_weight_set_batch).

Design of the synthetic weights (vocab: <unk>=0, a=1, b=2, ab=3; row[3] is
set far below the clamp target -21.0 for every language, so the clamp is a
documented no-op here and the DP always prefers the "a"+"b" segmentation
over the single "ab" token for the text "ab" used throughout -- verified
directly against the Rust extension's own best_of/top_k primitives, which
are dependency behavior, not the calibration logic under test):

    X: [0.0, -5.0,  -5.0, -100.0]  -> score("ab") = -10.0    (base argmax)
    Y: [0.0, -5.25, -5.0, -100.0]  -> score("ab") = -10.25   (walk target)
    Z: [0.0, -50.0, -50.0, -100.0] -> score("ab") = -100.0   (never involved)

X is a group-A member (N=500 < head_n) with tau=5.0; the top1/top2 gap for
"ab" is 0.25 (X vs Y), well under tau, so the gate always triggers; Y has
N=200000 >= replacement_min_n=100000 and is within the proximity_bound
(0.25 <= 21.0), so the walk always accepts it as the replacement.
"""
import numpy as np
import pytest

from unilid.calibration import Calibration, TauRow, UnilidCalibrationError
from unilid.model_io import UnilidModel, write_unilid

LANGS = ["X", "Y", "Z"]
WEIGHTS = np.array([
    [0.0, -5.0, -5.0, -100.0],
    [0.0, -5.25, -5.0, -100.0],
    [0.0, -50.0, -50.0, -100.0],
], dtype=np.float32)
TEXT = "ab"


def _calibration() -> Calibration:
    return Calibration(
        unseen_token_constant=-21.0, head_n=18000, replacement_min_n=100000,
        proximity_bound=21.0, topk=5, margin_q=5.0, group_b_percentile=5.0,
        calib_max=2000, min_calib_lines=200, calib_seed=0,
        group_a={"X": TauRow(tau=5.0, excluded=False, cause="",
                             n_scoreable=1000, n_self_won=900)},
        group_b={},
        train_counts={"X": 500, "Y": 200000, "Z": 1000000},
        provenance={})


@pytest.fixture
def v1_path(tmp_path, tiny_base_tok_json):
    path = tmp_path / "v1.unilid"
    write_unilid(path, tiny_base_tok_json.encode("utf-8"), LANGS, WEIGHTS)
    return path


@pytest.fixture
def v2_path(tmp_path, tiny_base_tok_json):
    path = tmp_path / "v2.unilid"
    write_unilid(path, tiny_base_tok_json.encode("utf-8"), LANGS, WEIGHTS,
                 calibration=_calibration())
    return path


def test_calibrated_default_moves_prediction_to_the_walk_target(v2_path):
    model = UnilidModel(v2_path)
    assert model.calibrated is True
    lang, tokens, score = model.predict(TEXT)
    assert lang == "Y"
    assert tokens == ["a", "b"]
    assert score == pytest.approx(-10.25)

    # The calibrated score must equal the segmentation pass's own score under
    # the final language, verified by calling the same Rust primitive
    # directly (tokens_of_cached_weight_set_batch), independent of the
    # UnilidModel code path.
    direct_tokens, direct_score = model.model.tokens_of_cached_weight_set_batch(
        [TEXT], [model.langs.index("Y")])[0]
    assert direct_tokens == tokens
    assert direct_score == pytest.approx(score)


def test_calibrated_false_reproduces_the_base_argmax(v2_path):
    model = UnilidModel(v2_path, calibrated=False)
    assert model.calibrated is False
    lang, tokens, score = model.predict(TEXT)
    assert lang == "X"
    assert tokens == ["a", "b"]
    assert score == pytest.approx(-10.0)


def test_v1_file_with_calibrated_default_raises_naming_file_and_remedies(v1_path):
    with pytest.raises(UnilidCalibrationError) as exc:
        UnilidModel(v1_path)
    msg = str(exc.value)
    assert str(v1_path) in msg
    assert "calibrated=False" in msg
    assert "calibration=" in msg


def test_v1_file_with_calibrated_false_loads_fine(v1_path):
    model = UnilidModel(v1_path, calibrated=False)
    assert model.calibrated is False
    assert model.calibration is None
    lang, tokens, score = model.predict(TEXT)
    assert lang == "X"


def test_v2_file_plus_calibration_argument_errors_already_bundled(v2_path, tmp_path):
    sidecar = tmp_path / "sidecar_calibration.json"
    _calibration().to_json_file(sidecar)
    with pytest.raises(UnilidCalibrationError, match="already bundles a calibration"):
        UnilidModel(v2_path, calibration=str(sidecar))


def test_predict_normalized_errors_when_calibrated(v2_path):
    model = UnilidModel(v2_path)
    with pytest.raises(UnilidCalibrationError):
        model.predict_normalized(TEXT)
    with pytest.raises(UnilidCalibrationError):
        model.predict_normalized_batch([TEXT])


def test_predict_normalized_works_when_not_calibrated(v2_path):
    model = UnilidModel(v2_path, calibrated=False)
    # score / n_tokens^alpha with n_tokens=2 (["a","b"]) and alpha=1.0, so
    # exactly half the base score (-10.0 / 2 = -5.0, exact in float32).
    lang, tokens, score = model.predict_normalized(TEXT, alpha=1.0)
    assert lang == "X"
    assert tokens == ["a", "b"]
    assert score == pytest.approx(-5.0)

    batch = model.predict_normalized_batch([TEXT], alpha=1.0)
    assert batch == [("X", ["a", "b"], pytest.approx(-5.0))]


def test_empty_after_preprocess_text_in_calibrated_batch(v2_path):
    model = UnilidModel(v2_path)
    results = model.predict_batch([TEXT, "", TEXT])
    assert results[1] == (None, [], float("-inf"))
    assert results[0][0] == "Y"
    assert results[2][0] == "Y"
    assert results[0][2] == pytest.approx(-10.25)
    assert results[2][2] == pytest.approx(-10.25)
