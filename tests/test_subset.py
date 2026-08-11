"""Language subsetting: load-time (languages=) and the unilid-calibrate subset
command. Scores below are hand-derived over the shared TINY_VOCAB
(<unk>=0, a=1, b=2, ab=3): score("ab") under a row w is
max(w[a] + w[b], w[ab]).

Model: four languages with score("ab") of aaa=-0.5, bbb=-1.0, ccc=-2.5,
ddd=-9.0. aaa is group A (train count 10 < head_n=100) with tau=1.0; bbb,
ccc, ddd have counts 5000 >= replacement_min_n=1000, so all three are
replacement candidates.

Full-model behavior on "ab": top-1 aaa, margin 0.5 < tau -> re-examined,
moved to bbb. The subset tests pin the carried-threshold consequence: a
smaller candidate set can only raise the margin, so re-examination fires at
most as often as calibrated.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from unilid.calibration import Calibration, TauRow, UnilidCalibrationError
from unilid.calibrate_cli import main as calibrate_main
from unilid.model_io import (
    UnilidModel,
    load_unilid_raw,
    read_calibration,
    subset_rows,
    write_unilid,
)

LANGS = ["aaa", "bbb", "ccc", "ddd"]
WEIGHTS = np.array([
    # <unk>,  a,     b,     ab      -> score("ab")
    [0.0,   -1.0,  -1.0,  -0.5],   # aaa: -0.5
    [0.0,   -1.5,  -1.5,  -1.0],   # bbb: -1.0
    [0.0,   -3.0,  -3.0,  -2.5],   # ccc: -2.5
    [0.0,   -9.0,  -9.0,  -9.0],   # ddd: -9.0
], dtype=np.float32)
CONSTANTS = {
    "unseen_token_constant": -21.0,
    "head_n": 100,
    "replacement_min_n": 1000,
    "proximity_bound": 21.0,
    "topk": 4,
    "margin_q": 5.0,
    "group_b_percentile": 5.0,
    "calib_max": 2000,
    "min_calib_lines": 2,
    "calib_seed": 0,
}
COUNTS = {"aaa": 10, "bbb": 5000, "ccc": 5000, "ddd": 5000}


def _calibration() -> Calibration:
    return Calibration(
        group_a={"aaa": TauRow(tau=1.0, excluded=False, cause="",
                               n_scoreable=10, n_self_won=10)},
        group_b={}, train_counts=dict(COUNTS), provenance={}, **CONSTANTS)


@pytest.fixture
def v2_model(tmp_path, tiny_base_tok_json):
    path = tmp_path / "full.unilid"
    write_unilid(path, tiny_base_tok_json.encode("utf-8"), LANGS, WEIGHTS,
                 _calibration())
    return path


@pytest.fixture
def v1_model(tmp_path, tiny_base_tok_json):
    path = tmp_path / "base.unilid"
    write_unilid(path, tiny_base_tok_json.encode("utf-8"), LANGS, WEIGHTS)
    return path


# ------------------------------------------------------------- subset_rows

def test_subset_rows_model_order_and_values():
    sub_w, sub_l = subset_rows(WEIGHTS, LANGS, ["ddd", "bbb"])
    assert sub_l == ["bbb", "ddd"]  # model order, not argument order
    assert np.array_equal(sub_w, WEIGHTS[[1, 3]])


@pytest.mark.parametrize("languages,message", [
    ([], "non-empty"),
    (["aaa", "aaa"], "duplicate"),
    (["aaa", "zzz"], "not in the model"),
])
def test_subset_rows_rejects(languages, message):
    with pytest.raises(ValueError, match=message):
        subset_rows(WEIGHTS, LANGS, languages)


# ------------------------------------------------- load-time subsetting

def test_base_subset_is_exact_argmax_restriction(v1_model):
    full = UnilidModel(v1_model, calibrated=False)
    assert full.predict("ab")[0] == "aaa"
    sub = UnilidModel(v1_model, calibrated=False, languages=["bbb", "ccc"])
    lang, _tokens, score = sub.predict("ab")
    # the argmax over the included rows, with the same score the full model
    # assigns that language
    assert lang == "bbb"
    assert score == pytest.approx(-1.0)


def test_calibrated_full_model_reexamines(v2_model):
    full = UnilidModel(v2_model, calibrated=True)
    lang, _t, _s = full.predict("ab")
    assert lang == "bbb"  # margin 0.5 < tau 1.0 -> walked to bbb
    assert full.last_reexamination_stats["group_a"]["n_moved"] == 1


def test_carried_threshold_fires_at_most_as_often(v2_model):
    # Removing the runner-up raises the margin above tau: no re-examination.
    sub = UnilidModel(v2_model, calibrated=True,
                      languages=["aaa", "ccc", "ddd"])
    lang, _t, _s = sub.predict("ab")
    assert lang == "aaa"  # margin now 2.0 > tau 1.0
    assert sub.last_reexamination_stats["group_a"]["n_examined"] == 0

    # Removing an unrelated language leaves the margin unchanged: still
    # re-examined and moved.
    sub2 = UnilidModel(v2_model, calibrated=True,
                       languages=["aaa", "bbb", "ccc"])
    assert sub2.predict("ab")[0] == "bbb"
    assert sub2.last_reexamination_stats["group_a"]["n_moved"] == 1


def test_subset_calibration_filtered_and_flagged(v2_model):
    sub = UnilidModel(v2_model, calibrated=True, languages=["aaa", "bbb"])
    cal = sub.calibration
    assert set(cal.group_a) == {"aaa"}
    assert set(cal.train_counts) == {"aaa", "bbb"}
    assert "carried" in cal.provenance["subset"]["thresholds"]


def test_subset_without_group_a_languages(v2_model):
    sub = UnilidModel(v2_model, calibrated=True, languages=["bbb", "ccc"])
    assert sub.calibration.group_a == {}
    assert sub.predict("ab")[0] == "bbb"  # nothing to re-examine


def test_v1_subset_with_default_calibrated_still_errors(v1_model):
    with pytest.raises(UnilidCalibrationError, match="no.*calibration"):
        UnilidModel(v1_model, languages=["aaa", "bbb"])


# ------------------------------------------------------- the CLI command

def test_cli_subset_v2_carried(v2_model, tmp_path):
    out = tmp_path / "sub.unilid"
    calibrate_main(["subset", str(v2_model), "-o", str(out),
                    "--langs", "aaa,ccc,ddd"])
    _tb, w, langs = load_unilid_raw(out)
    assert langs == ["aaa", "ccc", "ddd"]
    assert np.array_equal(np.asarray(w), WEIGHTS[[0, 2, 3]])
    cal = read_calibration(out)
    assert set(cal.group_a) == {"aaa"}
    assert cal.group_a["aaa"].tau == 1.0  # carried, not re-estimated
    assert "carried" in cal.provenance["subset"]["thresholds"]
    model = UnilidModel(out, calibrated=True)
    assert model.predict("ab")[0] == "aaa"  # margin 2.0 > carried tau 1.0


def test_cli_subset_langs_file(v2_model, tmp_path):
    langs_file = tmp_path / "keep.txt"
    langs_file.write_text("bbb\nddd\n")
    out = tmp_path / "sub2.unilid"
    calibrate_main(["subset", str(v2_model), "-o", str(out),
                    "--langs-file", str(langs_file)])
    _tb, _w, langs = load_unilid_raw(out)
    assert langs == ["bbb", "ddd"]


def test_cli_subset_v1(v1_model, tmp_path):
    out = tmp_path / "subv1.unilid"
    calibrate_main(["subset", str(v1_model), "-o", str(out),
                    "--langs", "aaa,bbb"])
    assert read_calibration(out) is None
    model = UnilidModel(out, calibrated=False)
    assert model.predict("ab")[0] == "aaa"


def test_cli_subset_v1_recalibrate_errors(v1_model, tmp_path):
    with pytest.raises(UnilidCalibrationError, match="version-1"):
        calibrate_main(["subset", str(v1_model), "-o",
                        str(tmp_path / "x.unilid"), "--langs", "aaa,bbb",
                        "--recalibrate", str(tmp_path)])


def test_cli_subset_recalibrate(v2_model, tmp_path):
    # Corpus for the one group A language: 10 lines "ab". In the subset
    # (aaa, ccc, ddd) every line's margin is -0.5 - (-2.5) = 2.0, and
    # q_L = 5 * (1 - 10/100) = 4.5, so the re-estimated tau is exactly 2.0.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "aaa_train.txt").write_text("ab\n" * 10)
    out = tmp_path / "recal.unilid"
    calibrate_main(["subset", str(v2_model), "-o", str(out),
                    "--langs", "aaa,ccc,ddd",
                    "--recalibrate", str(corpus)])
    cal = read_calibration(out)
    row = cal.group_a["aaa"]
    assert not row.excluded
    assert row.tau == pytest.approx(2.0)
    assert row.n_scoreable == 10 and row.n_self_won == 10
    assert "re-estimated" in cal.provenance["subset"]["thresholds"]
    # Under the re-estimated tau the gate is active again in the subset:
    # margin 2.0 < tau is false (2.0 < 2.0 fails, strict), so the prediction
    # stays; a threshold equal to the margin does not re-examine (reference
    # strict-inequality semantics).
    model = UnilidModel(out, calibrated=True)
    assert model.predict("ab")[0] == "aaa"
    assert model.last_reexamination_stats["group_a"]["n_examined"] == 0


def test_cli_subset_missing_corpus_file(v2_model, tmp_path):
    empty = tmp_path / "empty_corpus"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="aaa_train.txt"):
        calibrate_main(["subset", str(v2_model), "-o",
                        str(tmp_path / "y.unilid"), "--langs", "aaa,bbb",
                        "--recalibrate", str(empty)])
