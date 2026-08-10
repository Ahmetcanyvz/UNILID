"""Unit tests for unilid.add_language.add_language: container assembly,
ordering, guards, and the calibration update, with the actual training step
(_train_new_language) monkeypatched away. estimate_tau itself runs for real
against the Rust extension (as the design requires it to run on a
calibrated-loaded model), so the tau values below are hand-derived from the
same weights the fake trainer writes, not read back from a prior run.

Base model (langs ["X", "Y"], vocab <unk>=0,a=1,b=2,ab=3, row[3] far below
the clamp target -21.0 for every row so the clamp is a no-op and DP always
prefers "a"+"b" over the single "ab" token for the calibration text "ab"):

    X: [0.0, -5.0,  -5.0, -100.0]  -> score("ab") = -10.0
    Y: [0.0, -5.25, -5.0, -100.0]  -> score("ab") = -10.25

X is a group-A member (N=500 < head_n) with tau=5.0.
"""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from unilid.calibration import Calibration, TauRow, UnilidCalibrationError
from unilid.model_io import load_unilid, read_calibration, write_unilid

# unilid/__init__.py does `from unilid.add_language import add_language`,
# which rebinds the *attribute* `unilid.add_language` to the function (so
# library users can call `unilid.add_language(...)` directly). That means
# `import unilid.add_language as x` binds `x` to the function, not the
# module -- fetch the actual submodule via importlib instead, so tests can
# monkeypatch its module-level helpers (_train_new_language,
# _count_training_lines).
add_language_mod = importlib.import_module("unilid.add_language")

EXISTING_LANGS = ["X", "Y"]
EXISTING_WEIGHTS = np.array([
    [0.0, -5.0, -5.0, -100.0],
    [0.0, -5.25, -5.0, -100.0],
], dtype=np.float32)


def _existing_calibration() -> Calibration:
    return Calibration(
        unseen_token_constant=-21.0, head_n=18000, replacement_min_n=100000,
        proximity_bound=21.0, topk=5, margin_q=5.0, group_b_percentile=5.0,
        calib_max=2000, min_calib_lines=200, calib_seed=0,
        group_a={"X": TauRow(tau=5.0, excluded=False, cause="",
                             n_scoreable=1000, n_self_won=900)},
        group_b={},
        train_counts={"X": 500, "Y": 200000},
        provenance={})


@pytest.fixture
def existing_model_path(tmp_path, tiny_base_tok_json):
    path = tmp_path / "existing.unilid"
    write_unilid(path, tiny_base_tok_json.encode("utf-8"), EXISTING_LANGS,
                EXISTING_WEIGHTS, calibration=_existing_calibration())
    return path


def _write_lines(path: Path, text: str, n: int) -> None:
    path.write_text((text + "\n") * n, encoding="utf-8")


def make_fake_trainer(token_scores: dict, omit_token: str = None):
    """Replacement for unilid.add_language._train_new_language: clones the
    base tokenizer's serialized form and overwrites its vocab scores from
    `token_scores` (keyed by token string), optionally omitting one token
    entirely to simulate incomplete vocabulary coverage. Mirrors the vocab
    reconstruction pattern used by unilid.model_io.unpack_unilid."""
    def _fake_train_new_language(base_tok_json, lang, train_file, workdir,
                                 method, em_rounds):
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        base_tok = Tokenizer.from_str(base_tok_json)
        base_state = json.loads(base_tok.model.__getstate__().decode("utf-8"))
        base_state.pop("type", None)
        vocab_order = [tok for tok, _ in base_state["vocab"]]
        new_vocab = [(tok, token_scores[tok]) for tok in vocab_order
                    if tok != omit_token]
        state = dict(base_state)
        state["vocab"] = new_vocab
        lang_tok = Tokenizer.from_str(base_tok_json)
        lang_tok.model = lang_tok.model.__class__(**state)
        out_path = workdir / f"langspec_sp_{lang}.tokenizer.json"
        lang_tok.save(str(out_path))
        return out_path
    return _fake_train_new_language


# ---------------------------------------------------------------- happy path

def test_happy_path_small_n_appends_row_and_estimates_tau(
        tmp_path, existing_model_path, monkeypatch):
    train_file = tmp_path / "w_train.txt"
    _write_lines(train_file, "ab", 300)
    # W's own segmentation score("ab") = max(row[3]=-0.5, row[1]+row[2]=-100)
    # = -0.5, which beats X(-10.0) and Y(-10.25) on every one of the 300
    # identical calibration lines, so W wins every line with a constant
    # margin of -0.5 - (-10.0) = 9.5; np.percentile of a constant array is
    # that constant at any quantile, so tau = 9.5 regardless of q_L's exact
    # value (300 < head_n=18000, so q_L is > 0 and this is not zero_strength;
    # 300 >= min_calib_lines=200, so this is not low_calibration).
    token_scores = {"<unk>": 0.0, "a": -50.0, "b": -50.0, "ab": -0.5}
    monkeypatch.setattr(add_language_mod, "_train_new_language",
                        make_fake_trainer(token_scores))
    output_path = tmp_path / "extended.unilid"
    original_bytes = existing_model_path.read_bytes()

    result = add_language_mod.add_language(existing_model_path, "W",
                                            train_file, output_path)

    assert existing_model_path.read_bytes() == original_bytes  # input untouched

    _base_tok, weights_out, langs_out = load_unilid(output_path)
    assert langs_out == ["X", "Y", "W"]
    np.testing.assert_array_equal(np.array(weights_out[0]), EXISTING_WEIGHTS[0])
    np.testing.assert_array_equal(np.array(weights_out[1]), EXISTING_WEIGHTS[1])
    np.testing.assert_array_equal(
        np.array(weights_out[2]),
        np.array([0.0, -50.0, -50.0, -0.5], dtype=np.float32))

    cal2 = read_calibration(output_path)
    assert cal2.train_counts == {"X": 500, "Y": 200000, "W": 300}
    expected_row = TauRow(tau=9.5, excluded=False, cause="",
                          n_scoreable=300, n_self_won=300)
    assert cal2.group_a["W"] == expected_row
    assert cal2.group_a["X"] == _existing_calibration().group_a["X"]  # untouched

    added = cal2.provenance["added_languages"]
    assert len(added) == 1
    assert added[0]["lang"] == "W"
    assert added[0]["n"] == 300
    assert added[0]["method"] == "sp"

    tmp_container = output_path.with_name(output_path.stem + ".tmp.unilid")
    assert not tmp_container.exists()

    assert result["lang"] == "W"
    assert result["n"] == 300
    assert result["tau_row"] == expected_row
    assert result["replacement_candidate"] is False  # 300 < replacement_min_n
    assert "left unchanged" in result["clamp"]  # row.min()=-50 <= c=-21


def test_happy_path_large_n_skips_threshold_estimation(
        tmp_path, existing_model_path, monkeypatch):
    train_file = tmp_path / "w2_train.txt"
    train_file.write_text("ab\n", encoding="utf-8")  # content irrelevant: count is patched
    token_scores = {"<unk>": 0.0, "a": -3.0, "b": -3.0, "ab": -3.0}
    monkeypatch.setattr(add_language_mod, "_train_new_language",
                        make_fake_trainer(token_scores))
    monkeypatch.setattr(add_language_mod, "_count_training_lines",
                        lambda path: 20000)  # >= head_n=18000
    output_path = tmp_path / "extended_large.unilid"

    result = add_language_mod.add_language(existing_model_path, "W2",
                                            train_file, output_path)

    cal2 = read_calibration(output_path)
    assert "W2" not in cal2.group_a
    assert "W2" not in cal2.group_b
    assert cal2.train_counts["W2"] == 20000
    assert result["tau_row"] is None
    assert result["replacement_candidate"] is False  # 20000 < replacement_min_n=100000

    tmp_container = output_path.with_name(output_path.stem + ".tmp.unilid")
    assert not tmp_container.exists()


# -------------------------------------------------------------------- guards

def test_fill_leak_guard_raises_when_a_vocab_token_is_missing(
        tmp_path, existing_model_path, monkeypatch):
    train_file = tmp_path / "leaky_train.txt"
    _write_lines(train_file, "ab", 10)
    token_scores = {"<unk>": 0.0, "a": -1.0, "ab": -1.0}  # "b" omitted
    monkeypatch.setattr(add_language_mod, "_train_new_language",
                        make_fake_trainer(token_scores, omit_token="b"))
    output_path = tmp_path / "leaky.unilid"

    with pytest.raises(RuntimeError, match="missing"):
        add_language_mod.add_language(existing_model_path, "Leaky",
                                      train_file, output_path)


def test_lang_already_in_model_raises(tmp_path, existing_model_path, monkeypatch):
    train_file = tmp_path / "x_train.txt"
    _write_lines(train_file, "ab", 10)
    monkeypatch.setattr(
        add_language_mod, "_train_new_language",
        make_fake_trainer({"<unk>": 0.0, "a": -1.0, "b": -1.0, "ab": -1.0}))
    output_path = tmp_path / "dup.unilid"

    with pytest.raises(ValueError, match="already in the model"):
        add_language_mod.add_language(existing_model_path, "X", train_file, output_path)


def test_output_path_equal_to_input_raises(tmp_path, existing_model_path):
    train_file = tmp_path / "dummy_train.txt"
    _write_lines(train_file, "ab", 10)
    with pytest.raises(ValueError, match="must differ"):
        add_language_mod.add_language(existing_model_path, "W", train_file,
                                      existing_model_path)


def test_v1_input_without_calibration_raises(tmp_path, tiny_base_tok_json):
    v1_path = tmp_path / "v1_no_cal.unilid"
    write_unilid(v1_path, tiny_base_tok_json.encode("utf-8"), EXISTING_LANGS,
                EXISTING_WEIGHTS)  # no calibration bundled
    train_file = tmp_path / "w_train_v1.txt"
    _write_lines(train_file, "ab", 10)
    output_path = tmp_path / "out_v1.unilid"

    with pytest.raises(UnilidCalibrationError, match="carries no calibration artifact"):
        add_language_mod.add_language(v1_path, "W", train_file, output_path)


def test_empty_train_file_raises(tmp_path, existing_model_path):
    train_file = tmp_path / "empty_train.txt"
    train_file.write_text("\n\n\n", encoding="utf-8")
    output_path = tmp_path / "out_empty.unilid"

    with pytest.raises(ValueError, match="no non-empty lines"):
        add_language_mod.add_language(existing_model_path, "W", train_file, output_path)


# ------------------------------------------------------------ excluded outcome

def test_excluded_outcome_low_calibration_still_writes_container(
        tmp_path, existing_model_path, monkeypatch):
    train_file = tmp_path / "v_train.txt"
    _write_lines(train_file, "ab", 150)  # < min_calib_lines=200
    token_scores = {"<unk>": 0.0, "a": -3.0, "b": -3.0, "ab": -0.2}
    monkeypatch.setattr(add_language_mod, "_train_new_language",
                        make_fake_trainer(token_scores))
    output_path = tmp_path / "excluded.unilid"

    result = add_language_mod.add_language(existing_model_path, "V",
                                            train_file, output_path)

    cal2 = read_calibration(output_path)
    v_row = cal2.group_a["V"]
    assert v_row.excluded is True
    assert v_row.cause == "low_calibration"
    assert v_row.tau == float("-inf")
    assert result["tau_row"].cause == "low_calibration"

    _base_tok, _weights_out, langs_out = load_unilid(output_path)
    assert langs_out == ["X", "Y", "V"]
