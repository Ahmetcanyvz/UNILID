"""End-to-end add-language integration tests: the REAL trainer runs (no
monkeypatching), on tiny constructed corpora. Uses the same base-training
method as the worked example in examples/add_language (HuggingFace's
UnigramTrainer, byte-level) at a smaller vocabulary and line count, so the
whole flow stays fast.

Covers: base training -> v1 container -> calibration bootstrap -> v2 bundle ->
add_language with the pure-Python EM trainer -> prediction quality on held-out
lines of the added language, row byte-identity, and the calibration update.

Everything here runs without SentencePiece. The two tests that need it, the
add-language sp variant and the base-vocabulary seeding path, are skipped
unless both the spm_train binary and the sentencepiece package are present.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from .conftest import sentencepiece_available

sys.path.insert(0, str(Path(__file__).parent.parent))

from unilid.add_language import add_language
from unilid.calibration import (
    CAUSE_LOW_CALIBRATION,
    Calibration,
    TauRow,
    estimate_tau,
)
from unilid.api import train_language_specific_tokenizer
from unilid.model_io import (
    UnilidModel,
    load_unilid_raw,
    save_unilid,
    write_unilid,
)

SYLLABLES = {
    "aaa_Latn": ["ba", "ke", "ti", "mo", "ru", "san", "pel", "dor"],
    "bbb_Latn": ["zu", "gri", "vol", "ne", "sha", "kom", "tir", "bek"],
    "ccc_Latn": ["fi", "lo", "wen", "dar", "mu", "hes", "tal", "gor"],
    "ddd_Latn": ["fi", "lo", "pra", "kli", "ven", "dus", "mar", "tol"],
}
TRAIN_LINES = {"aaa_Latn": 400, "bbb_Latn": 400, "ccc_Latn": 150,
               "ddd_Latn": 180}
TEST_LINES = 40
CONSTANTS = {
    "unseen_token_constant": -21.0,
    "head_n": 200,
    "replacement_min_n": 300,
    "proximity_bound": 21.0,
    "topk": 4,
    "margin_q": 5.0,
    "group_b_percentile": 5.0,
    "calib_max": 2000,
    "min_calib_lines": 30,
    "calib_seed": 0,
}


def _sentence(rng, syllables):
    words = []
    for _ in range(rng.randint(4, 9)):
        words.append("".join(rng.choice(syllables)
                             for _ in range(rng.randint(1, 3))))
    return " ".join(words) + "."


@pytest.fixture(scope="module")
def toy_setup(tmp_path_factory):
    """Train the 3-language base model once, bundle a calibration, and return
    every path the tests need."""
    root = tmp_path_factory.mktemp("addlang_e2e")
    data = root / "data"
    data.mkdir()
    rng = random.Random(4242)
    for lang in sorted(SYLLABLES):
        train = [_sentence(rng, SYLLABLES[lang])
                 for _ in range(TRAIN_LINES[lang])]
        test = [_sentence(rng, SYLLABLES[lang]) for _ in range(TEST_LINES)]
        (data / f"{lang}_train.txt").write_text("\n".join(train) + "\n")
        (data / f"{lang}_test.txt").write_text("\n".join(test) + "\n")

    base_langs = ["aaa_Latn", "bbb_Latn", "ccc_Latn"]
    tok_dir = root / "tokenizers"
    tok_dir.mkdir()
    result = train_language_specific_tokenizer(
        corpus_info={l: str(data / f"{l}_train.txt") for l in base_langs},
        vocab_size=250,
        output_dir=str(tok_dir),
        byte_level=True,
        use_sentencepiece=False,
        # The base vocabulary is a separate training step from the per-language
        # re-estimation that use_sentencepiece selects. "hf" is train.py's
        # default and what examples/add_language/run_example.sh therefore runs,
        # and it needs no spm_train binary, so this module runs on an install
        # without the SentencePiece CLI.
        base_em_mode="hf",
    )
    assert result and set(result["language_paths"]) == set(base_langs)

    v1 = root / "toy_base.unilid"
    save_unilid(tok_dir, v1)

    base_tok_bytes, weights, langs = load_unilid_raw(v1)
    counts = {l: TRAIN_LINES[l] for l in langs}
    group_a_langs = [l for l in langs if counts[l] < CONSTANTS["head_n"]]
    assert group_a_langs == ["ccc_Latn"]

    placeholder = TauRow(tau=float("-inf"), excluded=True,
                         cause=CAUSE_LOW_CALIBRATION, n_scoreable=0,
                         n_self_won=0)
    provisional = Calibration(
        group_a={l: placeholder for l in group_a_langs}, group_b={},
        train_counts=counts, provenance={}, **CONSTANTS)
    weights_arr = weights.copy()
    del weights
    tmp = root / "tmp.unilid"
    write_unilid(tmp, base_tok_bytes, langs, weights_arr, provisional)
    model = UnilidModel(tmp, calibrated=True)
    rows = {}
    for lang in group_a_langs:
        lines = [l for l in
                 (data / f"{lang}_train.txt").read_text().splitlines() if l]
        rows[lang] = estimate_tau(model, lang, lines, counts[lang])
    del model
    tmp.unlink()

    final = Calibration(group_a=rows, group_b={}, train_counts=counts,
                        provenance={}, **CONSTANTS)
    v2 = root / "toy_calibrated.unilid"
    write_unilid(v2, base_tok_bytes, langs, weights_arr, final)
    return {"root": root, "data": data, "v2": v2, "langs": langs}


def _held_out_accuracy(model, data, lang):
    lines = [l for l in
             (data / f"{lang}_test.txt").read_text().splitlines() if l]
    preds = [p for p, _t, _s in model.predict_batch(lines)]
    return sum(p == lang for p in preds) / len(preds)


def _run_add(toy_setup, method, out_name):
    data = toy_setup["data"]
    out = toy_setup["root"] / out_name
    summary = add_language(
        toy_setup["v2"], "ddd_Latn", data / "ddd_Latn_train.txt", out,
        method=method, workdir=toy_setup["root"] / f"work_{method}")
    return out, summary


def test_add_language_em_end_to_end(toy_setup):
    data = toy_setup["data"]
    out, summary = _run_add(toy_setup, "em", "extended_em.unilid")

    # The real trainer ran and the added language wins on its own held-out data
    model = UnilidModel(out, calibrated=True)
    assert model.langs == toy_setup["langs"] + ["ddd_Latn"]
    assert _held_out_accuracy(model, data, "ddd_Latn") >= 0.9
    # Existing languages keep predicting their own held-out data
    for lang in toy_setup["langs"]:
        assert _held_out_accuracy(model, data, lang) >= 0.9

    # Existing weight rows are byte-identical to the input container's
    _tb, w_in, _l = load_unilid_raw(toy_setup["v2"])
    _tb2, w_out, _l2 = load_unilid_raw(out)
    assert np.array_equal(np.asarray(w_in), np.asarray(w_out[:-1]))

    # Calibration update: threshold estimated (N=120 < head_n=200), counts
    # updated, not a replacement candidate (N=120 < replacement_min_n=300)
    cal = model.calibration
    row = cal.group_a["ddd_Latn"]
    assert not row.excluded and np.isfinite(row.tau)
    assert cal.train_counts["ddd_Latn"] == TRAIN_LINES["ddd_Latn"]
    assert summary["replacement_candidate"] is False
    assert summary["tau_row"] == row

    # The input container is unchanged and the tmp file is gone
    assert not (toy_setup["root"] / "extended_em.tmp.unilid").exists()


def test_add_language_em_provenance_and_input_untouched(toy_setup):
    v2_before = toy_setup["v2"].read_bytes()
    out, _ = _run_add(toy_setup, "em", "extended_em2.unilid")
    assert toy_setup["v2"].read_bytes() == v2_before
    model = UnilidModel(out, calibrated=True)
    added = model.calibration.provenance["added_languages"]
    assert added[-1]["lang"] == "ddd_Latn" and added[-1]["method"] == "em"


@pytest.mark.skipif(not sentencepiece_available(),
                    reason="needs the spm_train binary and the "
                           "sentencepiece package")
def test_sp_seed_vocab_runs(tmp_path):
    """The base-vocabulary SentencePiece seeding path still works.

    The toy_setup fixture builds its base tokenizer with base_em_mode="hf", so
    nothing else in the suite reaches _sp_seed_vocab; without this test that
    code path would have no coverage at all.
    """
    from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer

    rng = random.Random(99)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(_sentence(rng, SYLLABLES["aaa_Latn"])
                                for _ in range(300)) + "\n")

    tok = StandardUnigramLMTokenizer(vocab_size=50, em_mode="soft",
                                     byte_level=True)
    vocab = tok._sp_seed_vocab([str(corpus)])

    assert isinstance(vocab, set) and len(vocab) > 50
    assert all(isinstance(t, str) for t in vocab)


@pytest.mark.parametrize("method_name,flag", [("_sp_seed_vocab", "use_sp_seed_vocab"),
                                              ("_sp_em", "use_sp_em")])
def test_sp_paths_name_the_missing_binary(monkeypatch, method_name, flag):
    """A missing spm_train is a named error, not a bare FileNotFoundError from
    subprocess several frames down."""
    from unilid.trainers import em_loop
    from unilid.trainers.standard_trainer import StandardUnigramLMTokenizer

    # Pin both preconditions so the assertion is about the binary check alone,
    # whatever the running environment happens to have installed.
    monkeypatch.setattr(em_loop, "spm", object())
    monkeypatch.setattr(em_loop.shutil, "which", lambda name: None)

    tok = StandardUnigramLMTokenizer(vocab_size=50, em_mode="soft",
                                     byte_level=True)
    with pytest.raises(RuntimeError) as excinfo:
        if method_name == "_sp_seed_vocab":
            tok._sp_seed_vocab(["nonexistent.txt"])
        else:
            tok._sp_em({"a": 0}, "nonexistent.txt")

    assert "spm_train" in str(excinfo.value)
    assert flag in str(excinfo.value)


@pytest.mark.skipif(not sentencepiece_available(),
                    reason="needs the spm_train binary and the "
                           "sentencepiece package")
def test_add_language_sp_end_to_end(toy_setup):
    """The SentencePiece path completes and calibrates; accuracy is not
    asserted at the EM path's level (at toy data sizes the sp trainer
    estimates a flatter distribution; see examples/add_language/README.md)."""
    out, summary = _run_add(toy_setup, "sp", "extended_sp.unilid")
    model = UnilidModel(out, calibrated=True)
    assert model.langs[-1] == "ddd_Latn"
    row = model.calibration.group_a["ddd_Latn"]
    assert row.excluded or np.isfinite(row.tau)
    _tb, w_in, _l = load_unilid_raw(toy_setup["v2"])
    _tb2, w_out, _l2 = load_unilid_raw(out)
    assert np.array_equal(np.asarray(w_in), np.asarray(w_out[:-1]))
