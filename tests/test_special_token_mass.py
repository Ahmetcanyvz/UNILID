"""Special tokens must hold no probability mass, under any training method.

No special token's stored weight is ever read when scoring: the Rust scorer
takes its unknown-token score from a global constant, and <s>/</s>/<pad> are
reachable only by text containing those literal substrings. Mass parked on them
is therefore mass taken from the tokens that do contribute, which lowers every
real token by a constant and makes rows trained by different methods
incomparable inside one model.
"""
import math

import numpy as np
import pytest

from unilid.constants import MIN_TOKEN_LOG_PROB
from unilid.model_io import UnilidModel, write_unilid
from unilid.vocab_io import renormalize_over_real_tokens, special_token_set


def _real_mass(logps, specials):
    return sum(math.exp(lp) for tk, lp in logps.items() if tk not in specials)


def test_specials_lose_their_mass_and_real_tokens_gain_it():
    specials = special_token_set()
    # <unk> holding a fifth of the mass is what the sp path used to produce.
    logps = {"<s>": 0.0, "</s>": 0.0, "<pad>": 0.0, "<unk>": math.log(0.2),
             "a": math.log(0.4), "b": math.log(0.4)}
    out = renormalize_over_real_tokens(logps, specials)

    assert _real_mass(out, specials) == pytest.approx(1.0)
    for token in specials:
        assert out[token] == MIN_TOKEN_LOG_PROB
    # The real tokens keep their relative sizes and share the whole mass.
    assert out["a"] == pytest.approx(math.log(0.5))
    assert out["b"] == pytest.approx(math.log(0.5))


def test_relative_order_of_real_tokens_is_untouched():
    specials = special_token_set()
    logps = {"<unk>": math.log(0.9), "a": math.log(0.06), "b": math.log(0.03),
             "c": math.log(0.01)}
    out = renormalize_over_real_tokens(logps, specials)
    assert out["a"] - out["b"] == pytest.approx(logps["a"] - logps["b"])
    assert out["b"] - out["c"] == pytest.approx(logps["b"] - logps["c"])


def test_empty_real_vocabulary_is_an_error():
    with pytest.raises(ValueError, match="no non-special tokens"):
        renormalize_over_real_tokens({"<unk>": 0.0, "<s>": 0.0},
                                     special_token_set())


def test_special_token_weights_do_not_affect_scores(tmp_path,
                                                    tiny_base_tok_json):
    """The premise the whole rule rests on, asserted rather than assumed."""
    langs = ["X", "Y"]
    weights = np.array([[0.0, -1.0, -2.0, -3.0],
                        [0.0, -2.0, -1.0, -3.0]], dtype=np.float32)
    texts = ["ab", "a", "b", "abab"]

    def scores(w):
        path = tmp_path / f"m{abs(hash(w.tobytes()))}.unilid"
        write_unilid(path, tiny_base_tok_json.encode("utf-8"), langs, w)
        model = UnilidModel(path, calibrated=False)
        out = [s for _l, _t, s in model.predict_batch(texts)]
        del model
        return np.array(out)

    reference = scores(weights)
    perturbed = weights.copy()
    unk_id = 0  # <unk> is index 0 of TINY_VOCAB
    perturbed[:, unk_id] = -500.0
    np.testing.assert_array_equal(scores(perturbed), reference)


def test_add_language_puts_the_new_row_on_the_model_s_scale(tmp_path):
    """A row trained at full mass must not outscore an older model's rows by a
    constant per token just because it kept more of its own mass."""
    from unilid.add_language import _match_real_token_scale

    ref_vocab = {"<unk>": 0, "a": 1, "b": 2, "ab": 3}
    # An old-scale model: a fifth of each row's mass sits on the specials.
    existing = np.array([[math.log(0.2), math.log(0.4), math.log(0.3),
                          math.log(0.1)]], dtype=np.float32)
    new_row = np.array([MIN_TOKEN_LOG_PROB, math.log(0.5), math.log(0.3),
                        math.log(0.2)], dtype=np.float32)

    out = _match_real_token_scale(new_row, existing, ref_vocab, "W")

    real = [1, 2, 3]
    assert float(np.exp(out[real].astype(np.float64)).sum()) == pytest.approx(0.8, rel=1e-5)
    # Only the scale moved: the real tokens keep their relative sizes, and the
    # special token stays at the floor.
    for i in real:
        assert out[i] - new_row[i] == pytest.approx(out[1] - new_row[1], abs=1e-5)
    assert out[0] == MIN_TOKEN_LOG_PROB


# --------------------------------------------------------------------------
# The load-time generation report
#
# The container header encodes only version 1 against version 2 and
# FORMAT_VERSION_MAX = 2, so a corrected file cannot be told from a pre-0.3.0
# one by a version bump without making it unreadable to every published reader.
# The mass its rows put on real tokens says it instead: 0.2 per row is the
# pre-0.3.0 handling, 1.0 a corrected file.
# --------------------------------------------------------------------------
UNK_COLUMN = 4          # <unk> is the LAST column of the vocabulary below
VOCAB_SIZE = 5


def _shifted_base_tokenizer():
    """A base tokenizer whose <unk> is the last column rather than the first.

    A reader that assumed the special tokens occupy columns 0-3 would subtract
    three real tokens and keep <unk>, measuring 0.8 on the pre-0.3.0 fixture
    below and 1e-12 on the corrected one, so both would be reported as neither
    signature. That is the failure this vocabulary exists to catch: a base
    tokenizer converted from an LLM's really does have them at non-contiguous
    high indices.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram

    vocab = [("a", -1.0), ("b", -2.0), ("ab", -1.5), ("c", -3.0), ("<unk>", 0.0)]
    return Tokenizer(Unigram(vocab, UNK_COLUMN))


def _row(real_mass: float, unk_mass: float) -> np.ndarray:
    """One row over that vocabulary: ``real_mass`` split evenly over the four
    real tokens, ``unk_mass`` on <unk> (the floor when it holds none)."""
    row = np.full(VOCAB_SIZE, math.log(real_mass / (VOCAB_SIZE - 1)),
                  dtype=np.float32)
    row[UNK_COLUMN] = np.float32(math.log(unk_mass) if unk_mass > 0.0
                                 else MIN_TOKEN_LOG_PROB)
    return row


def _load_reporting(tmp_path, name, rows, capsys):
    """Write a container of ``rows`` and load it, returning (model, stdout)."""
    from unilid.model_io import write_unilid

    weights = np.stack(rows).astype(np.float32)
    langs = [f"L{i}" for i in range(len(rows))]
    path = tmp_path / name
    write_unilid(path, _shifted_base_tokenizer().to_str().encode("utf-8"),
                 langs, weights)
    model = UnilidModel(path, calibrated=False)
    return model, capsys.readouterr().out


def test_load_names_the_pre_0_3_0_generation_and_its_upgrade_path(tmp_path,
                                                                  capsys):
    # 0.8 on the specials, 0.2 left for the real tokens: the released model.
    _model, out = _load_reporting(tmp_path, "old.unilid",
                                  [_row(0.2, 0.8), _row(0.2, 0.8)], capsys)

    assert "Real-token mass 0.200000 to 0.200000 over 2 rows" in out
    assert f"specials: columns [{UNK_COLUMN}]" in out
    assert "pre-0.3.0 special-token handling, real-token mass 0.200 per row" in out
    assert "WARNING:" in out
    assert "renormalize_over_real_tokens" in out


def test_load_names_a_corrected_file_without_warning(tmp_path, capsys):
    _model, out = _load_reporting(tmp_path, "new.unilid",
                                  [_row(1.0, 0.0), _row(1.0, 0.0)], capsys)

    assert "Real-token mass 1.000000 to 1.000000 over 2 rows" in out
    assert "corrected, 1.000 per row" in out
    assert "WARNING" not in out


def test_an_unrecognised_mass_warns_but_still_loads_and_scores(tmp_path,
                                                               capsys):
    """A published file must stay loadable whatever its rows hold, so a mass
    that is neither signature is a louder report, never an error."""
    model, out = _load_reporting(tmp_path, "odd.unilid",
                                 [_row(0.5, 0.5), _row(1.0, 0.0)], capsys)

    assert "Real-token mass 0.500000 to 1.000000 over 2 rows" in out
    assert "NEITHER SIGNATURE" in out
    assert "WARNING" in out
    lang, _tokens, _score = model.predict_batch(["ab"])[0]
    assert lang in {"L0", "L1"}


def test_the_blocked_mass_pass_is_exact_across_block_boundaries():
    """The pass is blocked so a released-scale matrix is never a full float64
    temporary; a block boundary must not change the answer."""
    from unilid.model_io import real_token_mass

    rng = np.random.default_rng(0)
    weights = rng.uniform(-12.0, -1.0, size=(7, 11)).astype(np.float32)
    specials = [2, 9]
    real = [c for c in range(weights.shape[1]) if c not in specials]
    expected = np.exp(weights[:, real].astype(np.float64)).sum(axis=1)

    for block in (1, 2, 3, 7, 100):
        got = real_token_mass(weights, specials, block=block)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_the_report_measures_the_stored_matrix_not_the_clamped_one(tmp_path,
                                                                   capsys):
    """Calibrated inference clamps every unseen-token entry down to the
    calibration constant before scoring. The generation is a property of the
    file, so it has to be read off the stored matrix; measuring the clamped one
    would report a corrected model as neither signature.
    """
    from unilid.calibration import Calibration, TauRow
    from unilid.model_io import UnilidModel, write_unilid

    # 0.97 on one token and an unseen plateau of 0.01 x 3: real mass 1.000
    # stored, 0.970 once the plateau is clamped to log(1e-6).
    row = np.array([math.log(0.97), math.log(0.01), math.log(0.01),
                    math.log(0.01), MIN_TOKEN_LOG_PROB], dtype=np.float32)
    weights = np.stack([row, row])
    cal = Calibration(
        unseen_token_constant=math.log(1e-6), head_n=1000,
        replacement_min_n=100000, proximity_bound=21.0, topk=5, margin_q=5.0,
        group_b_percentile=5.0, calib_max=2000, min_calib_lines=200,
        calib_seed=0,
        group_a={"L0": TauRow(tau=1.0, excluded=False, cause="",
                              n_scoreable=100, n_self_won=90)},
        group_b={}, train_counts={"L0": 500, "L1": 2000}, provenance={})
    path = tmp_path / "cal.unilid"
    write_unilid(path, _shifted_base_tokenizer().to_str().encode("utf-8"),
                 ["L0", "L1"], weights, calibration=cal)

    UnilidModel(path, calibrated=True)
    out = capsys.readouterr().out

    assert "Real-token mass 1.000000 to 1.000000" in out
    assert "corrected, 1.000 per row" in out
    # The clamp really does move both rows, so measuring after it would have
    # shown 0.970003 and reported neither signature.
    assert "2/2 languages modified" in out
    assert "0.970" not in out


def test_a_failed_measurement_reports_itself_and_the_model_still_loads(
        tmp_path, capsys, monkeypatch):
    """The report is printing only, so no failure inside it may stop a load."""
    from unilid import model_io

    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated measurement failure")

    monkeypatch.setattr(model_io, "real_token_mass", explode)
    model, out = _load_reporting(tmp_path, "broken.unilid",
                                 [_row(1.0, 0.0)], capsys)

    assert "could not measure the real-token mass" in out
    assert "RuntimeError: simulated measurement failure" in out
    lang, _tokens, _score = model.predict_batch(["ab"])[0]
    assert lang == "L0"


def test_the_mass_pass_reads_a_memmap_without_modifying_it(tmp_path):
    """Released models are memmapped, and the pass must copy out of the map
    rather than write into it."""
    from unilid.model_io import real_token_mass

    rng = np.random.default_rng(1)
    weights = rng.uniform(-9.0, -1.0, size=(5, 8)).astype(np.float32)
    path = tmp_path / "w.f32"
    weights.tofile(path)
    mapped = np.memmap(path, dtype=np.float32, mode="r", shape=weights.shape)

    specials = [0, 7]
    got = real_token_mass(mapped, specials, block=2)
    np.testing.assert_allclose(got, real_token_mass(weights, specials),
                               rtol=0, atol=0)
    np.testing.assert_array_equal(np.array(mapped), weights)


def test_the_directory_load_path_reports_the_generation_too(tmp_path, capsys):
    """UnilidModel has two loaders, and a tokenizers directory is the one the
    .unilid tests do not reach."""
    from unilid.model_io import UnilidModel, unpack_unilid, write_unilid

    path = tmp_path / "old.unilid"
    write_unilid(path, _shifted_base_tokenizer().to_str().encode("utf-8"),
                 ["L0", "L1"], np.stack([_row(0.2, 0.8), _row(0.2, 0.8)]))
    unpacked = unpack_unilid(path, tmp_path / "unpacked")
    capsys.readouterr()

    UnilidModel(unpacked, calibrated=False)
    out = capsys.readouterr().out

    assert "Real-token mass 0.200000 to 0.200000 over 2 rows" in out
    assert "pre-0.3.0 special-token handling" in out
