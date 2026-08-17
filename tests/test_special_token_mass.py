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
