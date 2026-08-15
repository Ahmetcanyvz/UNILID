"""Shared fixtures for the calibration test suite.

Inserts the repository root into ``sys.path`` so ``import unilid`` resolves
regardless of the current working directory the tests are invoked from (no
package installation is assumed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import Unigram  # noqa: E402


def sentencepiece_available() -> bool:
    """The sp training path needs two separate artifacts: the compiled
    spm_train binary, and the sentencepiece Python package that reads the model
    it writes. Checking only the binary is not enough.

    The import is not enough either. A `sentencepiece/` directory (the
    submodule) sits at the repository root, which conftest puts on sys.path, so
    when the pip package is absent the import succeeds and returns that
    directory as a namespace package. Hence the attribute check.
    """
    import shutil

    if shutil.which("spm_train") is None:
        return False
    try:
        import sentencepiece as spm
    except ImportError:
        return False
    return hasattr(spm, "SentencePieceProcessor")

# A 4-token vocabulary shared by every test that needs a minimal base
# tokenizer. List order fixes the vocab ids: <unk>=0, a=1, b=2, ab=3 (the
# tokenizers Unigram model assigns ids by position in the constructor's vocab
# list, confirmed via Tokenizer.get_vocab() on this exact list).
TINY_VOCAB = [("<unk>", 0.0), ("a", -1.0), ("b", -2.0), ("ab", -1.5)]

# The calibration artifact's constants block, at the values documented in
# OPEN_SOURCE_DESIGN.md section 4.2 (the released model's own constants).
# Individual tests are free to build a Calibration with different constants
# when a scenario needs cleaner arithmetic (e.g. a smaller head_n); this is
# just the shared default for tests that don't care.
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


def build_tiny_base_tokenizer() -> Tokenizer:
    """A minimal HF Unigram tokenizer over TINY_VOCAB, with no normalizer or
    pre-tokenizer. ``UnilidModel.preprocess`` on such a tokenizer returns the
    input text unchanged when it is non-empty, and None when it is empty
    (normalizer is None -> skipped; pre_tok is None -> ``text if text else
    None``), so tests can pass plain strings directly without needing a real
    pre-tokenizer.
    """
    model = Unigram(list(TINY_VOCAB), 0)
    return Tokenizer(model)


@pytest.fixture
def tiny_base_tokenizer() -> Tokenizer:
    return build_tiny_base_tokenizer()


@pytest.fixture
def tiny_base_tok_json() -> str:
    return build_tiny_base_tokenizer().to_str()
