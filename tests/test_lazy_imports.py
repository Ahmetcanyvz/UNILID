"""A prediction-only install (base dependencies: numpy, tqdm, the tokenizers
extension) must be able to `from unilid import load_model`. The training-side
classes pull in the optional [train] dependencies and are imported lazily."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent

_BLOCKED_IMPORT_CHECK = """
import sys

class _Blocker:
    BLOCKED = {"torch", "ujson", "transformers", "sentencepiece"}
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(f"blocked optional dependency: {name}")
        return None

sys.meta_path.insert(0, _Blocker())
from unilid import (
    Calibration,
    UnilidCalibrationError,
    UnilidModel,
    add_language,
    estimate_tau,
    load_model,
    read_calibration,
    save_unilid,
    unpack_unilid,
    write_unilid,
)
print("PREDICTION_IMPORTS_OK")
"""


def test_prediction_imports_without_train_extras():
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT_CHECK],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0 and "PREDICTION_IMPORTS_OK" in result.stdout, \
        f"stdout={result.stdout!r}\nstderr={result.stderr[-2000:]}"


def test_lazy_training_attributes_resolve():
    import unilid

    assert callable(unilid.train_language_specific_tokenizer)
    assert unilid.LanguageSpecificUnigramLMTokenizer.__name__ == \
        "LanguageSpecificUnigramLMTokenizer"
    assert "CorpusTokenizer" in dir(unilid)


def test_unknown_attribute_raises():
    import unilid

    try:
        unilid.does_not_exist
    except AttributeError as e:
        assert "does_not_exist" in str(e)
    else:
        raise AssertionError("expected AttributeError")
