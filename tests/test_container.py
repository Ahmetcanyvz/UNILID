"""Unit tests for the .unilid container (unilid/model_io.py): v1/v2 round
trip, the documented v1 trailing-bytes tolerance, truncated-trailer
corruption errors, and version rejection, against a tiny real base
tokenizer. Format per OPEN_SOURCE_DESIGN.md section 4.1: a 32-byte header
"<8sIIIII4x" (magic, version, num_langs, vocab_size, base_tok_len, langs_len,
4 reserved bytes), then base-tokenizer JSON, langs JSON, float32 weights,
and (version 2 only) a uint64-LE length-prefixed calibration JSON trailer.
"""
import json
import struct

import numpy as np
import pytest

from unilid.calibration import Calibration, TauRow, UnilidCalibrationError
from unilid.model_io import load_unilid, read_calibration, write_unilid

MAGIC = b"UNILID\x00\x00"
HEADER_FMT = "<8sIIIII4x"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 32

LANGS = ["aaa", "bbb", "ccc"]
WEIGHTS = np.array([
    [0.0, -1.0, -2.0, -1.5],
    [0.0, -0.5, -3.0, -0.8],
    [0.0, -5.0, -5.0, -5.0],
], dtype=np.float32)


def _expected_header(version, num_langs, vocab_size, base_tok_len, langs_len):
    return struct.pack(HEADER_FMT, MAGIC, version, num_langs, vocab_size,
                       base_tok_len, langs_len)


def _small_calibration() -> Calibration:
    return Calibration(
        unseen_token_constant=-21.0, head_n=1000, replacement_min_n=100000,
        proximity_bound=21.0, topk=5, margin_q=5.0, group_b_percentile=5.0,
        calib_max=2000, min_calib_lines=200, calib_seed=0,
        group_a={"aaa": TauRow(tau=0.7, excluded=False, cause="",
                               n_scoreable=100, n_self_won=90)},
        group_b={},
        train_counts={"aaa": 500, "bbb": 2000, "ccc": 100000},
        provenance={"derived": "unit test"})


# --------------------------------------------------------------------- v1

def test_v1_round_trip_and_no_trailer(tmp_path, tiny_base_tok_json):
    base_tok_bytes = tiny_base_tok_json.encode("utf-8")
    path = tmp_path / "model.unilid"
    write_unilid(path, base_tok_bytes, LANGS, WEIGHTS)

    _base_tok, weights_out, langs_out = load_unilid(path)
    assert langs_out == LANGS
    np.testing.assert_array_equal(np.array(weights_out), WEIGHTS)
    assert np.array(weights_out).dtype == np.float32

    assert read_calibration(path) is None

    langs_bytes = json.dumps(LANGS).encode("utf-8")
    expected = (_expected_header(1, len(LANGS), WEIGHTS.shape[1],
                                 len(base_tok_bytes), len(langs_bytes))
               + base_tok_bytes + langs_bytes
               + np.ascontiguousarray(WEIGHTS).tobytes())
    assert path.read_bytes() == expected


# --------------------------------------------------------------------- v2

def test_v2_round_trip_calibration_and_weights(tmp_path, tiny_base_tok_json):
    base_tok_bytes = tiny_base_tok_json.encode("utf-8")
    cal = _small_calibration()
    path = tmp_path / "model.unilid"
    write_unilid(path, base_tok_bytes, LANGS, WEIGHTS, calibration=cal)

    loaded_cal = read_calibration(path)
    assert loaded_cal == cal

    _base_tok, weights_out, langs_out = load_unilid(path)
    assert langs_out == LANGS
    np.testing.assert_array_equal(np.array(weights_out), WEIGHTS)


# ------------------------------------------------------------ corruption

def test_v2_truncated_calibration_payload_raises(tmp_path, tiny_base_tok_json):
    base_tok_bytes = tiny_base_tok_json.encode("utf-8")
    cal = _small_calibration()
    path = tmp_path / "model.unilid"
    write_unilid(path, base_tok_bytes, LANGS, WEIGHTS, calibration=cal)

    full = path.read_bytes()
    # Chop off the last 5 bytes: the length field is intact but the JSON
    # payload it declares is short.
    truncated_path = tmp_path / "truncated_payload.unilid"
    truncated_path.write_bytes(full[:-5])

    with pytest.raises(UnilidCalibrationError, match="truncated"):
        read_calibration(truncated_path)


def test_v2_missing_length_field_raises(tmp_path, tiny_base_tok_json):
    base_tok_bytes = tiny_base_tok_json.encode("utf-8")
    cal = _small_calibration()
    path = tmp_path / "model.unilid"
    write_unilid(path, base_tok_bytes, LANGS, WEIGHTS, calibration=cal)

    full = path.read_bytes()
    cal_bytes = cal.to_json_bytes()
    # Chop off the entire trailer (8-byte length field + JSON), leaving only
    # the header+body of what would otherwise be a v1 file, but the header
    # still claims version 2.
    body_only = full[: len(full) - 8 - len(cal_bytes)]
    truncated_path = tmp_path / "no_length_field.unilid"
    truncated_path.write_bytes(body_only)

    with pytest.raises(UnilidCalibrationError, match="length field"):
        read_calibration(truncated_path)


# ---------------------------------------------------------- unsupported version

def test_unsupported_version_rejected_by_both_readers(tmp_path, tiny_base_tok_json):
    base_tok_bytes = tiny_base_tok_json.encode("utf-8")
    langs_bytes = json.dumps(LANGS).encode("utf-8")
    header = _expected_header(3, len(LANGS), WEIGHTS.shape[1],
                              len(base_tok_bytes), len(langs_bytes))
    path = tmp_path / "future_version.unilid"
    path.write_bytes(header + base_tok_bytes + langs_bytes
                     + np.ascontiguousarray(WEIGHTS).tobytes())

    with pytest.raises(ValueError, match="3"):
        load_unilid(path)
    with pytest.raises(ValueError, match="3"):
        read_calibration(path)


# --------------------------------------------------- v1 trailing-bytes tolerance

def test_v1_trailing_garbage_bytes_are_tolerated(tmp_path, tiny_base_tok_json):
    base_tok_bytes = tiny_base_tok_json.encode("utf-8")
    path = tmp_path / "model.unilid"
    write_unilid(path, base_tok_bytes, LANGS, WEIGHTS)

    garbage_path = tmp_path / "model_with_garbage.unilid"
    garbage_path.write_bytes(path.read_bytes() + b"trailing garbage bytes not part of the format")

    _base_tok, weights_out, langs_out = load_unilid(garbage_path)
    assert langs_out == LANGS
    np.testing.assert_array_equal(np.array(weights_out), WEIGHTS)
