"""Tests for doctor.py, the setup checker.

The first test encodes the reason doctor.py is a top-level script instead of a
console script or a `unilid.doctor` module: it has to run in an environment
where the unilid package cannot be imported, because that is the environment it
exists to diagnose.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import doctor  # noqa: E402


def test_doctor_runs_when_the_package_cannot_be_imported(tmp_path):
    """doctor.py must import nothing from unilid, and must still report on a
    checkout whose tokenizers extension is not built."""
    blockers = tmp_path / "blockers"
    blockers.mkdir()
    for module in ("unilid", "tokenizers"):
        (blockers / f"{module}.py").write_text(
            f'raise ImportError("blocked: {module}")\n')

    done = subprocess.run(
        [sys.executable, str(REPO_ROOT / "doctor.py")],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(blockers)})

    # It ran to completion and diagnosed the extension rather than dying on an
    # import of its own.
    assert done.returncode == 1, done.stderr
    assert "tokenizers extension" in done.stdout
    assert "maturin develop --release" in done.stdout
    assert "Traceback" not in done.stderr


def test_missing_tokenizers_submodule_names_the_init_command(tmp_path):
    (tmp_path / ".gitmodules").write_text("")
    checks = {c.name: c for c in doctor.check_submodules(tmp_path)}

    tokenizers = checks["tokenizers submodule"]
    assert tokenizers.status == doctor.FAIL
    assert tokenizers.remedy == "git submodule update --init tokenizers"

    # The SentencePiece submodule is optional, so its absence is a note.
    assert checks["sentencepiece submodule"].status == doctor.NOTE


@pytest.mark.parametrize("version,expected_status", [
    ("rustc 1.51.0 (2fd73fabe 2021-03-23)", doctor.FAIL),
    ("rustc 1.84.9 (abcdef123 2025-01-01)", doctor.FAIL),
    ("rustc 1.85.0 (abcdef123 2025-02-20)", doctor.OK),
    ("rustc 1.93.1 (01f6ddf75 2026-02-11)", doctor.OK),
])
def test_rust_version_boundary(monkeypatch, version, expected_status):
    """The 2021-era stable that produced the reported weak-dep-features error
    is rejected, and the remedy is the one that actually fixes it."""
    monkeypatch.setattr(doctor, "_run", lambda command: version)
    check = doctor.check_rust()

    assert check.status == expected_status
    if expected_status == doctor.FAIL:
        assert check.remedy == "rustup update stable"


def test_build_only_checks_are_required_only_for_the_build(monkeypatch, capsys):
    """A wheel install has no submodule, no toolchain and no maturin, and needs
    none of them, so those must not fail a run whose extension already works."""
    monkeypatch.setattr(doctor, "check_submodules",
                        lambda: [doctor.Check("tokenizers submodule",
                                              doctor.FAIL, "",
                                              "git submodule update --init tokenizers",
                                              build_only=True)])
    monkeypatch.setattr(doctor, "check_spm_train",
                        lambda: doctor.Check("spm_train", doctor.OK, ""))
    monkeypatch.setattr(doctor, "check_rust",
                        lambda: doctor.Check("rust toolchain", doctor.FAIL, "",
                                             "rustup update stable",
                                             build_only=True))
    monkeypatch.setattr(doctor, "check_maturin",
                        lambda: doctor.Check("maturin", doctor.FAIL, "",
                                             "pip install maturin",
                                             build_only=True))

    monkeypatch.setattr(doctor, "check_extension",
                        lambda: doctor.Check("tokenizers extension", doctor.OK, ""))
    assert doctor.main() == 0

    monkeypatch.setattr(doctor, "check_extension",
                        lambda: doctor.Check("tokenizers extension", doctor.FAIL,
                                             "", "maturin develop --release"))
    assert doctor.main() == 1
    assert "rust toolchain" in capsys.readouterr().out
