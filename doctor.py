#!/usr/bin/env python3
"""Check that this checkout can build and run UNILID, and name the fix for
whatever is missing.

Run it from the repository root, before and after the tokenizers build:

    python doctor.py

This is a top-level script rather than a `unilid.doctor` module or a console
script on purpose, and it imports nothing from `unilid`. Importing any name
under `unilid` executes `unilid/__init__.py`, which imports `unilid.model_io`,
which does `from tokenizers import Tokenizer`. On the setups this script exists
to diagnose (submodules absent, extension not built) that import raises, so a
packaged form would crash before printing anything.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Minimum Rust toolchain for the bundled tokenizers fork.
#
# Derivation: the crates the fork's committed Cargo.lock pins declare their own
# minimums in their manifests (not in the lockfile, which carries no
# rust-version fields). The highest among the crates that a native build
# reaches is 1.85, from unicode-segmentation 1.13.2 and getrandom 0.4.2. The
# project's own floors are lower: tokenizers/bindings/python/Cargo.toml is
# edition 2021 (1.56) and pyo3 0.25 declares 1.63. This figure therefore tracks
# the pinned dependency graph and can move on any `cargo update` in the fork.
# Tested toolchain: rustc 1.93.1.
MIN_RUST_VERSION = (1, 85)

# The scorer methods model loading requires. Kept in step with
# UnilidModel._require_scorer_methods in unilid/model_io.py, which is the
# authority; duplicated here rather than imported because reading it would mean
# importing unilid, which fails in exactly the case this check detects.
SCORER_METHODS = (
    "set_weight_sets_numpy",
    "top_k_of_cached_weight_sets_batch",
    "tokens_of_cached_weight_set_batch",
)

OK, FAIL, NOTE = "ok", "fail", "note"


class Check:
    def __init__(self, name, status, detail, remedy=None, build_only=False):
        self.name = name
        self.status = status
        self.detail = detail
        self.remedy = remedy
        # build_only checks are what it takes to compile the extension. Once it
        # is compiled, or installed as a wheel, they are reported but do not
        # make the run fail: an installed-from-wheel environment legitimately
        # has no submodule, no rustc and no maturin.
        self.build_only = build_only


def _run(command):
    """Return the command's stdout, or None if it is missing or fails."""
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def check_submodules(root=REPO_ROOT):
    """The tokenizers submodule is required; sentencepiece is only needed for
    the sp training method."""
    root = Path(root)
    if not (root / ".gitmodules").exists():
        return [Check("submodules", NOTE,
                      "not a source checkout, nothing to initialise")]

    checks = []
    if (root / "tokenizers" / "tokenizers" / "Cargo.toml").exists():
        checks.append(Check("tokenizers submodule", OK, "checked out"))
    else:
        checks.append(Check(
            "tokenizers submodule", FAIL,
            "tokenizers/tokenizers/Cargo.toml is missing, so the submodule is "
            "not initialised; building now would fail inside Cargo with an "
            "error that does not mention submodules",
            "git submodule update --init tokenizers", build_only=True))

    if (root / "sentencepiece" / "CMakeLists.txt").exists():
        checks.append(Check("sentencepiece submodule", OK, "checked out"))
    else:
        checks.append(Check(
            "sentencepiece submodule", NOTE,
            "not initialised; needed only to build the spm_train binary for "
            "the sp training method",
            "git submodule update --init sentencepiece"))
    return checks


def _parse_rustc_version(output):
    match = re.search(r"rustc (\d+)\.(\d+)(?:\.(\d+))?", output)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups(default="0"))


def check_rust():
    output = _run(["rustc", "--version"])
    if output is None:
        return Check(
            "rust toolchain", FAIL,
            "rustc was not found on PATH",
            "install a Rust toolchain from https://rustup.rs, then "
            "`rustup update stable`", build_only=True)

    version = _parse_rustc_version(output)
    if version is None:
        return Check("rust toolchain", FAIL,
                     f"could not read a version out of {output.strip()!r}",
                     "rustup update stable", build_only=True)

    printed = ".".join(str(part) for part in version)
    required = ".".join(str(part) for part in MIN_RUST_VERSION)
    if version[:2] < MIN_RUST_VERSION:
        return Check(
            "rust toolchain", FAIL,
            f"rustc {printed} is older than the {required} the bundled "
            f"tokenizers fork needs; a too-old toolchain fails the build with "
            f"a message about `?` weak dependency features and the nightly "
            f"channel, which is about the toolchain age, not about nightly",
            "rustup update stable", build_only=True)
    return Check("rust toolchain", OK, f"rustc {printed}")


def check_maturin():
    if shutil.which("maturin") is None:
        return Check("maturin", FAIL, "not found on PATH",
                     "pip install maturin", build_only=True)
    return Check("maturin", OK, "found on PATH")


def check_extension():
    try:
        from tokenizers.models import Unigram
    except Exception as exc:
        return Check(
            "tokenizers extension", FAIL,
            f"importing the tokenizers extension failed: {exc}",
            "cd tokenizers/bindings/python && maturin develop --release")

    missing = [name for name in SCORER_METHODS if not hasattr(Unigram, name)]
    if missing:
        return Check(
            "tokenizers extension", FAIL,
            f"the installed tokenizers is missing {missing}; this is either "
            f"the stock HuggingFace package or an older build of the fork",
            "pip uninstall tokenizers -y && cd tokenizers/bindings/python && "
            "maturin develop --release")
    return Check("tokenizers extension", OK,
                 "imports and has all scorer methods")


def check_spm_train():
    """The sp training method needs two separate artifacts, and the second one
    fails in a way that is easy to misread.

    A `sentencepiece/` directory (the submodule) sits at the repository root, so
    when the pip package is absent `import sentencepiece` can still succeed, as
    a namespace package with none of the API. Test for the API, not the import.
    """
    missing = []
    if shutil.which("spm_train") is None:
        missing.append("the spm_train binary is not on PATH")
    try:
        import sentencepiece as spm
        has_package = hasattr(spm, "SentencePieceProcessor")
    except ImportError:
        has_package = False
    if not has_package:
        missing.append("the sentencepiece Python package is not installed")

    if missing:
        return Check(
            "sentencepiece", NOTE,
            f"{'; '.join(missing)}. Needed only for the sp training method "
            f"(unilid-add-language --method sp, "
            f"train.py --per-lang-counts-method sp) and for two tests, which "
            f"skip without it",
            "build the binary per the README's Training section, and "
            "pip install -e '.[train]' for the package")
    return Check("sentencepiece", OK, "binary and Python package both present")


def main(argv=None):
    checks = list(check_submodules())
    extension = check_extension()
    rust, maturin = check_rust(), check_maturin()
    checks.extend([rust, maturin, extension, check_spm_train()])

    label = {OK: "[ ok ]", FAIL: "[fail]", NOTE: "[note]"}
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{label[check.status]} {check.name.ljust(width)}  {check.detail}")
        if check.remedy and check.status != OK:
            print(f"{' ' * (len(label[OK]) + width + 3)}fix: {check.remedy}")

    # Anything only needed to compile the extension stops mattering once the
    # extension works, so those checks are reported but not fatal then.
    required = [c for c in checks
                if not c.build_only or extension.status == FAIL]
    failed = [c.name for c in required if c.status == FAIL]

    print()
    if failed:
        print(f"not ready: {', '.join(failed)}")
        return 1
    print("ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
