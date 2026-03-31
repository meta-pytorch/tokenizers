#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cross-platform wheel smoke-test entry point.

The PyTorch Tokenizers wheel should already be installed in the active
environment when this script runs. We execute the Python unit test suite
to ensure the wheel exposes the expected bindings and core functionality.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


def run_pytest(test_files: Sequence[Path], cwd: Path) -> int:
    """Execute pytest on the provided files and propagate the return code."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        *(str(test) for test in test_files),
    ]

    print(f"Running pytest from: {cwd}")
    print(f"Running pytest with: {' '.join(cmd)}")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return result.returncode


def ensure_dependencies() -> None:
    """Install test dependencies if not already available."""
    deps = ["pytest", "transformers"]
    missing = []
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        print(f"Installing missing test dependencies: {missing}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing],
            check=True,
        )


def main() -> int:
    ensure_dependencies()

    repo_root = Path(__file__).resolve().parents[3]
    test_dir = repo_root / "test"

    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}", file=sys.stderr)
        return 1

    # Run from a temp directory to avoid importing the source tree's
    # pytorch_tokenizers/ package instead of the installed wheel.
    # Copy the test/ directory (including resources/) so that relative
    # paths like "test/resources/..." resolve correctly.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        shutil.copytree(test_dir, tmp_path / "test")

        test_files = sorted((tmp_path / "test").glob("test_*.py"))
        return run_pytest(test_files, tmp_path)


if __name__ == "__main__":
    sys.exit(main())
