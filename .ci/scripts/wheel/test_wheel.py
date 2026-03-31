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
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


def run_pytest(test_files: Sequence[Path]) -> int:
    """Execute pytest on the provided files and propagate the return code."""
    # Run from a temp directory to avoid importing the source tree's
    # pytorch_tokenizers/ instead of the installed wheel package.
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-vv",
            *(str(test) for test in test_files),
        ]

        print(f"Running pytest with: {' '.join(cmd)}")
        # Strip PYTHONPATH to avoid any repo root entries that could
        # shadow the installed package.
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(cmd, cwd=tmpdir, env=env, check=False)
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

    return run_pytest(sorted(test_dir.glob("test_*.py")))


if __name__ == "__main__":
    sys.exit(main())
