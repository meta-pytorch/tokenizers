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
from pathlib import Path
from typing import Sequence


def run_pytest(test_files: Sequence[Path], repo_root: Path) -> int:
    """Execute pytest on the provided files and propagate the return code."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        *(str(test) for test in test_files),
    ]

    print(f"Running pytest with: {' '.join(cmd)}")
    # Use repo_root as cwd so that relative paths to test resources
    # (e.g. "test/resources/...") resolve correctly.
    # Strip PYTHONPATH to prevent the repo's source tree from shadowing
    # the installed wheel. Python won't auto-add cwd to sys.path when
    # running via "python -m pytest".
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(cmd, cwd=str(repo_root), env=env, check=False)
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

    return run_pytest(sorted(test_dir.glob("test_*.py")), repo_root)


if __name__ == "__main__":
    sys.exit(main())
