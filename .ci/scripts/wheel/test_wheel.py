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


def run_pytest(test_files: Sequence[Path], cwd: Path) -> int:
    """Execute pytest on the provided files and propagate the return code."""
    env = os.environ.copy()
    pythonpath_entries = [str(cwd)]
    if existing_path := env.get("PYTHONPATH"):
        pythonpath_entries.append(existing_path)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        *(str(test) for test in test_files),
    ]

    print(f"Running pytest with: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    return result.returncode


def ensure_pytest() -> None:
    """Install pytest if it is not already available."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest not found, installing...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pytest"],
            check=True,
        )


def main() -> int:
    ensure_pytest()

    repo_root = Path(__file__).resolve().parents[3]
    test_dir = repo_root / "test"

    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}", file=sys.stderr)
        return 1

    return run_pytest([test_dir], repo_root)


if __name__ == "__main__":
    sys.exit(main())
