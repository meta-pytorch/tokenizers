#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

# The reusable Python 3.15 smoke installer selects PyTorch's index.
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:+${PIP_EXTRA_INDEX_URL} }https://pypi.org/simple"
echo "PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}" >> "${GITHUB_ENV}"

UNAME_S=$(uname -s)

if [[ $UNAME_S == "Linux" && $(uname -m) == "aarch64" ]]; then
  # SentencePiece's source build needs the unversioned linker name.
  atomic_lib=$(cc -print-file-name=libatomic.so.1)
  if [[ -f "${atomic_lib}" ]]; then
    atomic_link_dir=$(mktemp -d)
    ln -s "${atomic_lib}" "${atomic_link_dir}/libatomic.so"
    export LIBRARY_PATH="${atomic_link_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    echo "LIBRARY_PATH=${LIBRARY_PATH}" >> "${GITHUB_ENV}"
  fi
fi

if [[ $UNAME_S == "Darwin" ]]; then
  python - <<'PYTHON'
import os
import subprocess
import sys

if sys.version_info >= (3, 15):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "setuptools>=63", "pybind11>=3.1.0"],
        check=True,
    )
    # SentencePiece 0.2.2 hardcodes a macOS 10.9 compiler target.
    sentencepiece_env = os.environ.copy()
    sentencepiece_env["CXXFLAGS"] = (
        f"{sentencepiece_env.get('CXXFLAGS', '')} -D_LIBCPP_DISABLE_AVAILABILITY"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "sentencepiece==0.2.2",
        ],
        env=sentencepiece_env,
        check=True,
    )
PYTHON
fi

# On Windows, enable symlinks and re-checkout the current revision to create
# symlinked directories needed for the wheel build.
if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
    echo "Enabling symlinks on Windows"
    git config core.symlinks true
    git checkout -f HEAD
fi
