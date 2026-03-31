#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

UNAME_S=$(uname -s)

# On Linux aarch64, the 'atomic' library may not be found during linking.
# Replace the hardcoded "atomic" with the full path to libatomic so the
# build system can locate it.
if [[ $(uname -m) == "aarch64" ]]; then
  ATOMIC_LIB=$(find / -name "libatomic.so" 2>/dev/null | head -1)
  if [[ -n "${ATOMIC_LIB}" ]]; then
    echo "Found libatomic at ${ATOMIC_LIB}"
    # Create a symlink in a standard library path so -latomic works
    ln -sf "${ATOMIC_LIB}" /usr/lib/libatomic.so || true
  fi
fi

# On Windows, enable symlinks and re-checkout the current revision to create
# symlinked directories needed for the wheel build.
if [[ $UNAME_S == *"MINGW"* || $UNAME_S == *"MSYS"* ]]; then
    echo "Enabling symlinks on Windows"
    git config core.symlinks true
    git checkout -f HEAD
fi
