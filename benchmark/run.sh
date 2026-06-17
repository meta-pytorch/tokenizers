#!/usr/bin/env bash
# Configure (Release) + build + run the encode latency benchmark.
#
#   run.sh <tokenizer.json> [reps]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tokenizer="${1:?usage: run.sh <tokenizer.json> [reps]}"
shift || true

build_dir="${here}/build"
cmake -S "${here}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${build_dir}" --target hf_tokenizer_encode_latency --parallel

"${build_dir}/hf_tokenizer_encode_latency" "${tokenizer}" "$@"
