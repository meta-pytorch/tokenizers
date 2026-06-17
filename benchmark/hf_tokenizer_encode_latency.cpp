/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Latency benchmark for HFTokenizer::encode. Encodes prompt templates
// (prose/code/dialogue) repeated to growing lengths and prints the mean wall
// time per encode. Correctness is covered by the unit tests; this only measures
// performance.
//
//   hf_tokenizer_encode_latency <tokenizer.json> [reps]

#include <pytorch/tokenizers/hf_tokenizer.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using tokenizers::HFTokenizer;

namespace {

struct Vector {
  std::string name;
  std::string base; // repeated to reach the target length
  size_t repeats;
};

std::vector<Vector> vectors() {
  return {
      {"prose_0.5k",
       "The quick brown fox jumps over the lazy dog near the riverbank. ",
       35},
      {"code_1k",
       "for (int i = 0; i < n; ++i) { sum += arr[i] * weights[i]; } // accumulate\n",
       61},
      {"dialogue_1.5k",
       "User: Can you summarize the previous section in two sentences please? "
       "Assistant: Sure, here is a concise summary of the main points raised. ",
       48},
      {"prose_2k",
       "In the beginning the system parsed every token sequentially, which made "
       "long prompts increasingly expensive to process as length grew larger. ",
       63},
      {"prose_8k",
       "In the beginning the system parsed every token sequentially, which made "
       "long prompts increasingly expensive to process as length grew larger. ",
       348},
  };
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::fprintf(stderr, "usage: %s <tokenizer.json> [reps]\n", argv[0]);
    return 2;
  }
  long reps = 5;
  if (argc > 2) {
    char* end = nullptr;
    errno = 0;
    reps = std::strtol(argv[2], &end, 10);
    if (*end != '\0' || errno == ERANGE || reps <= 0) {
      std::fprintf(stderr, "reps must be a positive integer, got '%s'\n", argv[2]);
      return 2;
    }
  }

  HFTokenizer tok;
  if (tok.load(argv[1]) != tokenizers::Error::Ok) {
    std::fprintf(stderr, "failed to load tokenizer: %s\n", argv[1]);
    return 1;
  }

  std::printf(
      "%-14s %10s %9s %12s %12s\n",
      "vector",
      "chars",
      "ids",
      "mean_ms",
      "ns/token");
  for (const auto& v : vectors()) {
    std::string text;
    text.reserve(v.base.size() * v.repeats);
    for (size_t i = 0; i < v.repeats; ++i) {
      text += v.base;
    }

    auto warm = tok.encode(text, 0, 0); // also warms up
    if (!warm.ok()) {
      std::fprintf(
          stderr,
          "encode failed for %s (error %d)\n",
          v.name.c_str(),
          static_cast<int>(warm.error()));
      return 1;
    }
    const size_t ids = warm.get().size();

    double total_ms = 0;
    for (long r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      auto res = tok.encode(text, 0, 0);
      const auto t1 = std::chrono::steady_clock::now();
      if (!res.ok()) {
        std::fprintf(
            stderr,
            "encode failed for %s (error %d)\n",
            v.name.c_str(),
            static_cast<int>(res.error()));
        return 1;
      }
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const double mean_ms = total_ms / reps;
    std::printf(
        "%-14s %10zu %9zu %12.3f %12.1f\n",
        v.name.c_str(),
        text.size(),
        ids,
        mean_ms,
        ids ? mean_ms * 1e6 / static_cast<double>(ids) : 0.0);
  }
  return 0;
}
