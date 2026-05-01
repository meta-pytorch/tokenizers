/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Stress test for concurrent invocation of HFTokenizer::encode().
//
// Background: HF tokenizer pre-tokenizers commonly use lookahead patterns
// (e.g. `\s+(?!\S)` in the Llama-3 / GPT-2 byte-level BPE split regex).
// RE2 rejects these patterns, so the regex backend falls back to PCRE2
// (see src/regex_lookahead.cpp). Previously, Pcre2Regex stored
// `pcre2_match_data*` as a class member shared across all calls, which
// silently corrupted match offsets under concurrent find_all() invocations
// and could trigger heap-buffer-overflow inside PCRE2.
//
// The fix in this diff allocates match_data per find_all() call. This test
// guards against regression and also exercises a SynchronizedTokenizer
// helper (callers that need an extra correctness guarantee, e.g. wrapping
// a tokenizer that may be backed by other future non-thread-safe regex
// engines).

#include <gtest/gtest.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tokenizers {
namespace {

std::string get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}

constexpr int kNumThreads = 8;
constexpr int kIterationsPerThread = 100;

// Long enough to maximize the number of regex match operations per encode()
// call, widening the race window.
const std::string kStressText =
    "Hello world! Hello world! Hello world! Hello world! "
    "Hello world! Hello world! Hello world! Hello world! "
    "Hello world! Hello world! Hello world! Hello world! "
    "Hello world! Hello world! Hello world! Hello world!";

// Run kNumThreads x kIterationsPerThread invocations of encode_fn(text), all
// racing on the same tokenizer state. Returns the number of results that
// differ from a single-threaded reference encoding.
template <typename EncodeFn>
size_t run_stress(EncodeFn encode_fn, const std::string& text) {
  auto ref_result = encode_fn(text);
  if (!ref_result.ok()) {
    ADD_FAILURE() << "Reference encode failed";
    return 0;
  }
  const auto reference = ref_result.get();

  std::atomic<size_t> mismatches{0};
  std::atomic<bool> ready{false};
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&]() {
      while (!ready.load(std::memory_order_acquire)) {
        // Spin until all threads are spawned, so they start hammering
        // simultaneously.
      }
      for (int i = 0; i < kIterationsPerThread; ++i) {
        auto r = encode_fn(text);
        if (!r.ok() || r.get() != reference) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  ready.store(true, std::memory_order_release);
  for (auto& th : threads) {
    th.join();
  }
  return mismatches.load();
}

// Minimal mutex wrapper that serializes all encode() calls on a tokenizer.
// This is the workaround intended for use until the underlying regex backend
// is made thread-safe (see Pcre2Regex::find_all).
class SynchronizedTokenizer {
 public:
  explicit SynchronizedTokenizer(Tokenizer* tok) : tok_(tok) {}

  Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t bos, int8_t eos) const {
    std::lock_guard<std::mutex> lock(mu_);
    return tok_->encode(text, bos, eos);
  }

 private:
  Tokenizer* tok_;
  mutable std::mutex mu_;
};

} // namespace

// Regression guard: concurrent encode() on an HF tokenizer that uses
// lookahead patterns must produce identical token sequences across threads.
//
// Before the Pcre2Regex fix that allocates match_data per find_all() call,
// this test crashed with AddressSanitizer heap-buffer-overflow (PCRE2's own
// "internal error - pattern overwritten?" diagnostic) within a few hundred
// iterations.
TEST(ConcurrentEncodeTest, BareEncodeIsThreadSafeOnLookaheadTokenizer) {
  HFTokenizer tokenizer;
  auto path = get_resource_path("hf_tokenizer_dir/");
  ASSERT_EQ(tokenizer.load(path), Error::Ok);

  auto encode_fn = [&](const std::string& s) {
    return tokenizer.encode(s, /*bos=*/0, /*eos=*/0);
  };

  size_t mismatches = run_stress(encode_fn, kStressText);
  EXPECT_EQ(mismatches, 0u)
      << "Concurrent encode() on a lookahead-using HF tokenizer produced "
         "inconsistent results. This indicates a regression in Pcre2Regex's "
         "thread-safety (match_data must be allocated per find_all() call).";
}

// Verifies the workaround: serializing encode() with std::mutex makes
// concurrent invocation produce identical results.
TEST(ConcurrentEncodeTest, MutexWrappedEncodeIsConsistent) {
  HFTokenizer tokenizer;
  auto path = get_resource_path("hf_tokenizer_dir/");
  ASSERT_EQ(tokenizer.load(path), Error::Ok);

  SynchronizedTokenizer sync_tok(&tokenizer);
  auto encode_fn = [&](const std::string& s) {
    return sync_tok.encode(s, /*bos=*/0, /*eos=*/0);
  };

  size_t mismatches = run_stress(encode_fn, kStressText);
  EXPECT_EQ(mismatches, 0u)
      << "With std::mutex serializing encode(), all threads must produce "
         "identical token sequences.";
}

} // namespace tokenizers
