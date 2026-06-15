/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Third Party
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <re2/re2.h>

#include <atomic>
#include <thread>
#include <vector>

// Local
#include <pytorch/tokenizers/pre_tokenizer.h>

using json = nlohmann::json;
using namespace tokenizers;

// Helpers /////////////////////////////////////////////////////////////////////

static void assert_split_match(
    const PreTokenizer& ptok,
    const std::string& prompt,
    const std::vector<std::string>& expected) {
  const auto& got = ptok.pre_tokenize(prompt);
  EXPECT_EQ(expected.size(), got.size());
  for (auto i = 0; i < got.size(); ++i) {
    EXPECT_EQ(expected[i], got[i]);
  }
}

// RegexPreTokenizer ///////////////////////////////////////////////////////////
class RegexPreTokenizerTest : public ::testing::Test {};

// Test the basic construction
TEST_F(RegexPreTokenizerTest, Construct) {
  RegexPreTokenizer ptok("[0-9]+");
}

// Test basic splitting using the expression for Tiktoken
TEST_F(RegexPreTokenizerTest, TiktokenExpr) {
  RegexPreTokenizer ptok(
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)");
  assert_split_match(
      ptok, "How are you doing?", {"How", " are", " you", " doing", "?"});
}

// DigitsPreTokenizer //////////////////////////////////////////////////////////
class DigitsPreTokenizerTest : public ::testing::Test {};

// Test digit splitting with individual digits
TEST_F(DigitsPreTokenizerTest, IndividualDigits) {
  DigitsPreTokenizer ptok(true);
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "2", "3", "4", " then ", "5", "."});
}

// Test digit splitting with contiguous digits
TEST_F(DigitsPreTokenizerTest, ContiguousDigits) {
  DigitsPreTokenizer ptok(false);
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "234", " then ", "5", "."});
}

// ByteLevelPreTokenizer ///////////////////////////////////////////////////////
class ByteLevelPreTokenizerTest : public ::testing::Test {};

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeDefault) {
  ByteLevelPreTokenizer ptok;
  assert_split_match(ptok, "Hello World", {"ĠHello", "ĠWorld"});
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"ĠThe", "Ġnumber", "Ġ1", "Ġthen", "Ġ234", "Ġthen", "Ġ5", "."});
}

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeNoPrefix) {
  ByteLevelPreTokenizer ptok(false);
  assert_split_match(ptok, "Hello World", {"Hello", "ĠWorld"});
}

TEST_F(ByteLevelPreTokenizerTest, PreTokenizeCustomRegex) {
  ByteLevelPreTokenizer ptok(false, R"(o)");
  assert_split_match(ptok, "Hello World", {"Hell", "o", "ĠW", "o", "rld"});
}

// SequencePreTokenizer ////////////////////////////////////////////////////////
class SequencePreTokenizerTest : public ::testing::Test {};

TEST_F(SequencePreTokenizerTest, PreTokenizeDigitAndByteLevel) {
  PreTokenizer::Ptr dptok(new DigitsPreTokenizer(true));
  PreTokenizer::Ptr bptok(new ByteLevelPreTokenizer(false));
  SequencePreTokenizer ptok({dptok, bptok});
  assert_split_match(
      ptok,
      "The number 1 then 234 then 5.",
      {"The",
       "Ġnumber",
       "Ġ",
       "1",
       "Ġthen",
       "Ġ",
       "2",
       "3",
       "4",
       "Ġthen",
       "Ġ",
       "5",
       "."});
}

// PreTokenizerConfig //////////////////////////////////////////////////////////
//
// NOTE: When adding a new pre-tokenizer or changing arguments, add it to these
//  tests!
class PreTokenizerConfigTest : public ::testing::Test {};

TEST_F(PreTokenizerConfigTest, AllTypesSuccess) {
  // Regex
  PreTokenizerConfig("Split").set_pattern(R"(o)").create();

  // Digits
  PreTokenizerConfig("Digits").create();
  PreTokenizerConfig("Digits").set_individual_digits(true).create();
  PreTokenizerConfig("Digits").set_individual_digits(false).create();

  // ByteLevel
  PreTokenizerConfig("ByteLevel").create();
  PreTokenizerConfig("ByteLevel").set_pattern(R"(o)").create();
  PreTokenizerConfig("ByteLevel").set_add_prefix_space(true).create();
  PreTokenizerConfig("ByteLevel")
      .set_add_prefix_space(false)
      .set_pattern(R"(o)")
      .create();

  // Sequence
  PreTokenizerConfig("Sequence")
      .set_pretokenizers(
          {PreTokenizerConfig("Digits"), PreTokenizerConfig("ByteLevel")})
      .create();
}

TEST_F(PreTokenizerConfigTest, AllTypesFailureCases) {
  // Regex
  EXPECT_THROW(PreTokenizerConfig("Split").create(), std::runtime_error);

  // Sequence
  EXPECT_THROW(PreTokenizerConfig("Sequence").create(), std::runtime_error);
  EXPECT_THROW(
      PreTokenizerConfig("Sequence").set_pretokenizers({}).create(),
      std::runtime_error);
  EXPECT_THROW(
      PreTokenizerConfig("Sequence")
          .set_pretokenizers({PreTokenizerConfig("Split")})
          .create(),
      std::runtime_error);

  // Unsupported
  EXPECT_THROW(PreTokenizerConfig("Unsupported").create(), std::runtime_error);
}

TEST_F(PreTokenizerConfigTest, ParseJson) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Sequence"},
                                {"pretokenizers",
                                 json{
                                     json{
                                         {"type", "Digits"},
                                         {"individual_digits", true},
                                     },
                                     json{
                                         {"type", "ByteLevel"},
                                         {"add_prefix_space", false},
                                     },
                                 }},
                            })
                        .create();
  assert_split_match(
      *ptok,
      "The number 1 then 234 then 5.",
      {"The",
       "Ġnumber",
       "Ġ",
       "1",
       "Ġthen",
       "Ġ",
       "2",
       "3",
       "4",
       "Ġthen",
       "Ġ",
       "5",
       "."});
}

TEST_F(PreTokenizerConfigTest, ParseJsonOptionalKey) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Digits"},
                            })
                        .create();
  assert_split_match(
      *ptok,
      "The number 1 then 234 then 5.",
      {"The number ", "1", " then ", "234", " then ", "5", "."});
}

TEST_F(PreTokenizerConfigTest, Split) {
  PreTokenizerConfig config;
  const auto ptok =
      config
          .parse_json(
              json{
                  {"type", "Split"},
                  {"pattern",
                   {{"Regex",
                     R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+)"}}},
              })
          .create();
  assert_split_match(*ptok, "Hello World", {"Hello", " World"});
}

TEST_F(PreTokenizerConfigTest, SplitWithStringPattern) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", " "}}},
                            })
                        .create();
  assert_split_match(*ptok, "Hello world!", {"Hello", "world!"});
}

TEST_F(PreTokenizerConfigTest, SplitWithStringPatternSpecialChars) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "."}}},
                            })
                        .create();
  assert_split_match(*ptok, "Hello.world.test", {"Hello", "world", "test"});
}

TEST_F(PreTokenizerConfigTest, SplitWithStringPatternNoMatches) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "xyz"}}},
                            })
                        .create();
  assert_split_match(*ptok, "Hello world", {"Hello world"});
}

TEST_F(PreTokenizerConfigTest, SplitWithRegexMetaCharacters) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "+"}}},
                            })
                        .create();
  assert_split_match(*ptok, "a+b+c", {"a", "b", "c"});
}

TEST_F(PreTokenizerConfigTest, SplitWithRegexBrackets) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "["}}},
                            })
                        .create();
  assert_split_match(*ptok, "a[b[c", {"a", "b", "c"});
}

TEST_F(PreTokenizerConfigTest, SplitEmptyInput) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", " "}}},
                            })
                        .create();
  assert_split_match(*ptok, "", {""});
}

TEST_F(PreTokenizerConfigTest, SplitSingleCharacterInput) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", " "}}},
                            })
                        .create();
  assert_split_match(*ptok, "a", {"a"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithPrevious) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "-"}}},
                                {"behavior", "MergedWithPrevious"},
                                {"invert", false},
                            })
                        .create();
  // Example from docstring: "the-final--countdown" with delimiter "-"
  // -> ["the-", "final-", "-", "countdown"]
  assert_split_match(
      *ptok, "the-final--countdown", {"the-", "final-", "-", "countdown"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithPreviousSpaces) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", " "}}},
                                {"behavior", "MergedWithPrevious"},
                                {"invert", false},
                            })
                        .create();
  assert_split_match(*ptok, "Hello world test", {"Hello ", "world ", "test"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithPreviousStartingDelimiter) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "-"}}},
                                {"behavior", "MergedWithPrevious"},
                                {"invert", false},
                            })
                        .create();
  assert_split_match(*ptok, "-hello-world", {"-", "hello-", "world"});
}

TEST_F(PreTokenizerConfigTest, SplitWithMergedWithPreviousEndingDelimiter) {
  PreTokenizerConfig config;
  const auto ptok = config
                        .parse_json(
                            json{
                                {"type", "Split"},
                                {"pattern", {{"String", "-"}}},
                                {"behavior", "MergedWithPrevious"},
                                {"invert", false},
                            })
                        .create();
  assert_split_match(*ptok, "hello-world-", {"hello-", "world-"});
}

TEST_F(PreTokenizerConfigTest, SplitWithUnsupportedBehavior) {
  PreTokenizerConfig config;
  EXPECT_THROW(
      config
          .parse_json(
              json{
                  {"type", "Split"},
                  {"pattern", {{"String", "-"}}},
                  {"behavior", "MergedWithNext"},
                  {"invert", false},
              })
          .create(),
      std::runtime_error);
}

// Regex cache (unicode_regex_split) ///////////////////////////////////////////
// The ByteLevel pre-tokenizer drives unicode_regex_split, which caches the
// compiled std::regex/std::wregex per pattern. This test guards that cache:
// (1) results are deterministic / behavior-neutral, including across the full
// Unicode White_Space class and near-miss non-whitespace codepoints, and
// (2) it is thread-safe (the tokenizer pool calls it concurrently).
class RegexCacheTest : public ::testing::Test {};

TEST_F(RegexCacheTest, ByteLevelDeterministicAndThreadSafe) {
  ByteLevelPreTokenizer ptok(/*add_prefix_space=*/false);

  const std::vector<std::string> corpus = {
      "Hello World",
      "  leading and   multiple   spaces  ",
      "tabs\t\tand\nnewlines\r\n",
      "code: { return x; }   // trailing   ",
      // Unicode White_Space codepoints (must be treated as whitespace).
      "a\xC2\x85"
      "b", // U+0085 NEL
      "a\x0B"
      "b\x0C"
      "c", // U+000B VT, U+000C FF
      "a\xE1\x9A\x80"
      "b", // U+1680 OGHAM SPACE MARK
      "a\xE2\x80\x80\xE2\x80\x8A"
      "b", // U+2000 .. U+200A
      "a\xE2\x80\xAF"
      "b\xE2\x81\x9F"
      "c", // U+202F NNBSP, U+205F MMSP
      "a\xE3\x80\x80"
      "b", // U+3000 ideographic space
      // Near-miss NON-whitespace codepoints (must NOT be treated as ws).
      "a\xE2\x80\x8B"
      "b", // U+200B ZERO WIDTH SPACE
      "a\xE1\xA0\x8E"
      "b", // U+180E MONGOLIAN VOWEL SEPARATOR
      "a\xEF\xBB\xBF"
      "b", // U+FEFF BOM / ZWNBSP
      // Unicode letters and a representative SID prompt fragment.
      "caf\xC3\xA9 \xE4\xBD\xA0\xE5\xA5\xBD",
      "history: i0:<1326-617-1617> i1:<197-296-385> next:",
  };

  // Single-threaded reference.
  std::vector<std::vector<std::string>> ref;
  ref.reserve(corpus.size());
  for (const auto& s : corpus) {
    ref.push_back(ptok.pre_tokenize(s));
    EXPECT_FALSE(ref.back().empty());
  }

  // Hammer the shared pre-tokenizer (and the static regex cache inside
  // unicode_regex_split) from many threads; every result must match the
  // reference. A recompile race or a wrong cache lookup would surface as a
  // mismatch (or a crash under TSAN).
  constexpr int kThreads = 16;
  constexpr int kIters = 200;
  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&]() {
      for (int iter = 0; iter < kIters; ++iter) {
        for (size_t i = 0; i < corpus.size(); ++i) {
          if (ptok.pre_tokenize(corpus[i]) != ref[i]) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(0, mismatches.load());
}

TEST_F(PreTokenizerConfigTest, SplitWithInvertTrue) {
  PreTokenizerConfig config;
  EXPECT_THROW(
      config
          .parse_json(
              json{
                  {"type", "Split"},
                  {"pattern", {{"String", "-"}}},
                  {"behavior", "MergedWithPrevious"},
                  {"invert", true},
              })
          .create(),
      std::runtime_error);
}
