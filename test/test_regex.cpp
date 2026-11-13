/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "pytorch/tokenizers/pcre2_regex.h"
#include "pytorch/tokenizers/re2_regex.h"
#include "pytorch/tokenizers/regex.h"

using namespace tokenizers;

class RegexTest : public ::testing::Test {};

// Test basic functionality
TEST_F(RegexTest, BasicMatching) {
  auto regex = TK_UNWRAP_THROW(create_regex("\\w+"));

  std::string text = "Hello world";
  auto matches = regex->find_all(text);
  ASSERT_EQ(matches.size(), 2);
  EXPECT_EQ(matches[0].start, 0);
  EXPECT_EQ(matches[0].end, 5);
  EXPECT_EQ(
      text.substr(matches[0].start, matches[0].end - matches[0].start),
      "Hello");
  EXPECT_EQ(matches[1].start, 6);
  EXPECT_EQ(matches[1].end, 11);
  EXPECT_EQ(
      text.substr(matches[1].start, matches[1].end - matches[1].start),
      "world");
}

// Test pattern that only PCRE2 supports (lookbehind)
TEST_F(RegexTest, Pcre2Specific) {
  const std::string pattern = "(?<=@)\\w+";

  // Verify that the factory function fallsback on a PCRE2 regex
  auto regex = TK_UNWRAP_THROW(create_regex(pattern));
  EXPECT_NE(dynamic_cast<Pcre2Regex*>(regex.get()), nullptr);

  std::string text = "user@example.com";
  auto matches = regex->find_all(text);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].start, 5);
  EXPECT_EQ(matches[0].end, 12);
  EXPECT_EQ(
      text.substr(matches[0].start, matches[0].end - matches[0].start),
      "example");
}

// Test that PCRE2 can handle patterns with invalid UTF-8 byte sequences
// by retrying without UTF-8 validation. This is important for Qwen tokenizers
// which contain patterns with byte sequences that aren't valid UTF-8.
TEST_F(RegexTest, InvalidUtf8PatternHandling) {
  // This pattern contains an invalid UTF-8 sequence (0xC0 0x80 is an overlong
  // encoding of the null character, which is not valid UTF-8)
  const std::string pattern_with_invalid_utf8 = "test\xC0\x80pattern";

  // The pattern should compile successfully using PCRE2 without UTF-8
  // validation
  Pcre2Regex regex;
  Error result = regex.compile(pattern_with_invalid_utf8);
  EXPECT_EQ(result, Error::Ok);

  // Verify that the compiled regex can actually match text
  const std::string text_with_same_sequence = "test\xC0\x80pattern here";
  auto matches = regex.find_all(text_with_same_sequence);
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].start, 0);
  EXPECT_EQ(matches[0].end, 13); // Length of "test\xC0\x80pattern"
}

// Test complex pattern with negative lookahead that should fall back to PCRE2.
// This specific pattern is from the Qwen2.5 1.5B pretokenizer.
// https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/tokenizer.json
TEST_F(RegexTest, ComplexPatternWithNegativeLookahead) {
  const std::string complex_pattern =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

  // Now verify that the factory function fallsback on a PCRE2 regex
  auto regex = TK_UNWRAP_THROW(create_regex(complex_pattern));
  EXPECT_NE(dynamic_cast<Pcre2Regex*>(regex.get()), nullptr);

  // Test the pattern with some sample text
  std::string text = "Hello's world\n  test";
  auto matches = regex->find_all(text);

  // We expect to match:
  // 1. "Hello" (word)
  // 2. "'s" (contraction)
  // 3. " world" (word with leading space)
  // 4. "\n" (newline)
  // 5. " " (whitespace)
  // 6. " test" (word with leading space)
  ASSERT_EQ(matches.size(), 6);

  EXPECT_EQ(matches[0].start, 0);
  EXPECT_EQ(matches[0].end, 5);
  EXPECT_EQ(
      text.substr(matches[0].start, matches[0].end - matches[0].start),
      "Hello");
  EXPECT_EQ(matches[1].start, 5);
  EXPECT_EQ(matches[1].end, 7);
  EXPECT_EQ(
      text.substr(matches[1].start, matches[1].end - matches[1].start), "'s");
  EXPECT_EQ(matches[2].start, 7);
  EXPECT_EQ(matches[2].end, 13);
  EXPECT_EQ(
      text.substr(matches[2].start, matches[2].end - matches[2].start),
      " world");
  EXPECT_EQ(matches[3].start, 13);
  EXPECT_EQ(matches[3].end, 14);
  EXPECT_EQ(
      text.substr(matches[3].start, matches[3].end - matches[3].start), "\n");
  EXPECT_EQ(matches[4].start, 14);
  EXPECT_EQ(matches[4].end, 15);
  EXPECT_EQ(
      text.substr(matches[4].start, matches[4].end - matches[4].start), " ");
  EXPECT_EQ(matches[5].start, 15);
  EXPECT_EQ(matches[5].end, 20);
  EXPECT_EQ(
      text.substr(matches[5].start, matches[5].end - matches[5].start),
      " test");
}

// A long text with many multi-byte characters gives the same matches as each
// of its chunks on its own.
TEST_F(RegexTest, Pcre2LongMultiByteText) {
  const std::string pattern =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  auto regex = TK_UNWRAP_THROW(create_regex(pattern));
  EXPECT_NE(dynamic_cast<Pcre2Regex*>(regex.get()), nullptr);

  const std::string chunk =
      "Hello w\xC3\xB6rld, \xE4\xBD\xA0\xE5\xA5\xBD \xF0\x9F\x99\x82 it's 42\n";
  const auto chunk_matches = regex->find_all(chunk);
  ASSERT_FALSE(chunk_matches.empty());

  constexpr size_t kRepeats = 1000;
  std::string text;
  for (size_t i = 0; i < kRepeats; ++i) {
    text += chunk;
  }
  const auto matches = regex->find_all(text);
  ASSERT_EQ(matches.size(), chunk_matches.size() * kRepeats);
  for (size_t i = 0; i < matches.size(); ++i) {
    const auto& expected = chunk_matches[i % chunk_matches.size()];
    const size_t shift = (i / chunk_matches.size()) * chunk.size();
    EXPECT_EQ(matches[i].start, expected.start + shift);
    EXPECT_EQ(matches[i].end, expected.end + shift);
  }
}

// After an empty match the next search has to start at the next character,
// not inside a multi-byte one.
TEST_F(RegexTest, Pcre2EmptyMatchBeforeMultiByteCharacter) {
  Pcre2Regex regex;
  ASSERT_EQ(regex.compile("x*"), Error::Ok);

  // "\xC4\x85" is 2 bytes, "\xE4\xBD\xA0" 3 and "\xF0\x9F\x99\x82" 4.
  const std::string text =
      "\xC4\x85"
      "b\xE4\xBD\xA0\xF0\x9F\x99\x82"
      "c";
  const auto matches = regex.find_all(text);
  const std::vector<size_t> expected_starts = {0, 2, 3, 6, 10};
  ASSERT_EQ(matches.size(), expected_starts.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    EXPECT_EQ(matches[i].start, expected_starts[i]);
    EXPECT_EQ(matches[i].end, expected_starts[i]);
  }
}

// Later calls skip PCRE2's UTF-8 check, so the first call has to reject the
// whole text when any part of it is invalid UTF-8, here a 3-byte character cut
// off at the end, even though the text starts with valid matches.
TEST_F(RegexTest, Pcre2InvalidUtf8AfterValidMatches) {
  Pcre2Regex regex;
  ASSERT_EQ(regex.compile("\\p{L}+| "), Error::Ok);
  EXPECT_TRUE(regex.find_all("Hello w\xC3\xB6rld \xE4\xBD").empty());
}

// RE2 compiles \s as ASCII whitespace only, but the HuggingFace tokenizers
// library treats every Unicode White_Space code point as \s. CLIP's normalizer
// (`\s+` -> " ") and pre-tokenizer (`[^\s\p{L}\p{N}]+`) both go through RE2.
TEST_F(RegexTest, Re2UnicodeWhitespace) {
  // U+00A0 NO-BREAK SPACE, U+3000 IDEOGRAPHIC SPACE, U+200B ZERO WIDTH SPACE.
  const std::string nbsp = "\xC2\xA0";
  const std::string ideographic = "\xE3\x80\x80";
  const std::string zwsp = "\xE2\x80\x8B";

  auto space = TK_UNWRAP_THROW(create_regex("\\s+"));
  EXPECT_NE(dynamic_cast<Re2Regex*>(space.get()), nullptr);
  // " ", "\t", U+000B, U+0085, U+00A0, U+2028 and U+3000.
  const std::string spaces =
      " \t\v\xC2\x85" + nbsp + "\xE2\x80\xA8" + ideographic;
  auto matches = space->find_all("a" + spaces + "b");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].start, 1);
  EXPECT_EQ(matches[0].end, 1 + spaces.size());
  // U+200B is not White_Space.
  EXPECT_TRUE(space->find_all("a" + zwsp + "b").empty());

  auto not_space = TK_UNWRAP_THROW(create_regex("\\S+"));
  EXPECT_NE(dynamic_cast<Re2Regex*>(not_space.get()), nullptr);
  EXPECT_EQ(not_space->find_all("a" + nbsp + "b").size(), 2);

  // Inside a class, as in CLIP's pre-tokenizer.
  auto symbols = TK_UNWRAP_THROW(create_regex("[^\\s\\p{L}]+"));
  EXPECT_NE(dynamic_cast<Re2Regex*>(symbols.get()), nullptr);
  matches = symbols->find_all("a" + nbsp + "!" + ideographic + "b");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].start, 1 + nbsp.size());
  EXPECT_EQ(matches[0].end, 2 + nbsp.size());

  // A POSIX class inside the class does not end it early.
  auto posix = TK_UNWRAP_THROW(create_regex("[[:digit:]\\s]+"));
  EXPECT_NE(dynamic_cast<Re2Regex*>(posix.get()), nullptr);
  EXPECT_EQ(posix->find_all("1" + nbsp + "2").size(), 1);

  // An escaped backslash followed by "s" is still a literal backslash and "s".
  auto literal = TK_UNWRAP_THROW(create_regex("\\\\s"));
  EXPECT_NE(dynamic_cast<Re2Regex*>(literal.get()), nullptr);
  matches = literal->find_all("\\s s");
  ASSERT_EQ(matches.size(), 1);
  EXPECT_EQ(matches[0].start, 0);
  EXPECT_EQ(matches[0].end, 2);
}
