/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/normalizer.h>

using namespace tokenizers;

TEST(NormalizerTest, ReplaceNormalizerBasic) {
  // Test basic string replacement
  ReplaceNormalizer normalizer(" ", "▁");
  std::string input = "Hello World Test";
  std::string expected = "Hello▁World▁Test";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ReplaceNormalizerNoMatch) {
  // Test when pattern doesn't match
  ReplaceNormalizer normalizer("xyz", "▁");
  std::string input = "Hello World";
  std::string expected = "Hello World";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ReplaceNormalizerMultipleMatches) {
  // Test multiple matches
  ReplaceNormalizer normalizer("a", "X");
  std::string input = "banana";
  std::string expected = "bXnXnX";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ReplaceNormalizerEmptyContent) {
  // Empty replacement deletes every match.
  ReplaceNormalizer normalizer("a", "");
  EXPECT_EQ(normalizer.normalize("banana"), "bnn");
}

TEST(NormalizerTest, ReplaceNormalizerAtBoundaries) {
  // Matches at the very start and very end of the input.
  ReplaceNormalizer normalizer(" ", "_");
  EXPECT_EQ(normalizer.normalize(" a b "), "_a_b_");
}

TEST(NormalizerTest, ReplaceNormalizerConsecutiveMatches) {
  // Adjacent matches with a multi-byte (3-byte) replacement.
  ReplaceNormalizer normalizer(" ", "▁");
  EXPECT_EQ(normalizer.normalize("a   b"), "a▁▁▁b");
}

TEST(NormalizerTest, ReplaceNormalizerVariableSpanMultiCharContent) {
  // Variable-length matched spans with a multi-char replacement, including
  // spans at both boundaries.
  ReplaceNormalizer normalizer("\\s+", "__");
  EXPECT_EQ(normalizer.normalize(" a  b "), "__a__b__");
}

TEST(NormalizerTest, ReplaceNormalizerZeroWidthMatch) {
  // Zero-width (lookahead) match: insert content before each 'b' without
  // consuming input. Exercises match.start == match.end in the forward pass.
  ReplaceNormalizer normalizer("(?=b)", "X");
  EXPECT_EQ(normalizer.normalize("abc"), "aXbc");
}

TEST(NormalizerTest, PrependNormalizerBasic) {
  // Test basic prepending
  PrependNormalizer normalizer("_");
  std::string input = "Hello";
  std::string expected = "_Hello";
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, PrependNormalizerEmptyInput) {
  // Test prepend with empty input (should return empty)
  PrependNormalizer normalizer("_");
  std::string input;
  std::string expected;
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, NormalizerConfigPrepend) {
  // Test JSON parsing for Prepend normalizer
  nlohmann::json config = {{"type", "Prepend"}, {"prepend", "_"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello";
  std::string expected = "_Hello";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, NormalizerConfigFromJson) {
  // Test JSON parsing for Replace normalizer
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello World Test";
  std::string expected = "Hello▁World▁Test";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, NormalizerConfigFromJsonRegex) {
  // Test JSON parsing for Replace normalizer with regex
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"Regex", "\\s+"}}}, {"content", "_"}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "Hello   World\t\tTest";
  std::string expected = "Hello_World_Test";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, SequenceNormalizer) {
  // Test sequence of normalizers
  std::vector<Normalizer::Ptr> normalizers;
  normalizers.push_back(std::make_shared<ReplaceNormalizer>(" ", "▁"));
  normalizers.push_back(std::make_shared<ReplaceNormalizer>("a", "X"));

  SequenceNormalizer seq_normalizer(normalizers);

  std::string input = "banana split";
  std::string expected = "bXnXnX▁split";
  std::string result = seq_normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, SequenceNormalizerFromConfig) {
  // Test sequence normalizer from config
  nlohmann::json config = {
      {"type", "Sequence"},
      {"normalizers",
       {{{"type", "Replace"}, {"pattern", {{"String", " "}}}, {"content", "▁"}},
        {{"type", "Replace"},
         {"pattern", {{"String", "a"}}},
         {"content", "X"}}}}};

  NormalizerConfig norm_config;
  norm_config.parse_json(config);
  auto normalizer = norm_config.create();

  std::string input = "banana split";
  std::string expected = "bXnXnX▁split";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, EmptyInput) {
  // Test with empty input
  ReplaceNormalizer normalizer(" ", "▁");
  std::string input;
  std::string expected;
  std::string result = normalizer.normalize(input);
  EXPECT_EQ(result, expected);
}

TEST(NormalizerTest, ConfigBuilder) {
  // Test config builder pattern
  auto normalizer =
      NormalizerConfig("Replace").set_pattern(" ").set_content("▁").create();

  std::string input = "Hello World";
  std::string expected = "Hello▁World";
  std::string result = normalizer->normalize(input);
  EXPECT_EQ(result, expected);
}
