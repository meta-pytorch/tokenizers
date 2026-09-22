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
  std::string input = "";
  std::string expected = "";
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
  std::string input = "";
  std::string expected = "";
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

TEST(NormalizerTest, NFCNormalizerCanonicalComposition) {
  NFCNormalizer normalizer;
  EXPECT_EQ(normalizer.normalize("Cafe\u0301"), "Caf\u00e9");
  EXPECT_EQ(normalizer.normalize("A\u030c"), "\u01cd");
  EXPECT_EQ(normalizer.normalize("s\u0323\u0307"), "\u1e69");
}

TEST(NormalizerTest, NFCNormalizerPreservesNormalizedText) {
  NFCNormalizer normalizer;
  EXPECT_EQ(normalizer.normalize(""), "");
  EXPECT_EQ(normalizer.normalize("Hello world"), "Hello world");
  EXPECT_EQ(
      normalizer.normalize("\u00e9\u01cd\u1e69\uac01"),
      "\u00e9\u01cd\u1e69\uac01");
}

TEST(NormalizerTest, NFCNormalizerCanonicalSingleton) {
  NFCNormalizer normalizer;
  // ANGSTROM SIGN canonically decomposes to LATIN CAPITAL LETTER A WITH RING.
  EXPECT_EQ(normalizer.normalize("\u212b"), "\u00c5");
}

TEST(NormalizerTest, NFCNormalizerCompositionExclusions) {
  NFCNormalizer normalizer;
  // These canonical decompositions must not recompose in NFC.
  EXPECT_EQ(normalizer.normalize("\u0958"), "\u0915\u093c");
  EXPECT_EQ(normalizer.normalize("\u0915\u093c"), "\u0915\u093c");
  EXPECT_EQ(normalizer.normalize("\U0001d15e"), "\U0001d157\U0001d165");
}

TEST(NormalizerTest, NFCNormalizerPreservesCompatibilityCharacters) {
  NFCNormalizer normalizer;
  // NFC preserves ligatures and fullwidth characters.
  EXPECT_EQ(normalizer.normalize("\ufb01\uff21"), "\ufb01\uff21");
}

TEST(NormalizerTest, NFCNormalizerCanonicalOrdering) {
  NFCNormalizer normalizer;
  EXPECT_EQ(normalizer.normalize("a\u0315\u0300"), "\u00e0\u0315");
  EXPECT_EQ(normalizer.normalize("\u05e9\u05c1\u05b8"), "\u05e9\u05b8\u05c1");
  EXPECT_EQ(normalizer.normalize("\u0628\u0651\u064e"), "\u0628\u064e\u0651");
  // Leading combining marks are ordered without crossing the next starter.
  EXPECT_EQ(normalizer.normalize("\u0315\u0300a"), "\u0300\u0315a");
}

TEST(NormalizerTest, NFCNormalizerCompositionBlocking) {
  NFCNormalizer normalizer;
  // An intervening mark of the same combining class blocks composition.
  EXPECT_EQ(normalizer.normalize("A\u0305\u030a"), "A\u0305\u030a");
  // An intervening mark of a lower combining class permits composition.
  EXPECT_EQ(normalizer.normalize("A\u0327\u030a"), "\u00c5\u0327");
}

TEST(NormalizerTest, NFCNormalizerHangulComposition) {
  NFCNormalizer normalizer;
  EXPECT_EQ(normalizer.normalize("\u1100\u1161"), "\uac00");
  EXPECT_EQ(normalizer.normalize("\u1100\u1161\u11a8"), "\uac01");
  EXPECT_EQ(normalizer.normalize("\uac00\u11a8"), "\uac01");
  // A combining mark between Jamo prevents their composition.
  EXPECT_EQ(normalizer.normalize("\u1100\u0301\u1161"), "\u1100\u0301\u1161");
}

TEST(NormalizerTest, NormalizerConfigNFC) {
  NormalizerConfig config;
  config.parse_json(nlohmann::json{{"type", "NFC"}});
  auto normalizer = config.create();
  EXPECT_EQ(normalizer->normalize("Cafe\u0301"), "Caf\u00e9");
}
