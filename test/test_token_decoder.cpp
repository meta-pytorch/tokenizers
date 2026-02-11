/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/token_decoder.h>
#include <unicode.h>

namespace tokenizers {

// Test ReplaceTokenDecoder
TEST(ReplaceTokenDecoderTest, TestBasicReplace) {
  ReplaceTokenDecoder decoder("_", " ");

  EXPECT_EQ(decoder.decode(std::vector<std::string>{"_Hello"})[0], " Hello");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"_world!"})[0], " world!");
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"Hello_world"})[0],
      "Hello world");
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"no_replacement"})[0],
      "no replacement");
}

TEST(ReplaceTokenDecoderTest, TestMultipleReplacements) {
  ReplaceTokenDecoder decoder("_", " ");

  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"_Hello_world_!"})[0],
      " Hello world !");
}

TEST(ReplaceTokenDecoderTest, TestEmptyPattern) {
  ReplaceTokenDecoder decoder("", "X");

  // Empty pattern should not cause infinite loop
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"test"})[0], "test");
}

// Test ByteFallbackTokenDecoder
TEST(ByteFallbackTokenDecoderTest, TestValidHexTokens) {
  ByteFallbackTokenDecoder decoder;

  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0x41>"})[0],
      "A"); // 0x41 = 65 = 'A'
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0x42>"})[0],
      "B"); // 0x42 = 66 = 'B'
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0x20>"})[0],
      " "); // 0x20 = 32 = space
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0x00>"})[0],
      std::string(1, '\0')); // null byte
}

TEST(ByteFallbackTokenDecoderTest, TestInvalidHexTokens) {
  ByteFallbackTokenDecoder decoder;

  // Invalid format - should return original token
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0xGG>"})[0], "<0xGG>");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0x>"})[0], "<0x>");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"0x41>"})[0], "0x41>");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0x41"})[0], "<0x41");
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"regular_token"})[0],
      "regular_token");
}

TEST(ByteFallbackTokenDecoderTest, TestOutOfRangeValues) {
  ByteFallbackTokenDecoder decoder;

  // Values > 255 should return original token
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0x100>"})[0], "<0x100>");
  EXPECT_EQ(
      decoder.decode(std::vector<std::string>{"<0xFFFF>"})[0], "<0xFFFF>");
}

// Test FuseTokenDecoder
TEST(FuseTokenDecoderTest, TestPassthrough) {
  FuseTokenDecoder decoder;

  EXPECT_EQ(decoder.decode(std::vector<std::string>{"test"})[0], "test");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"_Hello"})[0], "_Hello");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{"<0x41>"})[0], "<0x41>");
  EXPECT_EQ(decoder.decode(std::vector<std::string>{""})[0], "");
}

TEST(FuseTokenDecoderTest, TestBatchFuse) {
  FuseTokenDecoder decoder;

  std::vector<std::string> tokens = {"Hello", " ", "world", "!"};
  auto result = decoder.decode(tokens);

  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "Hello world!");
}

// Test SequenceTokenDecoder
TEST(SequenceTokenDecoderTest, TestEmptySequence) {
  std::vector<TokenDecoder::Ptr> decoders;
  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"test"})[0], "test");
}

TEST(SequenceTokenDecoderTest, TestSingleDecoder) {
  std::vector<TokenDecoder::Ptr> decoders;
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"_Hello"})[0], " Hello");
}

TEST(SequenceTokenDecoderTest, TestMultipleDecoders) {
  std::vector<TokenDecoder::Ptr> decoders;

  // Add Replace decoder to replace _ with space
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  // Add ByteFallback decoder
  decoders.push_back(TokenDecoder::Ptr(new ByteFallbackTokenDecoder()));

  // Add Fuse decoder
  decoders.push_back(TokenDecoder::Ptr(new FuseTokenDecoder()));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  // Test cases
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"_Hello"})[0], " Hello");
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"_world!"})[0],
      " world!");
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"<0x41>"})[0], "A");
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"normal_token"})[0],
      "normal token");
}

TEST(SequenceTokenDecoderTest, TestComplexSequence) {
  std::vector<TokenDecoder::Ptr> decoders;

  // First replace underscores with spaces
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  // Then replace _ with spaces
  decoders.push_back(TokenDecoder::Ptr(new ReplaceTokenDecoder("_", " ")));

  // Then handle byte fallback
  decoders.push_back(TokenDecoder::Ptr(new ByteFallbackTokenDecoder()));

  SequenceTokenDecoder sequence_decoder(std::move(decoders));

  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"Hello_world"})[0],
      "Hello world");
  EXPECT_EQ(
      sequence_decoder.decode(std::vector<std::string>{"_test_token"})[0],
      " test token");
}

// Test TokenDecoderConfig parsing and creation
TEST(TokenDecoderConfigTest, TestReplaceConfig) {
  nlohmann::json config = {
      {"type", "Replace"}, {"pattern", {{"String", "_"}}}, {"content", " "}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "Replace");
  EXPECT_EQ(decoder_config.replace_pattern, "_");
  EXPECT_EQ(decoder_config.replace_content, " ");

  auto decoder = decoder_config.create();
  EXPECT_EQ(decoder->decode(std::vector<std::string>{"_Hello"})[0], " Hello");
}

TEST(TokenDecoderConfigTest, TestSequenceConfig) {
  nlohmann::json config = {
      {"type", "Sequence"},
      {"decoders",
       {{{"type", "Replace"}, {"pattern", {{"String", "_"}}}, {"content", " "}},
        {{"type", "ByteFallback"}},
        {{"type", "Fuse"}}}}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "Sequence");
  EXPECT_EQ(decoder_config.sequence_decoders.size(), 3);

  auto decoder = decoder_config.create();
  EXPECT_EQ(decoder->decode(std::vector<std::string>{"_Hello"})[0], " Hello");
  EXPECT_EQ(decoder->decode(std::vector<std::string>{"<0x41>"})[0], "A");
}

// Test StripTokenDecoder
TEST(StripTokenDecoderTest, TestBasicStrip) {
  StripTokenDecoder decoder("H", 1, 0); // Strip 'H' from start, 1 char max
  std::vector<std::string> tokens = {"Hey", " friend!", "HHH"};
  std::vector<std::string> expected = {"ey", " friend!", "HH"};
  EXPECT_EQ(decoder.decode(tokens), expected);

  StripTokenDecoder decoder2("y", 0, 1); // Strip 'y' from end, 1 char max
  tokens = {"Hey", " friend!"};
  expected = {"He", " friend!"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(StripTokenDecoderTest, TestNoStrip) {
  StripTokenDecoder decoder("X", 1, 1); // Try to strip 'X'
  std::vector<std::string> tokens = {"Hello", "World"};
  std::vector<std::string> expected = {"Hello", "World"};
  EXPECT_EQ(decoder.decode(tokens), expected);

  StripTokenDecoder decoder2("H", 0, 0); // No stripping
  tokens = {"Hello", "World"};
  expected = {"Hello", "World"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(StripTokenDecoderTest, TestFullStrip) {
  StripTokenDecoder decoder("H", 3, 0); // Strip 'H' from start, 3 chars max
  std::vector<std::string> tokens = {"HHH"};
  std::vector<std::string> expected = {""};
  EXPECT_EQ(decoder.decode(tokens), expected);

  StripTokenDecoder decoder2("H", 1, 1); // Strip 'H' from start and end
  tokens = {"HHH"};
  expected = {"H"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(StripTokenDecoderTest, TestMixedContent) {
  StripTokenDecoder decoder(" ", 1, 1); // Strip spaces from start and end
  std::vector<std::string> tokens = {" Hello World ", "  TrimMe  "};
  std::vector<std::string> expected = {"Hello World", " TrimMe "};
  EXPECT_EQ(decoder.decode(tokens), expected);

  StripTokenDecoder decoder2("H", 5, 5); // Strip 'H's from start and end
  tokens = {"HHHHelloHHH", "HWorldH"};
  expected = {"ello", "World"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(StripTokenDecoderTest, TestUnicodeContent) {
  // Test with a multi-byte character for content
  StripTokenDecoder decoder("_", 1, 0); // Strip '_' from start, 1 char max
  std::vector<std::string> tokens = {"_Hello", "__World"};
  std::vector<std::string> expected = {"Hello", "_World"};
  EXPECT_EQ(decoder.decode(tokens), expected);

  // Test stripping from end with Unicode
  StripTokenDecoder decoder2("∎", 0, 1); // Strip '∎' from end, 1 char max
  tokens = {"Hello∎", "World∎∎"};
  expected = {"Hello", "World∎"};
  EXPECT_EQ(decoder2.decode(tokens), expected);
}

TEST(TokenDecoderConfigTest, TestStripConfig) {
  nlohmann::json config = {
      {"type", "Strip"}, {"content", "_"}, {"start", 1}, {"stop", 0}};

  TokenDecoderConfig decoder_config;
  decoder_config.parse_json(config);

  EXPECT_EQ(decoder_config.type, "Strip");
  EXPECT_EQ(decoder_config.strip_content, "_");
  EXPECT_EQ(decoder_config.strip_start, 1);
  EXPECT_EQ(decoder_config.strip_stop, 0);

  auto decoder = decoder_config.create();
  std::vector<std::string> tokens = {"_Hello", "__World"};
  std::vector<std::string> expected = {"Hello", "_World"};
  EXPECT_EQ(decoder->decode(tokens), expected);
}

} // namespace tokenizers
