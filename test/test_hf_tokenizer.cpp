/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/hf_tokenizer.h>

namespace tokenizers {

namespace {
static inline std::string _get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}
} // namespace

TEST(HFTokenizerTest, TestEncodeWithoutLoad) {
  HFTokenizer tokenizer;
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 0, /*eos*/ 1);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(HFTokenizerTest, TestDecodeWithoutLoad) {
  HFTokenizer tokenizer;
  auto result = tokenizer.decode(0, 0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(HFTokenizerTest, TestIdToPieceWithoutLoad) {
  HFTokenizer tokenizer;
  auto result = tokenizer.id_to_piece(0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(HFTokenizerTest, TestLoad) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
}

TEST(HFTokenizerTest, TestLoadInvalidPath) {
  HFTokenizer tokenizer;
  auto error = tokenizer.load("invalid_path");
  EXPECT_EQ(error, Error::LoadFailure);
}

TEST(HFTokenizerTest, TestSpecialTokensMap) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("hf_tokenizer_dir/");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  // Verify bos_token is loaded from special_tokens_map.json
  auto bos_token_id = tokenizer.bos_tok();
  EXPECT_EQ(bos_token_id, 128000); // <|begin_of_text|>

  // Verify eos_token is loaded from special_tokens_map.json
  auto eos_token_id = tokenizer.eos_tok();
  EXPECT_EQ(eos_token_id, 128009); // <|eot_id|>
}

TEST(HFTokenizerTest, TestIdToPiece) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto unk = tokenizer.id_to_piece(0);
  EXPECT_EQ(unk.error(), Error::Ok);
  EXPECT_EQ(unk.get(), "<unk>");

  auto bos = tokenizer.id_to_piece(1);
  EXPECT_EQ(bos.error(), Error::Ok);
  EXPECT_EQ(bos.get(), "<s>");

  auto hello = tokenizer.id_to_piece(8);
  EXPECT_EQ(hello.error(), Error::Ok);
  EXPECT_EQ(hello.get(), "▁Hello");

  auto world = tokenizer.id_to_piece(9);
  EXPECT_EQ(world.error(), Error::Ok);
  EXPECT_EQ(world.get(), "▁world!");
}

TEST(HFTokenizerTest, TestIdToPieceSpecialTokensMap) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("hf_tokenizer_dir/");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto bos = tokenizer.id_to_piece(tokenizer.bos_tok());
  EXPECT_EQ(bos.error(), Error::Ok);
  EXPECT_EQ(bos.get(), "<|begin_of_text|>");

  auto eos = tokenizer.id_to_piece(tokenizer.eos_tok());
  EXPECT_EQ(eos.error(), Error::Ok);
  EXPECT_EQ(eos.get(), "<|eot_id|>");
}

TEST(HFTokenizerTest, IdToPieceOutOfRangeFails) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto result =
      tokenizer.id_to_piece(static_cast<uint64_t>(tokenizer.vocab_size()) + 1);
  EXPECT_EQ(result.error(), Error::OutOfRange);
}

TEST(HFTokenizerTest, TestEncode) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 1, /*eos*/ 0);
  EXPECT_TRUE(result.ok());
  // Based on our test tokenizer vocab:
  // "Hello world!" should tokenize to something like [1, 8, 9] or [1, 4, 5, 6,
  // 7] depending on how the BPE merges work
  EXPECT_GT(result.get().size(), 0);
  EXPECT_EQ(result.get()[0], 0); // BOS token (default BOS ID)
}

TEST(HFTokenizerTest, TestDecode) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  // Test with tokens from our vocab: <s>, ▁Hello, ▁world!
  std::vector<uint64_t> tokens = {1, 8, 9}; // <s>, ▁Hello, ▁world!
  for (auto i = 0; i < static_cast<int>(tokens.size()) - 1; ++i) {
    auto result = tokenizer.decode(tokens[i], tokens[i + 1]);
    EXPECT_TRUE(result.ok());
    // The decoded strings should not be empty
    EXPECT_FALSE(result.get().empty());
  }
}

} // namespace tokenizers
