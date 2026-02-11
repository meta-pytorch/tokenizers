/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <gtest/gtest.h>
#include <pytorch/tokenizers/sentencepiece.h>

namespace tokenizers {

namespace {
static inline std::string _get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}
} // namespace

TEST(SPTokenizerTest, TestEncodeWithoutLoad) {
  SPTokenizer tokenizer;
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 0, /*eos*/ 1);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(SPTokenizerTest, TestDecodeWithoutLoad) {
  SPTokenizer tokenizer;
  auto result = tokenizer.decode(0, 0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(SPTokenizerTest, TestIdToPieceWithoutLoad) {
  SPTokenizer tokenizer;
  auto result = tokenizer.id_to_piece(0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(SPTokenizerTest, TestPieceToIdWithoutLoad) {
  SPTokenizer tokenizer;
  auto result = tokenizer.piece_to_id("Hello");
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(SPTokenizerTest, TestPieceToId) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto hello = tokenizer.piece_to_id("▁Hello");
  EXPECT_EQ(hello.error(), Error::Ok);
  EXPECT_EQ(hello.get(), 15043);

  auto world = tokenizer.piece_to_id("▁world");
  EXPECT_EQ(world.error(), Error::Ok);
  EXPECT_EQ(world.get(), 3186);

  auto bang = tokenizer.piece_to_id("!");
  EXPECT_EQ(bang.error(), Error::Ok);
  EXPECT_EQ(bang.get(), 29991);
}

TEST(SPTokenizerTest, PieceToIdNotFoundFails) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto result = tokenizer.piece_to_id("not_a_real_piece");
  EXPECT_EQ(result.error(), Error::OutOfRange);
}

TEST(SPTokenizerTest, TestLoad) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
}

TEST(SPTokenizerTest, TestLoadInvalidPath) {
  SPTokenizer tokenizer;
  auto error = tokenizer.load("invalid_path");
  EXPECT_EQ(error, Error::LoadFailure);
}

TEST(SPTokenizerTest, TestEncode) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 1, /*eos*/ 0);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get().size(), 4);
  EXPECT_EQ(result.get()[0], 1);
  EXPECT_EQ(result.get()[1], 15043);
  EXPECT_EQ(result.get()[2], 3186);
  EXPECT_EQ(result.get()[3], 29991);
}

TEST(SPTokenizerTest, TestDecode) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::vector<uint64_t> tokens = {1, 15043, 3186, 29991};
  std::vector<std::string> expected = {"", "Hello", " world", "!"};
  for (auto i = 0; i < 3; ++i) {
    auto result = tokenizer.decode(tokens[i], tokens[i + 1]);
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result.get(), expected[i + 1]);
  }
}

TEST(SPTokenizerTest, TestDecodeSpecialTokens) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  uint64_t bos = tokenizer.bos_tok();

  // skip_special_tokens = false
  auto res_false = tokenizer.decode(0, bos, false);
  EXPECT_TRUE(res_false.ok());
  // SPTokenizer returns " " for control tokens like BOS/EOS
  EXPECT_EQ(res_false.get(), " ");

  // skip_special_tokens = true
  // SPTokenizer ignores the skip_special_tokens flag
  auto res_true = tokenizer.decode(0, bos, true);
  EXPECT_TRUE(res_true.ok());
  EXPECT_EQ(res_true.get(), " ");
}

TEST(SPTokenizerTest, TestIdToPiece) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto hello = tokenizer.id_to_piece(15043);
  EXPECT_EQ(hello.error(), Error::Ok);
  EXPECT_EQ(hello.get(), "▁Hello");

  auto world = tokenizer.id_to_piece(3186);
  EXPECT_EQ(world.error(), Error::Ok);
  EXPECT_EQ(world.get(), "▁world");

  auto bang = tokenizer.id_to_piece(29991);
  EXPECT_EQ(bang.error(), Error::Ok);
  EXPECT_EQ(bang.get(), "!");
}

TEST(SPTokenizerTest, IdToPieceOutOfRangeFails) {
  SPTokenizer tokenizer;
  auto path = _get_resource_path("test_sentencepiece.model");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto result =
      tokenizer.id_to_piece(static_cast<uint64_t>(tokenizer.vocab_size()));
  EXPECT_EQ(result.error(), Error::OutOfRange);
}
} // namespace tokenizers
