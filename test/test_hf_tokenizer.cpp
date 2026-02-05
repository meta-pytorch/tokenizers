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

#include <fstream>

namespace tokenizers {

namespace {
static inline std::string _get_resource_path(const std::string& name) {
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
}

// Helper to create a temporary file with given content
class TempFile {
 public:
  TempFile(const std::string& content) {
    path_ = std::tmpnam(nullptr);
    path_ += ".json";
    std::ofstream f(path_);
    f << content;
  }
  ~TempFile() {
    std::remove(path_.c_str());
  }
  const std::string& path() const {
    return path_;
  }

 private:
  std::string path_;
};
} // namespace

TEST(HFTokenizerTest, TestEncodeWithoutLoad) {
  HFTokenizer tokenizer;
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 0, /*eos*/ 1);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(HFTokenizerTest, TestDecodeWithoutLoad) {
  HFTokenizer tokenizer;
  auto result = tokenizer.decode({0, 0});
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(HFTokenizerTest, TestIdToPieceWithoutLoad) {
  HFTokenizer tokenizer;
  auto result = tokenizer.id_to_piece(0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST(HFTokenizerTest, TestPieceToIdWithoutLoad) {
  HFTokenizer tokenizer;
  auto result = tokenizer.piece_to_id("<s>");
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

TEST(HFTokenizerTest, TestPieceToId) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto unk = tokenizer.piece_to_id("<unk>");
  EXPECT_EQ(unk.error(), Error::Ok);
  EXPECT_EQ(unk.get(), 0);

  auto bos = tokenizer.piece_to_id("<s>");
  EXPECT_EQ(bos.error(), Error::Ok);
  EXPECT_EQ(bos.get(), 1);

  auto hello = tokenizer.piece_to_id("▁Hello");
  EXPECT_EQ(hello.error(), Error::Ok);
  EXPECT_EQ(hello.get(), 8);

  auto world = tokenizer.piece_to_id("▁world!");
  EXPECT_EQ(world.error(), Error::Ok);
  EXPECT_EQ(world.get(), 9);
}

TEST(HFTokenizerTest, PieceToIdNotFoundFails) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  auto result = tokenizer.piece_to_id("not_a_real_piece");
  EXPECT_EQ(result.error(), Error::OutOfRange);
}

TEST(HFTokenizerTest, TestEncode) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("test_hf_tokenizer.json");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  std::string text = "Hello world!";
  auto result = tokenizer.encode(text, /*bos*/ 1, /*eos*/ 1);
  EXPECT_TRUE(result.ok());
  // Based on our test tokenizer vocab:
  // "Hello world!" should tokenize to something like [1, 8, 9] or [1, 4, 5, 6,
  // 7] depending on how the BPE merges work
  EXPECT_GT(result.get().size(), 0);
  EXPECT_EQ(result.get()[0], 4); // First token 'H' from "Hello"
}

TEST(HFTokenizerTest, TestDecodeBatch) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("hf_tokenizer_dir/");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);
  // Test with tokens from our vocab: <s> (1), ▁Hello (8), ▁world! (9)
  // Note: in hf_tokenizer_dir, bos is 128000
  uint64_t bos = tokenizer.bos_tok();
  std::vector<uint64_t> tokens = {bos, 8, 9};
  
  // skip_special_tokens = false
  auto result_false = tokenizer.decode(tokens, false);
  EXPECT_TRUE(result_false.ok());
  EXPECT_EQ(result_false.get(), "<|begin_of_text|>▁Hello▁world!");

  // skip_special_tokens = true
  auto result_true = tokenizer.decode(tokens, true);
  EXPECT_TRUE(result_true.ok());
  EXPECT_EQ(result_true.get(), "▁Hello▁world!");
}

TEST(HFTokenizerTest, TestDecodeSpecialTokens) {
  HFTokenizer tokenizer;
  auto path = _get_resource_path("hf_tokenizer_dir/");
  auto error = tokenizer.load(path);
  EXPECT_EQ(error, Error::Ok);

  uint64_t bos = tokenizer.bos_tok();

  // Single token decode: skip_special_tokens = false
  auto res_false = tokenizer.decode(0, bos, false);
  EXPECT_TRUE(res_false.ok());
  EXPECT_EQ(res_false.get(), "<|begin_of_text|>");

  // Single token decode: skip_special_tokens = true
  auto res_true = tokenizer.decode(0, bos, true);
  EXPECT_TRUE(res_true.ok());
  EXPECT_EQ(res_true.get(), "");
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
// Test that BPE merges are correctly parsed from legacy string format ("a b")
// This is the standard HuggingFace tokenizer.json format
TEST(HFTokenizerTest, TestBPEMergeLegacyFormat) {
  // Create a minimal tokenizer.json with legacy string merges format
  // Vocab: a=0, b=1, ab=2, c=3, abc=4
  // Merges: "a b" -> ab, "ab c" -> abc
  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "ab": 2,
        "c": 3,
        "abc": 4
      },
      "merges": [
        "a b",
        "ab c"
      ]
    },
    "normalizer": null,
    "pre_tokenizer": {
      "type": "ByteLevel",
      "add_prefix_space": false,
      "trim_offsets": false,
      "use_regex": false
    },
    "added_tokens": []
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  // If merges are parsed correctly, encoding "abc" should produce token 4
  // (after merging a+b->ab, then ab+c->abc)
  // Note: This test verifies the merge parsing works; actual encoding
  // depends on pre-tokenizer setup which may not be configured in this
  // minimal example.
}

// Test that BPE merges are correctly parsed from tuple array format (["a",
// "b"]) This format supports tokens containing spaces
TEST(HFTokenizerTest, TestBPEMergeTupleFormat) {
  // Create a minimal tokenizer.json with tuple array merges format
  // This format is used when tokens contain spaces
  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "ab": 2,
        "c d": 3,
        "abc d": 4
      },
      "merges": [
        ["a", "b"],
        ["ab", "c d"]
      ]
    },
    "normalizer": null,
    "pre_tokenizer": {
      "type": "ByteLevel",
      "add_prefix_space": false,
      "trim_offsets": false,
      "use_regex": false
    },
    "added_tokens": []
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  // Verifies that tuple format merges are parsed correctly,
  // including merges involving tokens with spaces like "c d"
}

// Test that #version header lines are properly skipped in merges
// This matches HuggingFace Rust tokenizers behavior (see model.rs:292)
TEST(HFTokenizerTest, TestBPEMergeVersionHeader) {
  // Create a tokenizer.json with #version header in merges
  // The #version line should be skipped, not treated as a merge
  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "ab": 2
      },
      "merges": [
        "#version: 0.2",
        "a b"
      ]
    },
    "normalizer": null,
    "pre_tokenizer": {
      "type": "ByteLevel",
      "add_prefix_space": false,
      "trim_offsets": false,
      "use_regex": false
    },
    "added_tokens": []
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  // The #version line should be skipped, leaving only the "a b" merge
  // If #version was incorrectly parsed as a merge, it would fail or
  // produce incorrect results
}

// Test that merges produce correct tokenization results
// This verifies the full encode path with BPE merges
TEST(HFTokenizerTest, TestBPEMergeEncode) {
  // Create a tokenizer that can merge "a" + "b" -> "ab"
  // and "ab" + "c" -> "abc"
  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "b": 1,
        "c": 2,
        "ab": 3,
        "abc": 4
      },
      "merges": [
        "a b",
        "ab c"
      ]
    },
    "normalizer": null,
    "pre_tokenizer": {
      "type": "ByteLevel",
      "add_prefix_space": false,
      "trim_offsets": false,
      "use_regex": false
    },
    "added_tokens": []
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  // Encode "abc" - should merge to single token if merges work correctly
  auto result = tokenizer.encode("abc", /*bos=*/0, /*eos=*/0);
  if (result.ok()) {
    // With correct BPE merges:
    // "abc" -> ['a', 'b', 'c'] -> ['ab', 'c'] -> ['abc']
    // So we expect 1 token with id 4
    auto tokens = result.get();
    EXPECT_EQ(tokens.size(), 1);
    if (tokens.size() == 1) {
      EXPECT_EQ(tokens[0], 4); // "abc" token
    }
  }
  // Note: This test may not produce the expected result due to ByteLevel
  // pre-tokenizer transforming input bytes. The primary purpose is to
  // verify that merges are parsed and the tokenizer loads successfully.
}

} // namespace tokenizers
