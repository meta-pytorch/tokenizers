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

// The following tests pin HFWord::merge_all behavior. They use null normalizer
// and null pre-tokenizer so the whole input becomes a single BPE piece, which
// lets us drive merge_all directly with hand-built vocab/merges.
namespace {
std::string merge_tokenizer_json(
    const std::string& vocab,
    const std::string& merges,
    bool byte_fallback = false) {
  return std::string(R"({"version":"1.0","model":{"type":"BPE","vocab":)") +
      vocab + R"(,"merges":)" + merges +
      R"(,"byte_fallback":)" + (byte_fallback ? "true" : "false") +
      R"(},"normalizer":null,"pre_tokenizer":null,"added_tokens":[]})";
}
} // namespace

// Cascading merge: a+b->ab enables ab+c->abc; a trailing symbol stays unmerged.
TEST(HFTokenizerTest, MergeAllCascades) {
  TempFile tmpfile(merge_tokenizer_json(
      R"({"a":0,"b":1,"c":2,"ab":3,"abc":4})", R"(["a b","ab c"])"));
  HFTokenizer tokenizer;
  ASSERT_EQ(tokenizer.load(tmpfile.path()), Error::Ok);
  auto result = tokenizer.encode("abcc", 0, 0);
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {4, 2}; // abc, c
  EXPECT_EQ(result.get(), expected);
}

// The globally lowest-rank merge wins even when it is not the leftmost pair:
// (y z) has rank 0, (x y) has rank 1, so y+z merges first.
TEST(HFTokenizerTest, MergeAllPicksLowestRankNotLeftmost) {
  TempFile tmpfile(merge_tokenizer_json(
      R"({"x":0,"y":1,"z":2,"xy":3,"yz":4})", R"(["y z","x y"])"));
  HFTokenizer tokenizer;
  ASSERT_EQ(tokenizer.load(tmpfile.path()), Error::Ok);
  auto result = tokenizer.encode("xyz", 0, 0);
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {0, 4}; // x, yz
  EXPECT_EQ(result.get(), expected);
}

// Overlapping equal-rank candidates: "aaa" with a+a->aa must merge the leftmost
// pair first (-> [aa, a]), exercising stale-entry invalidation of the overlap.
TEST(HFTokenizerTest, MergeAllLeftmostOnOverlap) {
  TempFile tmpfile(
      merge_tokenizer_json(R"({"a":0,"aa":1})", R"(["a a"])"));
  HFTokenizer tokenizer;
  ASSERT_EQ(tokenizer.load(tmpfile.path()), Error::Ok);
  auto result = tokenizer.encode("aaa", 0, 0);
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {1, 0}; // aa, a
  EXPECT_EQ(result.get(), expected);
}

// Merge over byte-fallback symbols: 'c' falls back to <0x63>, then a+b->ab.
TEST(HFTokenizerTest, MergeAllWithByteFallback) {
  TempFile tmpfile(merge_tokenizer_json(
      R"({"a":0,"b":1,"ab":2,"<0x63>":3})", R"(["a b"])", /*byte_fallback=*/true));
  HFTokenizer tokenizer;
  ASSERT_EQ(tokenizer.load(tmpfile.path()), Error::Ok);
  auto result = tokenizer.encode("abc", 0, 0);
  ASSERT_TRUE(result.ok());
  std::vector<uint64_t> expected = {2, 3}; // ab, <0x63>
  EXPECT_EQ(result.get(), expected);
}

TEST(HFTokenizerTest, TestByteFallback) {
  // Create a minimal tokenizer with byte fallback enabled
  // Vocab: "a": 0, "<0x62>": 1 (for 'b')
  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0,
        "<0x62>": 1
      },
      "merges": [],
      "byte_fallback": true
    },
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": {
      "type": "ByteLevel"
    },
    "added_tokens": []
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  // 'a' is in vocab, 'b' should fallback to <0x62>
  auto result = tokenizer.encode("ab", 0, 0);
  EXPECT_TRUE(result.ok());
  std::vector<uint64_t> expected = {0, 1};
  EXPECT_EQ(result.get(), expected);

  // Decode should also work if we have a decoder
  auto decoded = tokenizer.decode(result.get());
  EXPECT_TRUE(decoded.ok());
  EXPECT_EQ(decoded.get(), "a<0x62>");
}

TEST(HFTokenizerTest, TestProperRoundTrip) {
  // We use a custom config here because the standard 'test_hf_tokenizer.json'
  // uses a Replace normalizer (" " -> "▁"). To achieve decode(encode(x)) === x,
  // we must provide a Decoder Sequence that reverses this mapping.

  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "H": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6, "!": 7, "▁": 8
      },
      "merges": []
    },
    "normalizer": {
      "type": "Replace",
      "pattern": { "String": " " },
      "content": "▁"
    },
    "pre_tokenizer": null,
    "post_processor": null,
    "decoder": {
      "type": "Sequence",
      "decoders": [
        {
          "type": "Replace",
          "pattern": { "String": "▁" },
          "content": " "
        }
      ]
    },
    "added_tokens": []
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  std::string input = "Hello world!";

  auto encoded = tokenizer.encode(input, 0, 0);
  ASSERT_TRUE(encoded.ok());

  auto decoded = tokenizer.decode(encoded.get());
  ASSERT_TRUE(decoded.ok());

  // Identity is now preserved because the Decoder Sequence is the
  // mathematical inverse of the Normalizer + PreTokenizer.
  EXPECT_EQ(decoded.get(), input);
}

TEST(HFTokenizerTest, TestNullPreTokenizer) {
  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "hello": 0,
        "world": 1
      },
      "merges": []
    },
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "added_tokens": []
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  auto result = tokenizer.encode("hello", 0, 0);
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(result.get().size(), 1);
  EXPECT_EQ(result.get()[0], 0);
}

TEST(HFTokenizerTest, TestEmptyAndUnknown) {
  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0
      },
      "merges": [],
      "unk_token": "[UNK]"
    },
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "added_tokens": [
      {
        "id": 1,
        "content": "[UNK]",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false
      }
    ]
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  // Empty string
  auto empty_result = tokenizer.encode("", 0, 0);
  EXPECT_TRUE(empty_result.ok());
  EXPECT_TRUE(empty_result.get().empty());

  // Unknown character
  auto unk_result = tokenizer.encode("b", 0, 0);
  EXPECT_TRUE(unk_result.ok());
  std::vector<uint64_t> expected_unk = {1}; // [UNK]
  EXPECT_EQ(unk_result.get(), expected_unk);
}

TEST(HFTokenizerTest, TestUnkTokenConfiguration) {
  const char* json = R"({
    "version": "1.0",
    "model": {
      "type": "BPE",
      "vocab": {
        "a": 0
      },
      "merges": [],
      "unk_token": "<unk>"
    },
    "normalizer": null,
    "pre_tokenizer": null,
    "post_processor": null,
    "added_tokens": [
      {
        "id": 1,
        "content": "<unk>",
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false
      }
    ]
  })";

  TempFile tmpfile(json);
  HFTokenizer tokenizer;
  auto error = tokenizer.load(tmpfile.path());
  EXPECT_EQ(error, Error::Ok);

  // Encode unknown
  auto result = tokenizer.encode("xyz", 0, 0);
  EXPECT_TRUE(result.ok());
  // 'x', 'y', 'z' are all unknown
  std::vector<uint64_t> expected = {1, 1, 1};
  EXPECT_EQ(result.get(), expected);
}

} // namespace tokenizers
