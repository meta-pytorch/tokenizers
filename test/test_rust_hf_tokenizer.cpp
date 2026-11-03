/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <pytorch/tokenizers/rust_hf_tokenizer.h>

namespace {

constexpr uint32_t kByteLevelFlag = 1U << 2;
uint32_t g_config_flags = kByteLevelFlag;
bool g_native_decoder = false;
bool g_use_known_eos = true;

struct TokenRecord {
  uint32_t id;
  std::string text;
  bool added;
  bool special;
};

const std::array<TokenRecord, 4> kRecords = {
    {{1, "<s>", true, true},
     {2, "</s>", true, true},
     {3, "hello", false, false},
     {4, "\xC4\xA0world", false, false}}};

class TemporaryTokFile {
 public:
  TemporaryTokFile()
      : path_(
            std::filesystem::temp_directory_path() /
            "pytorch-tokenizers-rust-adapter-test.tok") {
    std::ofstream(path_).put('\0');
  }

  ~TemporaryTokFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  std::string string() const {
    return path_.string();
  }

 private:
  std::filesystem::path path_;
};

} // namespace

extern "C" {

void* tokenizers_hf_create(const char*) {
  return new uint8_t(0);
}

intptr_t tokenizers_hf_encode(
    const void*,
    const uint8_t*,
    size_t,
    uint32_t* output,
    size_t output_capacity) {
  constexpr std::array<uint32_t, 2> encoded = {3, 4};
  if (output_capacity < encoded.size()) {
    return encoded.size();
  }
  std::copy(encoded.begin(), encoded.end(), output);
  return encoded.size();
}

intptr_t tokenizers_hf_decode(
    const void*,
    uint32_t,
    uint8_t,
    uint32_t token,
    uint8_t skip_special,
    uint8_t* output,
    size_t output_capacity) {
  const std::string decoded =
      skip_special != 0 && token == 2 ? "" : "decoded world";
  if (output_capacity < decoded.size()) {
    return decoded.size();
  }
  if (!decoded.empty()) {
    std::copy(decoded.begin(), decoded.end(), output);
  }
  return decoded.size();
}

intptr_t tokenizers_hf_token_count(const void*) {
  return kRecords.size();
}

int32_t tokenizers_hf_token_at(
    const void*,
    size_t index,
    uint32_t* id,
    const uint8_t** text,
    size_t* text_len,
    uint8_t* is_added,
    uint8_t* is_special) {
  if (index >= kRecords.size()) {
    return -1;
  }
  const auto& record = kRecords[index];
  static const std::string kUnknownEos = "<stop>";
  const auto& record_text =
      index == 1 && !g_use_known_eos ? kUnknownEos : record.text;
  *id = record.id;
  *text = reinterpret_cast<const uint8_t*>(record_text.data());
  *text_len = record_text.size();
  *is_added = record.added;
  *is_special = record.special;
  return 0;
}

int32_t tokenizers_hf_post_token(const void*, uint8_t suffix, uint32_t* token) {
  if (suffix != 0) {
    return 1;
  }
  *token = 1;
  return 0;
}

uint32_t tokenizers_hf_config_flags(const void*) {
  return g_config_flags;
}

int32_t tokenizers_hf_has_native_decoder(const void*) {
  return g_native_decoder;
}

void tokenizers_hf_destroy(void* handle) {
  delete static_cast<uint8_t*>(handle);
}

} // extern "C"

namespace tokenizers {
namespace {

TEST(RustHFTokenizerTest, PreservesIndependentBosAndEosCounts) {
  g_config_flags = kByteLevelFlag;
  g_native_decoder = false;
  g_use_known_eos = true;
  TemporaryTokFile file;
  RustHFTokenizer tokenizer;
  ASSERT_EQ(tokenizer.load(file.string()), Error::Ok);
  EXPECT_EQ(tokenizer.bos_tok(), 1);
  EXPECT_EQ(tokenizer.eos_tok(), 2);

  auto tokens = tokenizer.encode("hello world", 2, 1);
  ASSERT_TRUE(tokens.ok());
  EXPECT_EQ(*tokens, (std::vector<uint64_t>{1, 1, 3, 4, 2}));

  auto bos_only_tokens = tokenizer.encode("hello world", 1, 0);
  ASSERT_TRUE(bos_only_tokens.ok());
  EXPECT_EQ(*bos_only_tokens, (std::vector<uint64_t>{1, 3, 4}));
}

TEST(RustHFTokenizerTest, DecodesByteLevelPieces) {
  g_config_flags = kByteLevelFlag;
  g_native_decoder = false;
  g_use_known_eos = true;
  TemporaryTokFile file;
  RustHFTokenizer tokenizer;
  ASSERT_EQ(tokenizer.load(file.string()), Error::Ok);

  auto decoded = tokenizer.decode(3, 4);
  ASSERT_TRUE(decoded.ok());
  EXPECT_EQ(*decoded, " world");

  auto skipped = tokenizer.decode(4, 2, true);
  ASSERT_TRUE(skipped.ok());
  EXPECT_TRUE(skipped->empty());
}

TEST(RustHFTokenizerTest, UsesNativeDecoderForJson) {
  g_config_flags = 0;
  g_native_decoder = true;
  g_use_known_eos = true;
  TemporaryTokFile file;
  RustHFTokenizer tokenizer;
  ASSERT_EQ(tokenizer.load(file.string()), Error::Ok);

  auto decoded = tokenizer.decode(3, 4);
  ASSERT_TRUE(decoded.ok());
  EXPECT_EQ(*decoded, "decoded world");

  auto skipped = tokenizer.decode(4, 2, true);
  ASSERT_TRUE(skipped.ok());
  EXPECT_TRUE(skipped->empty());
  g_config_flags = kByteLevelFlag;
  g_native_decoder = false;
}

TEST(RustHFTokenizerTest, RejectsTokWithoutSupportedDecoder) {
  g_use_known_eos = true;
  TemporaryTokFile file;
  RustHFTokenizer tokenizer;
  g_config_flags = 0;
  g_native_decoder = false;
  EXPECT_EQ(tokenizer.load(file.string()), Error::LoadFailure);
  EXPECT_FALSE(tokenizer.is_loaded());
  g_config_flags = kByteLevelFlag;
}

TEST(RustHFTokenizerTest, RejectsTokWithoutBosOrEosMetadata) {
  TemporaryTokFile file;
  RustHFTokenizer tokenizer;
  g_config_flags = kByteLevelFlag;
  g_use_known_eos = false;
  EXPECT_EQ(tokenizer.load(file.string()), Error::ParseFailure);
  EXPECT_FALSE(tokenizer.is_loaded());
  g_use_known_eos = true;
}

} // namespace
} // namespace tokenizers
