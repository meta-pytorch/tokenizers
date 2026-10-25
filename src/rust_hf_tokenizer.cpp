/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/rust_hf_tokenizer.h>

#include <filesystem>
#include <limits>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

extern "C" {
void* tokenizers_hf_create(const char* path);
intptr_t tokenizers_hf_encode(
    const void* handle,
    const uint8_t* text,
    size_t text_len,
    uint32_t* output,
    size_t output_capacity);
intptr_t tokenizers_hf_decode(
    const void* handle,
    uint32_t previous,
    uint8_t has_previous,
    uint32_t token,
    uint8_t skip_special,
    uint8_t* output,
    size_t output_capacity);
intptr_t tokenizers_hf_token_count(const void* handle);
int32_t tokenizers_hf_token_at(
    const void* handle,
    size_t index,
    uint32_t* id,
    const uint8_t** text,
    size_t* text_len,
    uint8_t* is_added,
    uint8_t* is_special);
int32_t
tokenizers_hf_post_token(const void* handle, uint8_t suffix, uint32_t* token);
uint32_t tokenizers_hf_config_flags(const void* handle);
void tokenizers_hf_destroy(void* handle);
}

namespace tokenizers {
namespace {

// Mirrors the byte-level flag returned by the Rust pipeline metadata.
constexpr uint32_t kByteLevelFlag = 1U << 2;

bool is_bos_token(std::string_view token) {
  return token == "<s>" || token == "[CLS]" || token == "<bos>" ||
      token == "<BOS>" || token == "<|bos|>" || token == "<|begin_of_text|>" ||
      token == "<|start_of_text|>" || token == "<|startoftext|>" ||
      token == "<|endoftext|>";
}

bool is_eos_token(std::string_view token) {
  return token == "</s>" || token == "[SEP]" || token == "<eos>" ||
      token == "<EOS>" || token == "<|eos|>" || token == "<|end_of_text|>" ||
      token == "<|endoftext|>";
}

std::optional<uint8_t> byte_level_codepoint_to_byte(uint32_t codepoint) {
  if ((codepoint >= 33 && codepoint <= 126) ||
      (codepoint >= 161 && codepoint <= 172) ||
      (codepoint >= 174 && codepoint <= 255)) {
    return static_cast<uint8_t>(codepoint);
  }
  if (codepoint < 256 || codepoint > 323) {
    return std::nullopt;
  }
  const auto index = codepoint - 256;
  if (index < 33) {
    return static_cast<uint8_t>(index);
  }
  if (index < 67) {
    return static_cast<uint8_t>(127 + index - 33);
  }
  return static_cast<uint8_t>(173);
}

std::string decode_byte_level(std::string_view piece) {
  std::string decoded;
  decoded.reserve(piece.size());
  for (size_t index = 0; index < piece.size();) {
    const auto first = static_cast<uint8_t>(piece[index]);
    uint32_t codepoint = 0;
    size_t length = 0;
    if ((first & 0x80) == 0) {
      codepoint = first;
      length = 1;
    } else if ((first & 0xE0) == 0xC0) {
      codepoint = first & 0x1F;
      length = 2;
    } else if ((first & 0xF0) == 0xE0) {
      codepoint = first & 0x0F;
      length = 3;
    } else if ((first & 0xF8) == 0xF0) {
      codepoint = first & 0x07;
      length = 4;
    } else {
      return std::string(piece);
    }
    if (length > piece.size() - index) {
      return std::string(piece);
    }
    for (size_t offset = 1; offset < length; ++offset) {
      const auto continuation = static_cast<uint8_t>(piece[index + offset]);
      if ((continuation & 0xC0) != 0x80) {
        return std::string(piece);
      }
      codepoint = (codepoint << 6) | (continuation & 0x3F);
    }
    const auto byte = byte_level_codepoint_to_byte(codepoint);
    if (!byte) {
      return std::string(piece);
    }
    decoded.push_back(static_cast<char>(*byte));
    index += length;
  }
  return decoded;
}

} // namespace

RustHFTokenizer::RustHFTokenizer() : handle_(nullptr) {}

RustHFTokenizer::~RustHFTokenizer() = default;

void RustHFTokenizer::RustHandleDeleter::operator()(void* handle) const {
  tokenizers_hf_destroy(handle);
}

Error RustHFTokenizer::load(const std::string& path) {
  initialized_ = false;
  handle_.reset();
  token_map_.reset();
  added_token_map_.reset();
  special_token_ids_.clear();
  has_bos_token_ = false;
  has_eos_token_ = false;
  byte_level_ = false;
  vocab_size_ = 0;
  bos_tok_ = 0;
  eos_tok_ = 0;

  std::error_code fs_error;
  if (!fs::exists(path, fs_error) || fs_error) {
    return Error::LoadFailure;
  }

  std::unique_ptr<void, RustHandleDeleter> handle(
      tokenizers_hf_create(path.c_str()));
  if (!handle) {
    return Error::LoadFailure;
  }

  const auto flags = tokenizers_hf_config_flags(handle.get());
  if (flags == std::numeric_limits<uint32_t>::max()) {
    return Error::ParseFailure;
  }
  byte_level_ = (flags & kByteLevelFlag) != 0;

  const auto metadata_error = load_metadata(handle.get());
  if (metadata_error != Error::Ok) {
    return metadata_error;
  }

  handle_ = std::move(handle);
  initialized_ = true;
  return Error::Ok;
}

Error RustHFTokenizer::load_metadata(const void* handle) {
  const auto count = tokenizers_hf_token_count(handle);
  if (count < 0 ||
      static_cast<uintmax_t>(count) >
          static_cast<uintmax_t>(std::numeric_limits<int32_t>::max())) {
    return Error::ParseFailure;
  }

  struct TokenRecord {
    std::string text;
    uint64_t id;
    bool added;
    bool special;
  };
  std::vector<TokenRecord> records;
  records.reserve(static_cast<size_t>(count));
  std::unordered_set<uint64_t> added_ids;
  for (size_t index = 0; index < static_cast<size_t>(count); ++index) {
    uint32_t id = 0;
    const uint8_t* text = nullptr;
    size_t text_len = 0;
    uint8_t is_added = 0;
    uint8_t is_special = 0;
    if (tokenizers_hf_token_at(
            handle, index, &id, &text, &text_len, &is_added, &is_special) !=
            0 ||
        (text == nullptr && text_len != 0)) {
      return Error::ParseFailure;
    }
    records.push_back(
        {std::string(reinterpret_cast<const char*>(text), text_len),
         id,
         is_added != 0,
         is_special != 0});
    if (is_added != 0) {
      added_ids.insert(id);
      if (is_special != 0) {
        special_token_ids_.insert(id);
      }
    }
  }

  std::vector<std::pair<std::string, uint64_t>> tokens;
  std::vector<std::pair<std::string, uint64_t>> added_tokens;
  std::vector<uint64_t> bos_candidates;
  std::vector<uint64_t> eos_candidates;
  tokens.reserve(records.size());
  added_tokens.reserve(added_ids.size());
  for (auto& record : records) {
    if (record.special) {
      if (is_bos_token(record.text)) {
        bos_candidates.push_back(record.id);
      }
      if (is_eos_token(record.text)) {
        eos_candidates.push_back(record.id);
      }
    }
    if (record.added) {
      added_tokens.emplace_back(std::move(record.text), record.id);
    } else if (added_ids.count(record.id) == 0) {
      tokens.emplace_back(std::move(record.text), record.id);
    }
  }
  auto added_map = TokenMap::create(added_tokens);
  if (!added_map.ok()) {
    return added_map.error();
  }
  added_token_map_.emplace(std::move(*added_map));
  auto token_map = TokenMap::create(tokens);
  if (!token_map.ok()) {
    return token_map.error();
  }
  token_map_.emplace(std::move(*token_map));
  vocab_size_ =
      static_cast<int32_t>(token_map_->size() + added_token_map_->size());

  uint32_t bos = 0;
  uint32_t eos = 0;
  const auto bos_status = tokenizers_hf_post_token(handle, 0, &bos);
  const auto eos_status = tokenizers_hf_post_token(handle, 1, &eos);
  if (bos_status < 0 || eos_status < 0) {
    return Error::ParseFailure;
  }
  if (bos_status == 0) {
    bos_tok_ = bos;
    has_bos_token_ = true;
  }
  if (eos_status == 0) {
    eos_tok_ = eos;
    has_eos_token_ = true;
  }
  if (!has_bos_token_ || !has_eos_token_) {
    if (!has_bos_token_ && bos_candidates.size() == 1) {
      bos_tok_ = bos_candidates.front();
      has_bos_token_ = true;
    }
    if (!has_eos_token_ && eos_candidates.size() == 1) {
      eos_tok_ = eos_candidates.front();
      has_eos_token_ = true;
    }
  }
  if (!has_bos_token_ || !has_eos_token_) {
    return Error::ParseFailure;
  }
  return Error::Ok;
}

Result<std::string> RustHFTokenizer::id_to_piece(uint64_t token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  if (auto piece = token_map_->tryGetString(token)) {
    return std::string(*piece);
  }
  if (auto piece = added_token_map_->tryGetString(token)) {
    return std::string(*piece);
  }
  return Error::OutOfRange;
}

Result<uint64_t> RustHFTokenizer::piece_to_id(const std::string& text) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  if (auto id = token_map_->tryGetInteger(text)) {
    return *id;
  }
  if (auto id = added_token_map_->tryGetInteger(text)) {
    return *id;
  }
  return Error::OutOfRange;
}

Result<std::vector<uint64_t>> RustHFTokenizer::encode(
    const std::string& input,
    int8_t bos,
    int8_t eos) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  if (input.size() >
      static_cast<size_t>(std::numeric_limits<intptr_t>::max())) {
    return Error::EncodeFailure;
  }
  if ((bos > 0 && !has_bos_token_) || (eos > 0 && !has_eos_token_)) {
    return Error::EncodeFailure;
  }

  std::vector<uint32_t> output(input.size());
  auto count = tokenizers_hf_encode(
      handle_.get(),
      reinterpret_cast<const uint8_t*>(input.data()),
      input.size(),
      output.data(),
      output.size());
  if (count < 0) {
    return Error::EncodeFailure;
  }
  if (static_cast<size_t>(count) > output.size()) {
    output.resize(static_cast<size_t>(count));
    count = tokenizers_hf_encode(
        handle_.get(),
        reinterpret_cast<const uint8_t*>(input.data()),
        input.size(),
        output.data(),
        output.size());
    if (count < 0 || static_cast<size_t>(count) > output.size()) {
      return Error::EncodeFailure;
    }
  }
  output.resize(static_cast<size_t>(count));
  const auto bos_count = bos > 0 ? static_cast<size_t>(bos) : 0;
  const auto eos_count = eos > 0 ? static_cast<size_t>(eos) : 0;
  std::vector<uint64_t> tokens;
  tokens.reserve(output.size() + bos_count + eos_count);
  tokens.insert(tokens.end(), bos_count, bos_tok_);
  tokens.insert(tokens.end(), output.begin(), output.end());
  tokens.insert(tokens.end(), eos_count, eos_tok_);
  return tokens;
}

Result<std::string> RustHFTokenizer::decode(
    uint64_t prev_token,
    uint64_t token,
    bool skip_special_tokens) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  if (!byte_level_) {
    if (token > std::numeric_limits<uint32_t>::max() ||
        prev_token > std::numeric_limits<uint32_t>::max()) {
      return Error::DecodeFailure;
    }
    auto count = tokenizers_hf_decode(
        handle_.get(),
        static_cast<uint32_t>(prev_token),
        prev_token != 0,
        static_cast<uint32_t>(token),
        skip_special_tokens,
        nullptr,
        0);
    if (count < 0) {
      return Error::DecodeFailure;
    }
    std::string decoded(static_cast<size_t>(count), '\0');
    count = tokenizers_hf_decode(
        handle_.get(),
        static_cast<uint32_t>(prev_token),
        prev_token != 0,
        static_cast<uint32_t>(token),
        skip_special_tokens,
        reinterpret_cast<uint8_t*>(decoded.data()),
        decoded.size());
    if (count < 0 || static_cast<size_t>(count) > decoded.size()) {
      return Error::DecodeFailure;
    }
    decoded.resize(static_cast<size_t>(count));
    return decoded;
  }

  std::string_view piece;
  if (auto regular = token_map_->tryGetString(token)) {
    piece = *regular;
  } else if (auto added = added_token_map_->tryGetString(token)) {
    if (skip_special_tokens && special_token_ids_.count(token) != 0) {
      return std::string();
    }
    piece = *added;
  } else {
    return Error::DecodeFailure;
  }

  return decode_byte_level(piece);
}

} // namespace tokenizers
