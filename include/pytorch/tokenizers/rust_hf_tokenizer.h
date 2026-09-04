/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <pytorch/tokenizers/string_integer_map.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace tokenizers {

class RustHFTokenizer final : public Tokenizer {
 public:
  RustHFTokenizer();
  ~RustHFTokenizer() override;

  Error load(const std::string& tokenizer_path) override;
  Result<std::string> id_to_piece(uint64_t token) const override;
  Result<uint64_t> piece_to_id(const std::string& text) const override;
  Result<std::vector<uint64_t>> encode(
      const std::string& input,
      int8_t bos = 0,
      int8_t eos = 0) const override;
  Result<std::string> decode(
      uint64_t prev_token,
      uint64_t token,
      bool skip_special_tokens = false) const override;

 private:
  using TokenMap = detail::StringIntegerMap<>;

  struct RustHandleDeleter {
    void operator()(void* handle) const;
  };

  Error load_metadata(const void* handle);

  std::unique_ptr<void, RustHandleDeleter> handle_;
  std::optional<TokenMap> token_map_;
  std::optional<TokenMap> added_token_map_;
  std::unordered_set<uint64_t> special_token_ids_;
  bool byte_level_ = false;
};

} // namespace tokenizers
