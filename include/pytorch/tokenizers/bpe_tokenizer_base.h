/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Base class for all BPE tokenizer implementations
#pragma once

// Standard
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Local
#include <pytorch/tokenizers/error.h>
#include <pytorch/tokenizers/regex.h>
#include <pytorch/tokenizers/result.h>
#include <pytorch/tokenizers/special_token_matcher.h>
#include <pytorch/tokenizers/string_integer_map.h>
#include <pytorch/tokenizers/tokenizer.h>

#include "re2/re2.h"

namespace tokenizers {
namespace detail {

using TokenMap = StringIntegerMap<>;

template <typename TToken, typename TRank>
static Result<TokenMap> build_token_map(
    std::vector<std::pair<TToken, TRank>> container) {
  static_assert(
      std::is_same_v<TToken, std::string> ||
          std::is_same_v<TToken, std::string_view>,
      "TToken must be std::string or std::string_view");
  static_assert(
      std::is_integral_v<TRank> && std::is_unsigned_v<TRank>,
      "TRank must be an unsigned integer");

  return TokenMap::create(container);
};

template <typename TContainer, typename TTokenAccessor, typename TRankAccessor>
static Result<TokenMap> build_token_map(
    const TContainer& container,
    TTokenAccessor token_accessor,
    TRankAccessor rank_accessor) {
  using TokenType = std::invoke_result_t<TTokenAccessor, const TContainer&>;
  using RankType = std::invoke_result_t<TRankAccessor, const TContainer&>;

  static_assert(
      std::is_same_v<TokenType, std::string> ||
          std::is_same_v<TokenType, std::string_view>,
      "TokenType must be std::string or std::string_view");
  static_assert(
      std::is_integral_v<RankType> && std::is_unsigned_v<RankType>,
      "RankType must be an unsigned integer");

  std::vector<std::pair<TokenType, RankType>> pairs;
  pairs.reserve(container.size());
  for (const auto& value : container) {
    pairs.emplace_back(token_accessor(value), rank_accessor(value));
  }

  return build_token_map(std::move(pairs));
}

inline Result<std::unique_ptr<IRegex>> build_special_token_regex(
    const TokenMap& special_token_map) {
  const std::size_t count = special_token_map.size();
  if (count == 0) {
    return static_cast<std::unique_ptr<IRegex>>(nullptr);
  }

  // Special tokens are a fixed set of literal strings, not a general regex.
  // Encoding them as an RE2 alternation `(tok1|tok2|...|tokN)` does not scale:
  // for large added-token vocabularies (e.g. thousands of SID-style tokens)
  // the alternation exceeds RE2's DFA memory budget and falls back to the slow
  // NFA engine, dominating encode latency. A trie matcher matches in
  // O(text_len * max_token_len), independent of vocabulary size (see
  // SpecialTokenMatcher), and returns non-overlapping leftmost-longest matches
  // — identical to the alternation whenever no special token is a prefix of
  // another (always true for SID and standard chat-marker vocabularies) and
  // matching HuggingFace AddedVocabulary semantics otherwise.
  std::vector<std::string> tokens;
  tokens.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& [token, _] = special_token_map.getElement(i);
    tokens.emplace_back(token);
  }
  std::unique_ptr<IRegex> matcher =
      std::make_unique<SpecialTokenMatcher>(tokens);
  return matcher;
}

class BPETokenizerBase : public Tokenizer {
 public:
  Result<std::vector<uint64_t>>
  encode(const std::string& input, int8_t bos, int8_t eos) const override;

  Result<std::string> id_to_piece(uint64_t token) const override;
  Result<uint64_t> piece_to_id(const std::string& text) const override;

  Result<std::string> decode(
      uint64_t prev_token,
      uint64_t token,
      bool skip_special_tokens = false) const override;

 protected:
  explicit BPETokenizerBase() {}
  virtual ~BPETokenizerBase() override {}

  std::pair<std::optional<std::string>, std::string>
  split_with_allowed_special_token_(
      const std::string& input,
      const TokenMap& allowed_special) const;

  std::pair<std::optional<std::string>, std::string>
  split_with_allowed_special_token_(
      const std::string& input,
      size_t offset,
      const TokenMap& allowed_special) const;

  Result<std::pair<std::vector<uint64_t>, uint64_t>> encode_with_special_token_(
      const std::string& text,
      const TokenMap& allowed_special) const;

  virtual Result<std::vector<uint64_t>> byte_pair_encode_(
      const std::string& piece,
      const TokenMap& encoder) const;

  // Virtual method for BPE merging - can be overridden by derived classes
  // The passed in `ranks` param for the base impl is just a regular token map
  // and that the actual ranks are derived implicitly from the regular token
  // map. This is the same implementation as Tiktoken.
  virtual std::vector<uint64_t> _byte_pair_merge(
      const std::string& piece,
      const TokenMap& ranks,
      std::function<uint64_t(uint64_t, uint64_t)> func) const;

  // Protected members that can be overloaded by other BPE tokenizers
  std::unique_ptr<IRegex> special_token_regex_;
  std::optional<TokenMap> token_map_;
  std::optional<TokenMap> special_token_map_;

 private:
  virtual Error _encode(
      const std::string& input,
      std::vector<uint64_t>& ret,
      uint64_t& last_piece_token_len) const = 0;

  virtual void _decode(const std::string& input, std::string& ret) const = 0;
};

} // namespace detail
} // namespace tokenizers
