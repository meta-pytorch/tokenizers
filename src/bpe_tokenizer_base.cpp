/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/bpe_tokenizer_base.h>

// Standard
#include <inttypes.h>
#include <cstdint>
#include <functional>

namespace tokenizers {
namespace detail {

// ---- Helper utils start -----------------------------------------------------
namespace {

static uint64_t _max_size() {
  return std::numeric_limits<uint64_t>::max();
}

} // namespace

// ---- Helper utils end -------------------------------------------------------
// ---- protected start --------------------------------------------------------

std::vector<uint64_t> BPETokenizerBase::_byte_pair_merge(
    const std::string& piece,
    const TokenMap& ranks,
    std::function<uint64_t(uint64_t, uint64_t)> func) const {
  // This is a vector of (start, rank).
  // The rank is of the byte pair starting at position start.
  // The rank of the last item in the vector is not a valid value.
  std::vector<std::pair<uint64_t, uint64_t>> parts;
  parts.reserve(piece.size() + 1);
  for (auto idx = 0U; idx < piece.size() + 1; ++idx) {
    parts.emplace_back(idx, _max_size());
  }

  auto get_rank = [&piece, &ranks](
                      const std::vector<std::pair<uint64_t, uint64_t>>& parts,
                      uint64_t start_idx,
                      uint64_t skip) -> std::optional<uint64_t> {
    if (start_idx + skip + 2 < parts.size()) {
      auto s = parts[start_idx].first;
      auto e = parts[start_idx + skip + 2].first;
      auto key = piece.substr(s, e - s);
      return ranks.tryGetInteger(key);
    }
    return std::nullopt;
  };

  // We look up the ranks once in the beginning and iteratively update
  // them during each merge, which reduces the number of rank lookups.
  for (auto i = 0U; i < parts.size() - 2; ++i) {
    auto rank = get_rank(parts, i, 0);
    if (rank) {
      // usize::MAX is a sentinel value and cannot be a valid rank
      if (*rank == _max_size()) {
        TK_LOG(Error, "at %" PRIu32 " rank is too large\n", i);
      }
      parts[i].second = *rank;
    }
  }

  // If you have n parts and m merges, this does O(mn) work.
  // We could do something with a heap and do O(m log n) work.
  // It is important to consider that n is often small (<100), and as such
  // the cache-locality benefits outweigh the algorithmic complexity downsides
  // of the `parts` vector data structure above.

  // Note that we hash bytes, not token pairs. As long as we train BPE the way
  // we currently do, this is equivalent. An easy way to break this would be
  // to decouple merge priority from token index or to prevent specific token
  // merges.
  while (true) {
    if (parts.size() == 1) {
      break;
    }

    // usize::MAX is a sentinel rank value allowing us to
    // take the min more quickly
    auto min_rank = std::make_pair<uint64_t, uint64_t>(_max_size(), 0);
    for (auto i = 0U; i < parts.size() - 1; ++i) {
      auto rank = parts[i].second;
      if (rank < min_rank.first) {
        min_rank.first = rank;
        min_rank.second = i;
      }
    }

    if (min_rank.first != _max_size()) {
      auto i = min_rank.second;

      // NOTE: We are about to remove parts[i + 1]. We do not do it
      // yet because there are cache-locality benefits to updating
      // parts[i] and parts[i-1] before removing, which could thrash
      // the cache. Thus, we update the rank calculation by skipping over
      // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
      auto rank = get_rank(parts, i, 1);
      if (rank) {
        parts[i].second = *rank;
      } else {
        parts[i].second = _max_size();
      }
      if (i > 0) {
        rank = get_rank(parts, i - 1, 1);
        if (rank) {
          parts[i - 1].second = *rank;
        } else {
          parts[i - 1].second = _max_size();
        }
      }

      parts.erase(parts.begin() + (i + 1));
    } else {
      break;
    }
  }
  std::vector<uint64_t> out;
  out.reserve(parts.size() - 1);
  for (auto i = 0U; i < parts.size() - 1; ++i) {
    auto s = parts[i].first;
    auto e = parts[i + 1].first;
    out.push_back(func(s, e));
  }
  return out;
}

std::pair<std::optional<std::string>, std::string>
BPETokenizerBase::split_with_allowed_special_token_(
    const std::string& input,
    size_t offset,
    const TokenMap& allowed_special) const {
  if (!special_token_regex_) {
    return std::make_pair(std::nullopt, input.substr(offset));
  }

  auto matches = special_token_regex_->find_all(input.substr(offset));

  for (const auto& m : matches) {
    std::string matched_text = input.substr(offset + m.start, m.end - m.start);
    if (allowed_special.tryGetInteger(matched_text).has_value()) {
      return {matched_text, input.substr(offset, m.start)};
    }
  }

  return {std::nullopt, input.substr(offset)};
}

Result<std::pair<std::vector<uint64_t>, uint64_t>>
BPETokenizerBase::encode_with_special_token_(
    const std::string& text,
    const TokenMap& allowed_special) const {
  std::vector<uint64_t> tokens;
  uint64_t last_piece_token_len = 0;

  // The original implementation guarded the entire encode loop with
  // `while (offset < text.size())`, so empty input never reached _encode.
  // Preserve that: skip _encode entirely for empty text.
  if (text.empty()) {
    return std::make_pair(tokens, last_piece_token_len);
  }

  // Fast path: scan the input ONCE for every special-token match, then walk
  // the precomputed match list in order. The previous implementation called
  // find_all on a fresh substring per outer iteration but used only the
  // first match per call, giving O(N_special * text_len) regex work and
  // O(N_special) std::string copies on prompts with many specials.
  //
  // Semantics preserved: matches whose text is not in `allowed_special` are
  // skipped (treated as part of the surrounding text piece), matching the
  // behavior of split_with_allowed_special_token_.
  if (special_token_regex_) {
    const auto all_matches = special_token_regex_->find_all(text);
    size_t offset = 0;
    for (const auto& m : all_matches) {
      // A skipped (disallowed) special earlier may have left `offset`
      // beyond this match — drop matches that overlap already-consumed
      // text.
      if (m.start < offset) {
        continue;
      }
      // Filter by allowed_special: the special-token regex matches every
      // registered special, but the caller may have restricted which ones
      // are actually special-tokenized. Disallowed matches fall through
      // and are handled as ordinary text by _encode below.
      std::string_view matched_text(text.data() + m.start, m.end - m.start);
      const auto sid = allowed_special.tryGetInteger(matched_text);
      if (!sid.has_value()) {
        continue;
      }
      // Encode the regular-text piece [offset .. m.start) before the
      // special token. The piece may be empty (special at offset 0 or two
      // specials adjacent); the original loop called _encode unconditionally
      // in this position and HFTokenizer's normalizer/pretokenizer pipeline
      // can have observable effects on "", so preserve that call here.
      std::string piece(text.data() + offset, m.start - offset);
      TK_CHECK_OK_OR_RETURN_ERROR(_encode(piece, tokens, last_piece_token_len));
      tokens.push_back(*sid);
      last_piece_token_len = 0;
      offset = m.end;
    }
    // Encode the trailing piece [offset .. text.size()) only if non-empty.
    // The original loop exited via `break` (no trailing _encode) when the
    // input ended exactly on a special token, so don't synthesize one here.
    if (offset < text.size()) {
      std::string tail(text.data() + offset, text.size() - offset);
      TK_CHECK_OK_OR_RETURN_ERROR(_encode(tail, tokens, last_piece_token_len));
    }
  } else {
    // No special-token regex configured: fall back to encoding the whole
    // text as a single piece, matching the original loop's behavior when
    // split_with_allowed_special_token_ returns {nullopt, full input}.
    // Empty-text early-return above ensures we never call _encode("") here.
    TK_CHECK_OK_OR_RETURN_ERROR(_encode(text, tokens, last_piece_token_len));
  }

  return std::make_pair(tokens, last_piece_token_len);
}

Result<std::vector<uint64_t>> BPETokenizerBase::byte_pair_encode_(
    const std::string& piece,
    const TokenMap& token_map) const {
  if (piece.size() == 1) {
    const auto result = token_map.tryGetInteger(piece);
    if (result) {
      return std::vector<uint64_t>(1, *result);
    } else {
      TK_LOG(Error, "unknown token: '%s'", piece.c_str());
      return Error::EncodeFailure;
    }
  }

  // Use the original _byte_pair_merge function with the proper merge ranks
  return _byte_pair_merge(
      piece, token_map, [&piece, &token_map](uint64_t start, uint64_t stop) {
        std::string key = piece.substr(start, stop - start);
        const auto result = token_map.tryGetInteger(key);
        if (result) {
          return *result;
        } else {
          TK_LOG(Error, "BPE merge produced unknown token: '%s'", key.c_str());
          return uint64_t(0); // Return unknown token ID instead of padding
        }
      });
}

// ---- protected end ----------------------------------------------------------
// ---- public start -----------------------------------------------------------

Result<std::vector<uint64_t>> BPETokenizerBase::encode(
    const std::string& text,
    int8_t bos,
    int8_t eos) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  auto encode_result = encode_with_special_token_(text, *special_token_map_);
  if (!encode_result.ok()) {
    return encode_result.error();
  }
  auto res = std::move((*encode_result).first);
  for (auto i = 0; i < bos; ++i) {
    res.insert(res.begin(), bos_tok_);
  }
  for (auto i = 0; i < eos; ++i) {
    res.push_back(eos_tok_);
  }
  return Result<std::vector<uint64_t>>(std::move(res));
}

Result<std::string> BPETokenizerBase::id_to_piece(uint64_t token) const {
  if (!initialized_) {
    return Error::Uninitialized;
  }
  if (!token_map_.has_value() || !special_token_map_.has_value()) {
    return Error::Internal;
  }

  auto result = token_map_->tryGetString(token);
  if (!result) {
    result = special_token_map_->tryGetString(token);
  }
  if (!result) {
    return Error::OutOfRange;
  }
  return std::string(*result);
}

Result<uint64_t> BPETokenizerBase::piece_to_id(const std::string& text) const {
  if (!initialized_) {
    TK_LOG(Error, "Tokenizer not initialized");
    return Error::Uninitialized;
  }
  if (!token_map_.has_value() || !special_token_map_.has_value()) {
    return Error::Internal;
  }
  auto result = token_map_->tryGetInteger(text);
  if (!result) {
    result = special_token_map_->tryGetInteger(text);
  }
  if (!result) {
    TK_LOG(Debug, "Piece '%s' not found in vocabulary", text.c_str());
    return Error::OutOfRange;
  }
  return *result;
}

Result<std::string> BPETokenizerBase::decode(
    uint64_t prev,
    uint64_t cur,
    bool skip_special_tokens) const {
  (void)prev;
  if (!initialized_) {
    return Error::Uninitialized;
  }
  std::string ret;

  std::string_view token_bytes;
  auto regular_token_result = token_map_->tryGetString(cur);
  if (regular_token_result) { // Found in regular tokens
    token_bytes = *regular_token_result;
  } else { // Not a regular token, check if it's a special token
    auto special_token_result = special_token_map_->tryGetString(cur);
    if (special_token_result) { // It's a special token
      if (skip_special_tokens) {
        return std::string(""); // Skip it
      }
      token_bytes = *special_token_result; // Don't skip, use its string
    } else { // Unknown token
      TK_LOG(Error, "unknown token: %" PRIu64 "\n", cur);
      return Error::DecodeFailure;
    }
  }
  _decode(std::string(token_bytes), ret);

  return ret;
}

// ---- public end -------------------------------------------------------------

} // namespace detail
} // namespace tokenizers
