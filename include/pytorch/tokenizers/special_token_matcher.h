/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pytorch/tokenizers/regex.h>

namespace tokenizers {

/**
 * @brief Literal multi-string matcher for special/added tokens.
 *
 * The special-token set is a fixed list of literal strings, not a general
 * regex. Encoding a mega-alternation `(tok1|tok2|...|tokN)` into RE2 does not
 * scale: for vocabularies with thousands of added tokens (e.g. SID-style
 * `<sid_..._N>` tokens) the alternation blows RE2's DFA memory budget and
 * falls back to its NFA engine, making every encode O(N_vocab)-ish and
 * dominating request latency (measured: ~27ms to tokenize a 6KB SID prompt).
 *
 * A trie over the literal token strings matches in O(text_len * max_token_len)
 * — independent of vocabulary size — with no regex compilation. find_all
 * returns non-overlapping, leftmost-longest matches, which is the
 * HuggingFace AddedVocabulary semantics (longest added token wins at each
 * position) and is identical to the RE2 alternation whenever no special token
 * is a prefix of another (the common case, including all SID vocabularies).
 */
class SpecialTokenMatcher : public IRegex {
 public:
  SpecialTokenMatcher() {
    nodes_.emplace_back(); // root
  }

  explicit SpecialTokenMatcher(const std::vector<std::string>& tokens) {
    nodes_.emplace_back(); // root
    for (const auto& t : tokens) {
      insert_(t);
    }
  }

  // Not used — SpecialTokenMatcher is built from the literal token list via
  // the constructor, not from a pattern string. Present to satisfy IRegex.
  Error compile(const std::string& /*pattern*/) override {
    return Error::Ok;
  }

  std::vector<Match> find_all(const std::string& text) const override {
    std::vector<Match> matches;
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
      // Walk the trie from position i, remembering the end of the longest
      // token that terminates on the path (leftmost-longest at this start).
      int32_t node = 0;
      size_t longest_end = 0; // exclusive; 0 == no match starting at i
      for (size_t j = i; j < n; ++j) {
        const auto& children = nodes_[node].children;
        auto it = children.find(static_cast<unsigned char>(text[j]));
        if (it == children.end()) {
          break;
        }
        node = it->second;
        if (nodes_[node].terminal) {
          longest_end = j + 1;
        }
      }
      if (longest_end > i) {
        matches.push_back({i, longest_end});
        i = longest_end; // non-overlapping
      } else {
        ++i;
      }
    }
    return matches;
  }

 private:
  struct Node {
    std::unordered_map<unsigned char, int32_t> children;
    bool terminal = false;
  };

  void insert_(const std::string& token) {
    // Skip empty tokens intentionally: a zero-length token has no bytes to walk
    // and would mark the root terminal, which find_all() would then "match" at
    // every position as a zero-width match — corrupting offsets and looping.
    // (The old RE2 alternation had the same latent hazard: an empty branch
    // `(tok1||tok2)` yields zero-length matches.) Real added-token vocabularies
    // never contain an empty `content`, so this is a defensive no-op, not a
    // behavior change for any valid tokenizer.
    if (token.empty()) {
      return;
    }
    int32_t node = 0;
    for (const char c : token) {
      const auto byte = static_cast<unsigned char>(c);
      auto& children = nodes_[node].children;
      auto it = children.find(byte);
      if (it == children.end()) {
        const auto next = static_cast<int32_t>(nodes_.size());
        nodes_.emplace_back();
        // nodes_ may have reallocated; re-fetch the child map by index.
        nodes_[node].children.emplace(byte, next);
        node = next;
      } else {
        node = it->second;
      }
    }
    nodes_[node].terminal = true;
  }

  std::vector<Node> nodes_;
};

} // namespace tokenizers
