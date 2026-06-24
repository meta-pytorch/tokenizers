/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

// Local
#include <pytorch/tokenizers/normalizer.h>
#include <pytorch/tokenizers/regex.h>

// Third Party
#include <unicode.h>

// Standard
#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace tokenizers {

// NormalizerConfig ////////////////////////////////////////////////////////////

NormalizerConfig::NormalizerConfig(std::string type) : type(std::move(type)) {}

Normalizer::Ptr NormalizerConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/normalizers/mod.rs
  if (type == "Replace") {
    if (!pattern) {
      throw std::runtime_error(
          "Missing pattern for Normalizer of type Replace");
    }
    if (!content) {
      throw std::runtime_error(
          "Missing content for Normalizer of type Replace");
    }
    return Normalizer::Ptr(new ReplaceNormalizer(*pattern, *content));
  }
  if (type == "Prepend") {
    if (!prepend) {
      throw std::runtime_error(
          "Missing prepend for Normalizer of type Prepend");
    }
    return Normalizer::Ptr(new PrependNormalizer(*prepend));
  }
  if (type == "Sequence") {
    if (!normalizers || normalizers->empty()) {
      throw std::runtime_error(
          "Missing normalizers for Normalizer of type Sequence");
    }
    std::vector<Normalizer::Ptr> norms;
    std::transform(
        normalizers->begin(),
        normalizers->end(),
        std::back_inserter(norms),
        [](const NormalizerConfig& cfg) { return cfg.create(); });
    return Normalizer::Ptr(new SequenceNormalizer(norms));
  }
  if (type == "NFC") {
    return Normalizer::Ptr(new NFCNormalizer());
  }
  throw std::runtime_error("Unsupported Normalizer type: " + type);
}

NormalizerConfig& NormalizerConfig::parse_json(const json& json_config) {
  type = json_config.at("type");
  if (type == "Replace") {
    try {
      pattern = json_config.at("pattern").at("Regex");
    } catch (json::out_of_range&) {
      // "Regex" is not there, check "String", which is a literal string
      std::string literal = json_config.at("pattern").at("String");
      // For string patterns, escape regex special characters to treat them as
      // literal strings (same as Rust's regex::escape)
      pattern = IRegex::escape(literal);
    }

    content = json_config.at("content");
  } else if (type == "Prepend") {
    prepend = json_config.at("prepend");
  } else if (type == "Sequence") {
    normalizers = std::vector<NormalizerConfig>();
    for (const auto& entry : json_config.at("normalizers")) {
      normalizers->push_back(NormalizerConfig().parse_json(entry));
    }
  } else if (type == "NFC") {
    // NFC normalizer has no additional configuration parameters
    TK_LOG(
        Info,
        "Using NFC normalizer. Please notice that our implementation may not handle all edge cases.");
  } else {
    throw std::runtime_error("Unsupported Normalizer type: " + type);
  }
  return *this;
}

// ReplaceNormalizer ///////////////////////////////////////////////////////////

std::unique_ptr<IRegex> ReplaceNormalizer::create_regex_(
    const std::string& pattern) {
  assert(!pattern.empty());
  auto regex_result = create_regex(pattern);
  if (!regex_result.ok()) {
    std::string error =
        "Error: " + std::to_string(static_cast<int>(regex_result.error()));
    throw std::runtime_error(error);
  }
  return std::move(regex_result.get());
}

std::string ReplaceNormalizer::normalize(const std::string& input) const {
  if (!regex_) {
    return input;
  }

  auto matches = regex_->find_all(input);
  if (matches.empty()) {
    return input;
  }

  // Single forward pass: copy each unmatched span then the replacement. The
  // previous code mutated the string in place once per match, shifting the
  // tail each time (O(N^2)); this is O(N) with a single allocation. Relies on
  // find_all returning matches in left-to-right order with non-overlapping,
  // non-decreasing offsets (see IRegex::find_all).
  std::string result;
  // Exact output size: each match replaces [start, end) with content_. Computed
  // by addition only (no multiplication) so it cannot overflow before the
  // matches themselves would.
  size_t out_size = input.size();
  for (const auto& match : matches) {
    out_size += content_.size();
    out_size -= match.end - match.start;
  }
  result.reserve(out_size);
  size_t prev_end = 0;
  for (const auto& match : matches) {
    // Guard the find_all ordering contract so a misbehaving regex wrapper
    // fails here rather than underflowing match.start - prev_end.
    assert(match.start >= prev_end && match.end >= match.start);
    result.append(input, prev_end, match.start - prev_end);
    result.append(content_);
    prev_end = match.end;
  }
  result.append(input, prev_end, input.size() - prev_end);

  return result;
}

// PrependNormalizer ///////////////////////////////////////////////////////////

std::string PrependNormalizer::normalize(const std::string& input) const {
  if (input.empty()) {
    return "";
  }
  return prepend_ + input;
}

// SequenceNormalizer //////////////////////////////////////////////////////////

SequenceNormalizer::SequenceNormalizer(std::vector<Normalizer::Ptr> normalizers)
    : normalizers_(std::move(normalizers)) {}

std::string SequenceNormalizer::normalize(const std::string& input) const {
  std::string result = input;
  for (const auto& normalizer : normalizers_) {
    result = normalizer->normalize(result);
  }
  return result;
}

// NFCNormalizer ///////////////////////////////////////////////////////////////

std::string NFCNormalizer::normalize(const std::string& input) const {
  // Convert UTF-8 string to codepoints
  auto codepoints = unicode_cpts_from_utf8(input);

  // Apply NFC normalization
  auto normalized_cpts = unicode_cpts_normalize_nfc(codepoints);

  // Convert back to UTF-8 string
  std::string result;
  for (uint32_t cpt : normalized_cpts) {
    result += unicode_cpt_to_utf8(cpt);
  }

  return result;
}

} // namespace tokenizers
