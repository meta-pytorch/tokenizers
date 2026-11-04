/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/token_decoder.h>

#include <pytorch/tokenizers/log.h>

#include <string_view>

// Third Party
#include <nlohmann/json.hpp>

// Local
#include <unicode.h>

using json = nlohmann::json;

namespace tokenizers {

// TokenDecoderConfig //////////////////////////////////////////////////////////

TokenDecoderConfig::TokenDecoderConfig(std::string type)
    : type(std::move(type)) {}

TokenDecoder::Ptr TokenDecoderConfig::create() const {
  // NOTE: These types must line up with the type strings found in the
  //  tokenizers library
  //  https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/mod.rs#L55
  if (type == "ByteLevel") {
    return TokenDecoder::Ptr(new ByteLevelTokenDecoder());
  } else if (type == "Replace") {
    // Use parsed pattern and content from JSON
    return TokenDecoder::Ptr(
        new ReplaceTokenDecoder(replace_pattern, replace_content));
  } else if (type == "ByteFallback") {
    return TokenDecoder::Ptr(new ByteFallbackTokenDecoder());
  } else if (type == "Fuse") {
    return TokenDecoder::Ptr(new FuseTokenDecoder());
  } else if (type == "Strip") {
    // Use parsed content, start, and stop from JSON
    return TokenDecoder::Ptr(
        new StripTokenDecoder(strip_content, strip_start, strip_stop));
  } else if (type == "Sequence") {
    // Parse the decoders array from JSON and create sub-decoders
    std::vector<TokenDecoder::Ptr> decoders;
    for (const auto& decoder_json : sequence_decoders) {
      TokenDecoderConfig sub_config;
      sub_config.parse_json(decoder_json);
      decoders.push_back(sub_config.create());
    }
    return TokenDecoder::Ptr(new SequenceTokenDecoder(std::move(decoders)));
  }
  throw std::runtime_error("Unsupported TokenDecoder type: " + type);
}

TokenDecoderConfig& TokenDecoderConfig::parse_json(const json& json_config) {
  type = json_config.at("type");
  if (type == "ByteLevel") {
    // No parameters to parse
  } else if (type == "Replace") {
    // Parse pattern and content for Replace decoder
    if (json_config.contains("pattern") && json_config.contains("content")) {
      if (json_config["pattern"].contains("String")) {
        replace_pattern = json_config["pattern"]["String"];
      }
      replace_content = json_config["content"];
    }
  } else if (type == "ByteFallback") {
    // No parameters to parse
  } else if (type == "Fuse") {
    // No parameters to parse
  } else if (type == "Strip") {
    // Parse content, start, and stop for Strip decoder
    if (json_config.contains("content")) {
      strip_content = json_config["content"];
    } else {
      throw std::runtime_error("Strip decoder 'content' is required.");
    }
    strip_start = json_config.value("start", 0);
    strip_stop = json_config.value("stop", 0);
  } else if (type == "Sequence") {
    // Parse decoders array for Sequence decoder
    if (json_config.contains("decoders")) {
      sequence_decoders = json_config["decoders"];
    }
  } else {
    throw std::runtime_error("Unsupported TokenDecoder type: " + type);
  }
  return *this;
}

// ByteLevel ///////////////////////////////////////////////////////////////////

std::vector<std::string> ByteLevelTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<uint8_t> all_bytes;

  for (const auto& token : tokens) {
    std::vector<uint8_t> current_token_bytes;
    bool all_chars_are_bytes = true;

    const auto cpts = unicode_cpts_from_utf8(token);
    for (const auto cpt : cpts) {
      const auto utf8 = unicode_cpt_to_utf8(cpt);
      try {
        current_token_bytes.push_back(unicode_utf8_to_byte(utf8));
      } catch (const std::out_of_range& /*e*/) {
        all_chars_are_bytes = false;
        break;
      }
    }

    if (all_chars_are_bytes) {
      all_bytes.insert(
          all_bytes.end(),
          current_token_bytes.begin(),
          current_token_bytes.end());
    } else {
      all_bytes.insert(all_bytes.end(), token.begin(), token.end());
    }
  }

  std::string final_string(all_bytes.begin(), all_bytes.end());
  return {final_string};
}

// ReplaceTokenDecoder ////////////////////////////////////////////////////////

ReplaceTokenDecoder::ReplaceTokenDecoder(
    const std::string& pattern,
    const std::string& content)
    : pattern_(pattern), content_(content) {}

std::vector<std::string> ReplaceTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded_tokens;
  decoded_tokens.reserve(tokens.size());
  for (const auto& token : tokens) {
    // Guard against empty pattern to prevent infinite loop
    if (pattern_.empty()) {
      decoded_tokens.push_back(token);
      continue;
    }

    std::string result = token;
    size_t pos = 0;
    while ((pos = result.find(pattern_, pos)) != std::string::npos) {
      result.replace(pos, pattern_.length(), content_);
      pos += content_.length();
    }
    decoded_tokens.push_back(result);
  }
  return decoded_tokens;
}

// ByteFallbackTokenDecoder ///////////////////////////////////////////////////

std::vector<std::string> ByteFallbackTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded_tokens;
  decoded_tokens.reserve(tokens.size());
  for (const auto& token : tokens) {
    // ByteFallback handles tokens that represent individual bytes
    // For tokens that start with <0x and end with >, extract the hex value
    if (token.length() >= 5 && token.substr(0, 3) == "<0x" &&
        token.back() == '>') {
      std::string hex_str = token.substr(3, token.length() - 4);
      try {
        unsigned long byte_val = std::stoul(hex_str, nullptr, 16);
        if (byte_val <= 255) {
          decoded_tokens.push_back(std::string(1, static_cast<char>(byte_val)));
          continue;
        }
      } catch (const std::exception&) {
        // Fall through to return original token
      }
    }
    decoded_tokens.push_back(token);
  }
  return decoded_tokens;
}

// StripTokenDecoder //////////////////////////////////////////////////////////

StripTokenDecoder::StripTokenDecoder(
    const std::string& content_str,
    size_t start,
    size_t stop)
    : content_(content_str), start_(start), stop_(stop) {
  if (content_str.length() == 0) {
    throw std::runtime_error("Strip decoder 'content' cannot be empty.");
  }
  const auto cpts = unicode_cpts_from_utf8(content_str);
  if (cpts.size() != 1) {
    throw std::runtime_error(
        "Strip decoder 'content' must represent a single Unicode character.");
  }
}

std::vector<std::string> StripTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> decoded_tokens;
  decoded_tokens.reserve(tokens.size());

  // Match content_ on bytes, not code points: after ByteFallback a token can be
  // an incomplete UTF-8 sequence, e.g. the first byte of an emoji.
  const size_t n = content_.size();
  for (const auto& token : tokens) {
    std::string_view rest(token);
    for (size_t i = 0; i < start_ && rest.substr(0, n) == content_; ++i) {
      rest.remove_prefix(n);
    }
    for (size_t i = 0; i < stop_ && rest.size() >= n; ++i) {
      if (rest.substr(rest.size() - n) != content_) {
        break;
      }
      rest.remove_suffix(n);
    }
    decoded_tokens.emplace_back(rest);
  }
  return decoded_tokens;
}

//  FuseTokenDecoder ///////////////////////////////////////////////////////
std::vector<std::string> FuseTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::string fusedToken = "";
  for (auto& token : tokens) {
    fusedToken += token;
  }
  return {fusedToken};
}

// SequenceTokenDecoder ///////////////////////////////////////////////////////

SequenceTokenDecoder::SequenceTokenDecoder(
    std::vector<TokenDecoder::Ptr> decoders)
    : decoders_(std::move(decoders)) {}

std::vector<std::string> SequenceTokenDecoder::decode(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> results = tokens;
  for (const auto& decoder : decoders_) {
    results = decoder->decode(results);
  }
  return results;
}

} // end  namespace tokenizers
