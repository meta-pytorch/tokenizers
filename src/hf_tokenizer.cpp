/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// @lint-ignore-every LICENSELINT

#include <pytorch/tokenizers/hf_tokenizer.h>

// Standard
#include <algorithm>
#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Third Party
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace tokenizers {

namespace {
// Helper to extract token string from either string or object format
std::string extract_token_string(const json& token_json) {
  if (token_json.is_string()) {
    return token_json.get<std::string>();
  } else if (token_json.is_object() && token_json.contains("content")) {
    return token_json["content"].get<std::string>();
  }
  return "";
};
} // namespace
// -------------------------private method end-------------------------------
// -------------------------public method start-------------------------------

Error HFTokenizer::load(const std::string& path) {
  // If this is a directory, look for tokenizer.json and tokenizer_config.json
  std::string model_json = path;
  std::string model_config_json = "";
  std::string special_tokens_map_json;

  // Check if bos/eos found.
  bool bos_found = false;
  bool eos_found = false;

  if (fs::is_directory(path)) {
    const fs::path root(path);
    model_json = (root / "tokenizer.json").string();
    if (!fs::exists(model_json)) {
      TK_LOG(Info, "no tokenizer.json found in %s", path.c_str());
      return Error::LoadFailure;
    }
    const auto model_config_json_path = root / "tokenizer_config.json";
    if (fs::exists(model_config_json_path)) {
      model_config_json = model_config_json_path.string();
    }

    const auto special_tokens_map_json_path = root / "special_tokens_map.json";
    if (fs::exists(special_tokens_map_json_path)) {
      special_tokens_map_json = special_tokens_map_json_path.string();
    }
  }

  // Load the tokenizer.json file
  std::ifstream file(model_json);
  if (!file) {
    TK_LOG(Info, "failed to open encoder file: %s", path.c_str());
    return Error::LoadFailure;
  }
  std::string contents(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  json parsed_json;
  try {
    parsed_json = json::parse(contents);
  } catch (const std::exception& e) {
    TK_LOG(Error, "Error parsing json file: %s", e.what());
    return Error::LoadFailure;
  }

  // Parse the special tokens
  try {
    const auto& special_tokens = parsed_json.at("added_tokens");
    auto special_token_map_result = detail::build_token_map(
        special_tokens,
        [](const auto& it) -> std::string { return it.at("content"); },
        [](const auto& it) -> std::uint64_t { return it.at("id"); });
    if (!special_token_map_result.ok()) {
      return special_token_map_result.error();
    }
    auto special_token_map = std::move(*special_token_map_result);

    // Create special token regex to help later with encoding.
    auto special_token_regex_result =
        detail::build_special_token_regex(special_token_map);
    if (!special_token_regex_result.ok()) {
      return special_token_regex_result.error();
    }
    special_token_regex_ = std::move(*special_token_regex_result);

    // Store for future use.
    special_token_map_.emplace(std::move(special_token_map));
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse special tokens: %s", e.what());
    return Error::LoadFailure;
  }

  // Parse the standard tokens
  try {
    std::vector<std::pair<std::string, std::uint64_t>> token_pairs;
    const auto& vocab = parsed_json.at("/model/vocab"_json_pointer);
    for (const auto& entry : vocab.items()) {
      const std::string token = entry.key();
      const uint64_t token_id = entry.value();
      // Skip adding special tokens to the standard encoder/decoder
      if (!special_token_map_->tryGetString(token_id)) {
        token_pairs.emplace_back(token, token_id);
      }
    }

    auto token_map_result = detail::build_token_map(std::move(token_pairs));
    if (!token_map_result.ok()) {
      return token_map_result.error();
    }
    auto token_map = std::move(*token_map_result);
    token_map_.emplace(std::move(token_map));
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse tokens: %s", e.what());
    return Error::LoadFailure;
  }

  // Set the vocab size to include special tokens
  vocab_size_ = token_map_->size() + special_token_map_->size();

  // Set up the normalizer (optional)
  try {
    TK_LOG(Info, "Setting up normalizer...");
    const auto& normalizer_json = parsed_json.at("normalizer");
    if (!normalizer_json.is_null()) {
      _normalizer = NormalizerConfig().parse_json(normalizer_json).create();
      TK_LOG(Info, "Normalizer set up");
    } else {
      TK_LOG(Info, "Normalizer field is null, skipping");
    }
  } catch (const std::exception& e) {
    // No "Normalizer" field found
    TK_LOG(
        Info,
        "No 'Normalizer' field found in json, out of range error: %s",
        e.what());
  }

  // Set up the pre-tokenizer
  try {
    TK_LOG(Info, "Setting up pretokenizer...");
    const auto& pretokenizer_json = parsed_json.at("pre_tokenizer");
    if (!pretokenizer_json.is_null()) {
      _pretokenizer =
          PreTokenizerConfig().parse_json(pretokenizer_json).create();
      TK_LOG(Info, "Pretokenizer set up");
    } else {
      TK_LOG(Info, "Pretokenizer field is null, skipping");
    }
  } catch (const std::exception& e) {
    TK_LOG(Info, "Could not parse pre_tokenizer: %s", e.what());
    return Error::LoadFailure;
  }

  // Set up the decoder (optional)
  try {
    _decoder =
        TokenDecoderConfig().parse_json(parsed_json.at("decoder")).create();
  } catch (const std::exception&) {
    // No decoder specified
  }

  // Parse the BPE merges
  try {
    TK_LOG(Info, "Loading BPE merges...");
    const auto& merges = parsed_json.at("/model/merges"_json_pointer);
    std::vector<std::pair<std::string, std::string>> merge_pairs;

    for (const auto& merge : merges) {
      std::string first, second;

      if (merge.is_string()) {
        // Legacy format: "token1 token2" (space-separated string)
        // This is the standard HuggingFace tokenizer.json format
        std::string merge_str = merge.get<std::string>();

        // Skip #version header lines (like HuggingFace does)
        if (merge_str.rfind("#version", 0) == 0) {
          continue;
        }

        auto space_pos = merge_str.find(' ');
        if (space_pos != std::string::npos) {
          first = merge_str.substr(0, space_pos);
          second = merge_str.substr(space_pos + 1);
        }
      } else if (merge.is_array() && merge.size() == 2) {
        // Tuple format: ["token1", "token2"] (array of two strings)
        // This format supports tokens containing spaces
        first = merge[0].get<std::string>();
        second = merge[1].get<std::string>();
      }

      if (!first.empty() && !second.empty()) {
        merge_pairs.emplace_back(first, second);
      }
    }

    // Build merge map: (token_id_1, token_id_2) -> (rank, merged_token_id)
    merge_map_ = std::make_unique<detail::MergeMap>();
    for (size_t i = 0; i < merge_pairs.size(); ++i) {
      const auto& [first, second] = merge_pairs[i];

      // Get token IDs for the merge pair
      auto first_id = token_map_->tryGetInteger(first);
      auto second_id = token_map_->tryGetInteger(second);

      if (first_id && second_id) {
        // Create merged token string
        std::string merged = first + second;
        auto merged_id = token_map_->tryGetInteger(merged);

        if (merged_id) {
          // Store merge rule: (first_id, second_id) -> (rank, merged_id)
          merge_map_->emplace(
              std::make_pair(*first_id, *second_id),
              std::make_pair(static_cast<uint32_t>(i), *merged_id));
        }
      }
    }

    TK_LOG(
        Info,
        "Loaded %" PRId64 " BPE merge rules",
        static_cast<int64_t>(merge_map_->size()));

    // Pre-compute merge ranks for efficient BPE encoding
    auto merge_ranks_result =
        detail::build_merge_ranks_map(*merge_map_, *token_map_);
    if (!merge_ranks_result.ok()) {
      return merge_ranks_result.error();
    }
    auto merge_ranks = std::move(*merge_ranks_result);
    TK_LOG(
        Info,
        "Built merge ranks map with %" PRId64 " entries",
        static_cast<int64_t>(merge_ranks.size()));
    merge_ranks_.emplace(std::move(merge_ranks));
  } catch (const std::exception& e) {
    TK_LOG(Error, "Could not parse merges: %s", e.what());
    return Error::LoadFailure;
  }

  // First, try to extract BOS/EOS tokens explicitly from config files
  // (special_tokens_map.json or tokenizer_config.json)
  std::string config_bos_token;
  std::string config_eos_token;
  std::string config_unk_token;

  // Parse byte_fallback property from model config
  try {
    const auto& model_json = parsed_json.at("model");
    if (model_json.contains("byte_fallback")) {
      byte_fallback_ = model_json.at("byte_fallback").get<bool>();
    }
    // Parse unk_token from model config
    if (model_json.contains("unk_token") &&
        !model_json.at("unk_token").is_null()) {
      config_unk_token = model_json.at("unk_token").get<std::string>();
    }
  } catch (const std::exception& e) {
    // byte_fallback or unk_token not found, just ignore.
  }
  if (!special_tokens_map_json.empty()) {
    std::ifstream special_file(special_tokens_map_json);
    if (special_file) {
      try {
        json special_tokens_json = json::parse(
            std::string(
                (std::istreambuf_iterator<char>(special_file)),
                std::istreambuf_iterator<char>()));

        if (special_tokens_json.contains("bos_token")) {
          config_bos_token =
              extract_token_string(special_tokens_json["bos_token"]);
        }
        if (special_tokens_json.contains("eos_token")) {
          config_eos_token =
              extract_token_string(special_tokens_json["eos_token"]);
        }

        TK_LOG(
            Info,
            "Loaded tokens from special_tokens_map.json: bos='%s', eos='%s'",
            config_bos_token.c_str(),
            config_eos_token.c_str());
      } catch (const std::exception& e) {
        TK_LOG(Info, "Could not parse special_tokens_map.json: %s", e.what());
      }
    }
  }
  // Fallback to tokenizer_config.json if BOS/EOS not found in
  // special_tokens_map.json
  if ((config_bos_token.empty() || config_eos_token.empty()) &&
      !model_config_json.empty()) {
    // Load it and parse it as json
    std::ifstream config_file(model_config_json);
    if (!config_file) {
      TK_LOG(
          Error, "failed to open config file: %s", model_config_json.c_str());
      return Error::LoadFailure;
    }
    std::string config_contents(
        (std::istreambuf_iterator<char>(config_file)),
        std::istreambuf_iterator<char>());
    try {
      json parsed_config_json = json::parse(config_contents);
      if (config_bos_token.empty() &&
          parsed_config_json.contains("bos_token")) {
        config_bos_token =
            extract_token_string(parsed_config_json["bos_token"]);
      }
      if (config_eos_token.empty() &&
          parsed_config_json.contains("eos_token")) {
        config_eos_token =
            extract_token_string(parsed_config_json["eos_token"]);
      }
      TK_LOG(
          Info,
          "Loaded tokens from tokenizer_config.json: bos='%s', eos='%s'",
          config_bos_token.c_str(),
          config_eos_token.c_str());
    } catch (const std::exception& e) {
      TK_LOG(Error, "Error parsing model config json file: %s", e.what());
      return Error::LoadFailure;
    }
  }

  // Set BOS/EOS IDs if explicitly configured and found in special_token_map
  if (!config_bos_token.empty()) {
    auto bos_candidate = special_token_map_->tryGetInteger(config_bos_token);
    if (bos_candidate) {
      bos_tok_ = *bos_candidate;
      bos_found = true;
    } else {
      TK_LOG(
          Info,
          "BOS token '%s' from config not found in special tokens map",
          config_bos_token.c_str());
    }
  }
  if (!config_eos_token.empty()) {
    auto eos_candidate = special_token_map_->tryGetInteger(config_eos_token);
    if (eos_candidate) {
      eos_tok_ = *eos_candidate;
      eos_found = true;
    } else {
      TK_LOG(
          Info,
          "EOS token '%s' from config not found in special tokens map",
          config_eos_token.c_str());
    }
  }

  // Set UNK ID if explicitly configured and found in special_token_map
  if (!config_unk_token.empty()) {
    auto unk_candidate = special_token_map_->tryGetInteger(config_unk_token);
    if (unk_candidate) {
      unk_tok_ = *unk_candidate;
      TK_LOG(
          Info,
          "UNK token '%s' from config set to ID: %" PRIu64,
          config_unk_token.c_str(),
          unk_tok_);
    } else {
      TK_LOG(
          Info,
          "UNK token '%s' from config not found in special tokens map",
          config_unk_token.c_str());
    }
  }

  // Make educated guess for BOS/EOS tokens only if they were not explicitly
  // found
  if (!bos_found || !eos_found) {
    std::vector<std::string_view> bos_candidates;
    std::vector<std::string_view> eos_candidates;
    for (std::size_t token_idx = 0; token_idx < special_token_map_->size();
         ++token_idx) {
      const auto [token, _] = special_token_map_->getElement(token_idx);
      if (!bos_found &&
          (token.find("bos") != std::string::npos ||
           token.find("begin") != std::string::npos)) {
        bos_candidates.push_back(token);
      }
      if (!eos_found &&
          (token.find("eos") != std::string::npos ||
           token.find("end") != std::string::npos)) {
        eos_candidates.push_back(token);
      }
    }

    if (!bos_found && bos_candidates.size() == 1) {
      bos_tok_ = *(special_token_map_->tryGetInteger(bos_candidates[0]));
      bos_found = true;
    }
    if (!eos_found && eos_candidates.size() == 1) {
      eos_tok_ = *(special_token_map_->tryGetInteger(eos_candidates[0]));
      eos_found = true;
    }

    // Make them the same if only one found
    // If only one is found after guessing, assume the other is the same
    if (bos_found && !eos_found) {
      eos_tok_ = bos_tok_;
      eos_found = true; // Mark as found to prevent further guessing
    } else if (!bos_found && eos_found) {
      bos_tok_ = eos_tok_;
      bos_found = true; // Mark as found to prevent further guessing
    }
  }

  // Mark initialized once everything is done
  initialized_ = true;

  return Error::Ok;
}
// -------------------------public method end-----------------------------------
// -------------------------private method start--------------------------------

Error HFTokenizer::_encode(
    const std::string& input,
    std::vector<uint64_t>& ret,
    uint64_t& last_piece_token_len) const {
  // Apply normalization first if normalizer is available
  std::string normalized_input = input;
  if (_normalizer) {
    normalized_input = _normalizer->normalize(input);
    TK_LOG(
        Info,
        "normalized input: '%s' -> '%s'",
        input.c_str(),
        normalized_input.c_str());
  }

  std::vector<std::string> pre_tokenized_pieces;
  if (_pretokenizer) {
    pre_tokenized_pieces = _pretokenizer->pre_tokenize(normalized_input);
  } else {
    // If no pretokenizer, treat the whole normalized input as one piece
    pre_tokenized_pieces.push_back(normalized_input);
  }

  for (const auto& piece : pre_tokenized_pieces) {
    // Check if the entire word is already a token to skip merging.
    const auto result = token_map_->tryGetInteger(piece);
    if (result) {
      last_piece_token_len = 1;
      ret.push_back(*result);
      continue;
    }
    auto tokens_result = byte_pair_encode_(piece, *token_map_);
    if (!tokens_result.ok()) {
      return tokens_result.error();
    }
    auto tokens = std::move(*tokens_result);

    last_piece_token_len = tokens.size();
    ret.insert(ret.end(), tokens.begin(), tokens.end());
  }
  return Error::Ok;
}

std::vector<std::string> HFTokenizer::_decode(
    const std::vector<std::string>& pieces) const {
  if (_decoder) {
    return _decoder->decode(pieces);
  }
  return pieces;
}

Result<std::vector<uint64_t>> HFTokenizer::byte_pair_encode_(
    const std::string& piece,
    const detail::TokenMap& token_map) const {
  if (piece.size() == 1) {
    const auto result = token_map.tryGetInteger(piece);
    if (result) {
      return std::vector<uint64_t>(1, *result);
    }
    // If the single piece itself is unknown, apply fallback logic
    if (byte_fallback_) {
      std::vector<uint64_t> byte_tokens;
      char c = piece[0]; // As piece.size() is 1 here
      char hex[7];
      snprintf(hex, sizeof(hex), "<0x%02X>", static_cast<unsigned char>(c));
      std::string byte_token_str(hex);
      const auto byte_result = token_map.tryGetInteger(byte_token_str);
      if (byte_result) {
        byte_tokens.push_back(*byte_result);
      } else {
        // Byte representation not found, fall back to UNK if configured
        if (unk_tok_ != 0) {
          TK_LOG(
              Info,
              "Byte token '%s' not found for piece '%s', falling back to UNK token %" PRIu64,
              byte_token_str.c_str(),
              piece.c_str(),
              unk_tok_);
          byte_tokens.push_back(unk_tok_);
        } else {
          TK_LOG(
              Error,
              "Unknown byte token '%s' for piece '%s' and no UNK token configured",
              byte_token_str.c_str(),
              piece.c_str());
          return Error::EncodeFailure;
        }
      }
      return byte_tokens;
    } else { // byte_fallback_ is false
      if (unk_tok_ != 0) {
        TK_LOG(
            Info,
            "Unknown token: '%s', falling back to UNK token %" PRIu64,
            piece.c_str(),
            unk_tok_);
        return std::vector<uint64_t>(unk_tok_);
      } else {
        TK_LOG(
            Error,
            "Unknown token: '%s' and no UNK token configured",
            piece.c_str());
        return Error::EncodeFailure;
      }
    }
  }

  // Use the pre-computed merge ranks (computed once during loading)
  const detail::TokenMap& merge_ranks =
      merge_ranks_ ? *merge_ranks_ : token_map;

  // Use the overridden _byte_pair_merge function with the proper merge ranks
  return _byte_pair_merge(
      piece,
      merge_ranks,
      [this, &piece, &token_map](uint64_t start, uint64_t stop) {
        std::string key = piece.substr(start, stop - start);
        const auto result = token_map.tryGetInteger(key);
        if (result) {
          return *result;
        } else {
          TK_LOG(
              Error,
              "BPE merge produced unknown token: '%s', start: %" PRIu64
              ", stop: %" PRIu64,
              key.c_str(),
              start,
              stop);
          return uint64_t(0); // Return unknown token ID instead of padding
          // If key is not found, and byte_fallback is true, signal for byte
          // fallback. Otherwise, attempt to use unk_tok_ or return an error.
          if (byte_fallback_) {
            return UINT64_MAX; // Signal to perform byte fallback in
                               // _byte_pair_merge
          } else {
            if (unk_tok_ != 0) {
              return unk_tok_;
            } else {
              TK_LOG(
                  Error,
                  "BPE merge produced unknown token: '%s', start: %" PRIu64
                  ", stop: %" PRIu64 " and no UNK token configured",
                  key.c_str(),
                  start,
                  stop);
              return uint64_t(0); // Return 0 (error) if no UNK token
            }
          }
        }
      });
}

std::vector<uint64_t> HFTokenizer::_byte_pair_merge(
    const std::string& piece,
    const detail::TokenMap& ranks,
    std::function<uint64_t(uint64_t, uint64_t)> func) const {
  // HF-specific BPE implementation that uses the Rust-style approach
  // with pre-computed merge ranks

  // Start with individual characters (like Rust implementation)
  HFWord word;

  // Process each UTF-8 character individually
  size_t i = 0;
  while (i < piece.size()) {
    size_t char_start = i;
    size_t char_len = 1;

    // Determine UTF-8 character length
    unsigned char byte = static_cast<unsigned char>(piece[i]);
    if ((byte & 0x80) == 0) {
      // ASCII character (0xxxxxxx)
      char_len = 1;
    } else if ((byte & 0xE0) == 0xC0) {
      // 2-byte UTF-8 character (110xxxxx)
      char_len = 2;
    } else if ((byte & 0xF0) == 0xE0) {
      // 3-byte UTF-8 character (1110xxxx)
      char_len = 3;
    } else if ((byte & 0xF8) == 0xF0) {
      // 4-byte UTF-8 character (11110xxx)
      char_len = 4;
    } else {
      // Invalid UTF-8 start byte, treat as single byte
      char_len = 1;
    }

    // Make sure we don't go beyond the string boundary
    if (char_start + char_len > piece.size()) {
      char_len = piece.size() - char_start;
    }

    uint64_t token_id = func(char_start, char_start + char_len);
    if (token_id == UINT64_MAX) { // func signals byte fallback
      std::string char_str = piece.substr(char_start, char_len);
      for (char c : char_str) { // Process each byte of the current UTF-8 char
        char hex[7];
        snprintf(hex, sizeof(hex), "<0x%02X>", static_cast<unsigned char>(c));
        std::string byte_token_str(hex);
        const auto byte_result = token_map_->tryGetInteger(byte_token_str);
        if (byte_result) {
          word.add(*byte_result, 1);
        } else {
          // If byte representation itself is unknown, fall back to UNK if
          // configured
          if (unk_tok_ != 0) {
            word.add(unk_tok_, 1);
          } else {
            TK_LOG(
                Error,
                "Unknown byte token '%s' for '%s' during fallback at position %zu and no UNK token configured",
                byte_token_str.c_str(),
                char_str.c_str(),
                char_start);
            return {}; // Signify failure
          }
        }
      }
    } else if (token_id != 0) { // Valid token found by func (0 is
                                // error/padding)
      word.add(token_id, char_len);
    } else { // func returned 0 (error or UNK was 0 and not found)
      // This path is taken if the substring was not found, byte_fallback_ is
      // false and unk_tok_ was also 0 (meaning func returned 0 as error).
      TK_LOG(
          Error,
          "Unknown character in HF BPE at position %zu and no UNK token configured (func returned 0)",
          char_start);
      return {}; // Signify failure
    }

    i += char_len;
  }

  // Apply BPE merges using the pre-computed merge ranks and token map
  if (merge_ranks_ && token_map_) {
    word.merge_all(*merge_ranks_, *token_map_);
  }

  return word.tokens;
}

} // namespace tokenizers
