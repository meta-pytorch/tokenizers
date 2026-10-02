/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/result.h>
#include <memory>
#include <string>

namespace tokenizers {

/** Render HF chat templates without loading model weights or generating text.
 */
class HFChatTemplate {
 public:
  HFChatTemplate();
  ~HFChatTemplate();
  HFChatTemplate(const HFChatTemplate&) = delete;
  HFChatTemplate& operator=(const HFChatTemplate&) = delete;

  /**
   * Load a model directory, tokenizer_config.json, chat_template.json or Jinja.
   * Standalone Jinja takes precedence over the config's legacy template.
   * Select a named template explicitly; the default name is "default".
   * A failed reload preserves the previously loaded template.
   */
  Error load(const std::string& path, const std::string& name = "default");

  /**
   * Equivalent to rendering apply_chat_template(..., tokenize=False).
   * kwargs can include enable_thinking, tools and other model-specific values.
   * messages and add_generation_prompt are supplied by the dedicated arguments.
   * Encode the returned text with BOS/EOS counts zero to avoid duplicate
   * tokens.
   */
  Result<std::string> apply(
      const nlohmann::ordered_json& messages,
      bool add_generation_prompt = true,
      const nlohmann::ordered_json& kwargs =
          nlohmann::ordered_json::object()) const;

  bool is_loaded() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tokenizers
