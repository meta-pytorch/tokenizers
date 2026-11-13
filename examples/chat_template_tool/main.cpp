/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <pytorch/tokenizers/chat_template.h>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::cerr
        << "Usage: chat_template_tool MODEL_DIR MESSAGES_JSON [KWARGS_JSON] [0|1 add_generation_prompt]\n";
    return 1;
  }
  try {
    std::ifstream messages_file(argv[2]);
    const auto messages = nlohmann::ordered_json::parse(messages_file);
    auto kwargs = nlohmann::ordered_json::object();
    if (argc >= 4) {
      std::ifstream kwargs_file(argv[3]);
      kwargs = nlohmann::ordered_json::parse(kwargs_file);
    }
    if (argc == 5 && std::string(argv[4]) != "0" &&
        std::string(argv[4]) != "1") {
      throw std::runtime_error("Generation prompt must be 0 or 1");
    }
    tokenizers::HFChatTemplate renderer;
    if (renderer.load(argv[1]) != tokenizers::Error::Ok) {
      throw std::runtime_error("Cannot load HF chat template");
    }
    const auto prompt = renderer.apply(
        messages, argc < 5 || std::string(argv[4]) == "1", kwargs);
    if (!prompt.ok()) {
      throw std::runtime_error("Cannot render HF chat template");
    }
    std::cout << *prompt;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
