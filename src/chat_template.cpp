/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <minja/minja.hpp>
#include <pytorch/tokenizers/chat_template.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace tokenizers {
namespace {
using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;
constexpr std::uintmax_t kMaxArtifactBytes = 1024 * 1024;

std::string read(const fs::path& path) {
  if (fs::file_size(path) > kMaxArtifactBytes) {
    throw std::runtime_error("Chat template artifact exceeds 1 MiB");
  }
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Cannot open chat template artifact");
  }
  std::string result(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  if (stream.bad()) {
    throw std::runtime_error("Cannot read chat template artifact");
  }
  return result;
}

std::string legacy_template(const Json& value, const std::string& name) {
  if (value.is_string()) {
    return name == "default" ? value.get<std::string>() : "";
  }
  if (value.is_object() && value.contains(name) && value[name].is_string()) {
    return value[name].get<std::string>();
  }
  if (value.is_array()) {
    for (const auto& entry : value) {
      if (entry.is_object() && entry.value("name", "") == name &&
          entry.contains("template") && entry["template"].is_string()) {
        return entry["template"].get<std::string>();
      }
    }
  }
  return "";
}

void special_tokens(const Json& config, Json& context) {
  for (auto it = config.begin(); it != config.end(); ++it) {
    const auto& key = it.key();
    if (key.size() < 6 || key.compare(key.size() - 6, 6, "_token") != 0) {
      continue;
    }
    if (it.value().is_string()) {
      context[key] = it.value();
    } else if (
        it.value().is_object() && it.value().contains("content") &&
        it.value()["content"].is_string()) {
      context[key] = it.value()["content"];
    }
  }
}
} // namespace

struct HFChatTemplate::Impl {
  std::shared_ptr<minja::TemplateNode> node;
  Json tokens = Json::object();
};

HFChatTemplate::HFChatTemplate() = default;
HFChatTemplate::~HFChatTemplate() = default;
bool HFChatTemplate::is_loaded() const {
  return impl_ != nullptr;
}

Error HFChatTemplate::load(const std::string& path, const std::string& name) {
  // Named templates are file names under additional_chat_templates.
  if (name.empty() ||
      !std::all_of(name.begin(), name.end(), [](unsigned char ch) {
        return std::isalnum(ch) || ch == '_' || ch == '-';
      })) {
    return Error::ParseFailure;
  }
  std::error_code error;
  const fs::path artifact(path);
  if (!fs::exists(artifact, error) || error) {
    return Error::LoadFailure;
  }
  try {
    const bool directory = fs::is_directory(artifact);
    const fs::path root = directory ? artifact : artifact.parent_path();
    auto next = std::make_unique<Impl>();
    std::string source;
    const auto config_path = root / "tokenizer_config.json";
    if (fs::exists(config_path)) {
      const auto config = Json::parse(read(config_path));
      if (!config.is_object()) {
        return Error::ParseFailure;
      }
      special_tokens(config, next->tokens);
      if (config.contains("chat_template")) {
        source = legacy_template(config["chat_template"], name);
      }
    }
    const auto tokens_path = root / "special_tokens_map.json";
    if (fs::exists(tokens_path)) {
      const auto tokens = Json::parse(read(tokens_path));
      if (!tokens.is_object()) {
        return Error::ParseFailure;
      }
      special_tokens(tokens, next->tokens);
    }
    const auto processor_path = root / "chat_template.json";
    if (source.empty() && fs::exists(processor_path)) {
      const auto processor = Json::parse(read(processor_path));
      if (processor.contains("chat_template")) {
        source = legacy_template(processor["chat_template"], name);
      }
    }
    fs::path standalone = name == "default"
        ? root / "chat_template.jinja"
        : root / "additional_chat_templates" / (name + ".jinja");
    if (!directory && artifact.extension() == ".jinja") {
      standalone = artifact;
    }
    if (fs::exists(standalone)) {
      source = read(standalone);
    }
    if (source.empty()) {
      return Error::LoadFailure;
    }
    // Jinja normalizes template source newlines, including inside literals.
    // Preserve message content: only the template source is normalized.
    std::string normalized;
    normalized.reserve(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
      if (source[i] == '\r') {
        normalized += '\n';
        if (i + 1 < source.size() && source[i + 1] == '\n')
          ++i;
      } else {
        normalized += source[i];
      }
    }
    minja::Options options{};
    options.trim_blocks = true;
    options.lstrip_blocks = true;
    next->node = minja::Parser::parse(normalized, options);
    impl_ = std::move(next);
    return Error::Ok;
  } catch (const std::exception&) {
    return Error::ParseFailure;
  }
}

Result<std::string> HFChatTemplate::apply(
    const Json& messages,
    bool add_generation_prompt,
    const Json& kwargs) const {
  if (!impl_) {
    return Error::Uninitialized;
  }
  if (!messages.is_array() || messages.empty() || !kwargs.is_object() ||
      kwargs.contains("messages") || kwargs.contains("add_generation_prompt")) {
    return Error::ParseFailure;
  }
  for (const auto& message : messages) {
    if (!message.is_object() || !message.contains("role") ||
        !message["role"].is_string() || !message.contains("content")) {
      return Error::ParseFailure;
    }
  }
  try {
    Json context = impl_->tokens;
    context.update(kwargs);
    context["messages"] = messages;
    context["add_generation_prompt"] = add_generation_prompt;
    // Per-call context avoids leaking mutable Jinja namespace values across
    // turns or between concurrent callers.
    auto environment = minja::Context::make(minja::Value(context));
    environment->set(
        "capitalize",
        minja::simple_function(
            "capitalize",
            {"text"},
            [](const std::shared_ptr<minja::Context>&,
               minja::Value& args) -> minja::Value {
              auto text = args.at("text").get<std::string>();
              // HF role names are ASCII. Do not silently approximate Python's
              // Unicode title/lower mappings (which can expand code points).
              for (size_t i = 0; i < text.size(); ++i) {
                const auto ch = static_cast<unsigned char>(text[i]);
                if (ch >= 128) {
                  throw std::runtime_error("Unicode capitalize is unsupported");
                }
                if (i == 0 && ch >= 'a' && ch <= 'z')
                  text[i] -= 'a' - 'A';
                if (i != 0 && ch >= 'A' && ch <= 'Z')
                  text[i] += 'a' - 'A';
              }
              return minja::Value(text);
            }));
    environment->set(
        "strftime_now",
        minja::simple_function(
            "strftime_now",
            {"format"},
            [](const std::shared_ptr<minja::Context>&,
               minja::Value& args) -> minja::Value {
              const auto format = args.at("format").get<std::string>();
              if (format.empty())
                return minja::Value("");
              const auto now = std::time(nullptr);
              std::tm local{};
#ifdef _WIN32
              if (localtime_s(&local, &now) != 0)
                throw std::runtime_error("Cannot read local time");
#else
              if (localtime_r(&now, &local) == nullptr)
                throw std::runtime_error("Cannot read local time");
#endif
              std::vector<char> buffer(128);
              while (buffer.size() <= 4096) {
                const auto count = std::strftime(
                    buffer.data(), buffer.size(), format.c_str(), &local);
                if (count)
                  return minja::Value(std::string(buffer.data(), count));
                buffer.resize(buffer.size() * 2);
              }
              throw std::runtime_error("Date format exceeds output limit");
            }));
    return impl_->node->render(environment);
  } catch (const std::exception&) {
    return Error::ParseFailure;
  }
}
} // namespace tokenizers
