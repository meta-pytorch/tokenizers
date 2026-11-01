/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>
#include <pytorch/tokenizers/chat_template.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <thread>

namespace tokenizers {
namespace {
using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

struct Directory {
  fs::path path;
  Directory() {
    static std::atomic<unsigned> next{0};
    path = fs::temp_directory_path() /
        ("hf-chat-template-" + std::to_string(next++) + "-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(path);
  }
  ~Directory() {
    std::error_code error;
    fs::remove_all(path, error);
  }
  void write(const std::string& name, const std::string& content) {
    fs::create_directories((path / name).parent_path());
    std::ofstream(path / name, std::ios::binary) << content;
  }
};

const Json kMessages =
    Json::array({{{"role", "user"}, {"content", "Café 世界 👋"}}});

TEST(HFChatTemplateTest, UninitializedAndMissingArtifact) {
  HFChatTemplate renderer;
  EXPECT_EQ(renderer.apply(kMessages).error(), Error::Uninitialized);
  EXPECT_EQ(renderer.load("/missing/chat_template.jinja"), Error::LoadFailure);
  EXPECT_FALSE(renderer.is_loaded());
}

TEST(HFChatTemplateTest, StandaloneTakesPrecedenceAndUsesSpecialTokenContent) {
  Directory directory;
  directory.write(
      "tokenizer_config.json",
      R"({"chat_template":"wrong","bos_token":{"content":"<s>"},"eos_token":"</s>"})");
  directory.write(
      "chat_template.jinja",
      "{{ bos_token }}{{ messages[0]['content'] }}{{ eos_token }}{% if add_generation_prompt %}Assistant:{% endif %}");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  auto rendered = renderer.apply(kMessages);
  ASSERT_TRUE(rendered.ok());
  EXPECT_EQ(*rendered, "<s>Café 世界 👋</s>Assistant:");
  EXPECT_EQ(*renderer.apply(kMessages, false), "<s>Café 世界 👋</s>");
}

TEST(HFChatTemplateTest, LegacyNamedAndStandaloneNamedTemplates) {
  Directory directory;
  directory.write(
      "tokenizer_config.json",
      R"({"chat_template":[{"name":"default","template":"DEFAULT"},{"name":"tool_use","template":"LEGACY TOOL"}]})");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  EXPECT_EQ(*renderer.apply(kMessages), "DEFAULT");
  ASSERT_EQ(renderer.load(directory.path.string(), "tool_use"), Error::Ok);
  EXPECT_EQ(*renderer.apply(kMessages), "LEGACY TOOL");
  directory.write(
      "additional_chat_templates/tool_use.jinja", "{{ tools[0]['name'] }}");
  ASSERT_EQ(renderer.load(directory.path.string(), "tool_use"), Error::Ok);
  EXPECT_EQ(
      *renderer.apply(
          kMessages, true, {{"tools", Json::array({{{"name", "camera"}}})}}),
      "camera");
  EXPECT_EQ(
      renderer.load(directory.path.string(), "../escape"), Error::ParseFailure);
}

TEST(HFChatTemplateTest, LegacyStringAndJsonObjectTemplate) {
  Directory directory;
  directory.write(
      "tokenizer_config.json",
      R"({"chat_template":"{{ messages[0]['role'] }}"})");
  HFChatTemplate renderer;
  ASSERT_EQ(
      renderer.load((directory.path / "tokenizer_config.json").string()),
      Error::Ok);
  EXPECT_EQ(*renderer.apply(kMessages), "user");
  directory.write(
      "tokenizer_config.json", R"({"chat_template":{"default":"OBJECT"}})");
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  EXPECT_EQ(*renderer.apply(kMessages), "OBJECT");
}

TEST(HFChatTemplateTest, FailedReloadPreservesPreviousTemplate) {
  Directory directory;
  directory.write("chat_template.jinja", "VALID");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  directory.write("chat_template.jinja", "{% if true %}");
  EXPECT_EQ(renderer.load(directory.path.string()), Error::ParseFailure);
  EXPECT_TRUE(renderer.is_loaded());
  EXPECT_EQ(*renderer.apply(kMessages), "VALID");
  directory.write("tokenizer_config.json", "{");
  EXPECT_EQ(renderer.load(directory.path.string()), Error::ParseFailure);
}

TEST(HFChatTemplateTest, InvalidMessagesAndReservedKwargsFail) {
  Directory directory;
  directory.write("chat_template.jinja", "VALID");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  EXPECT_EQ(renderer.apply(Json::array()).error(), Error::ParseFailure);
  EXPECT_EQ(renderer.apply(Json::object()).error(), Error::ParseFailure);
  EXPECT_EQ(
      renderer.apply(Json::array({{{"content", "missing role"}}})).error(),
      Error::ParseFailure);
  EXPECT_EQ(
      renderer.apply(kMessages, true, {{"messages", "override"}}).error(),
      Error::ParseFailure);
  EXPECT_EQ(
      renderer.apply(kMessages, true, {{"add_generation_prompt", false}})
          .error(),
      Error::ParseFailure);
}

TEST(HFChatTemplateTest, RuntimeExceptionIsAnError) {
  Directory directory;
  directory.write(
      "chat_template.jinja", "{{ raise_exception('Invalid conversation') }}");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  EXPECT_EQ(renderer.apply(kMessages).error(), Error::ParseFailure);
}

TEST(HFChatTemplateTest, LegacyProcessorTemplateIsLoaded) {
  Directory directory;
  directory.write(
      "chat_template.json",
      R"({"chat_template":"{{ messages[0]['content'] }}"})");
  HFChatTemplate renderer;
  ASSERT_EQ(
      renderer.load((directory.path / "chat_template.json").string()),
      Error::Ok);
  EXPECT_EQ(*renderer.apply(kMessages), "Café 世界 👋");
}

TEST(HFChatTemplateTest, CurrentDateHelperIsAvailable) {
  Directory directory;
  directory.write("chat_template.jinja", "{{ strftime_now('%Y') }}");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  const auto now = std::time(nullptr);
  std::tm local{};
#ifdef _WIN32
  ASSERT_EQ(localtime_s(&local, &now), 0);
#else
  ASSERT_NE(localtime_r(&now, &local), nullptr);
#endif
  EXPECT_EQ(*renderer.apply(kMessages), std::to_string(1900 + local.tm_year));
}

TEST(HFChatTemplateTest, ThinkingAndMutableContextArePerCall) {
  Directory directory;
  directory.write(
      "chat_template.jinja",
      "{% set ns = namespace(value=0) %}{% set ns.value = ns.value + 1 %}{{ ns.value }}{% if enable_thinking is defined and not enable_thinking %}OFF{% else %}ON{% endif %}");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  EXPECT_EQ(*renderer.apply(kMessages), "1ON");
  EXPECT_EQ(
      *renderer.apply(kMessages, true, {{"enable_thinking", false}}), "1OFF");
  EXPECT_EQ(
      *renderer.apply(kMessages, true, {{"enable_thinking", true}}), "1ON");
}

TEST(
    HFChatTemplateTest,
    CapitalizeHandlesAsciiRolesAndRejectsUnicodeApproximation) {
  Directory directory;
  directory.write(
      "chat_template.jinja", "{{ messages[0]['role'] | capitalize }}");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  EXPECT_EQ(*renderer.apply(kMessages), "User");
  EXPECT_EQ(
      *renderer.apply(Json::array({{{"role", "aSSISTANT"}, {"content", ""}}})),
      "Assistant");
  EXPECT_EQ(
      renderer.apply(Json::array({{{"role", "ß"}, {"content", ""}}})).error(),
      Error::ParseFailure);
}

TEST(HFChatTemplateTest, ConcurrentRenderingKeepsMessagesSeparate) {
  Directory directory;
  directory.write("chat_template.jinja", "{{ messages[0]['content'] }}");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  std::atomic<bool> success{true};
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 3; ++i) {
    threads.emplace_back([&, i] {
      for (unsigned n = 0; n < 20; ++n) {
        const auto text = std::to_string(i) + ":" + std::to_string(n);
        const auto rendered = renderer.apply(
            Json::array({{{"role", "user"}, {"content", text}}}));
        if (!rendered.ok() || *rendered != text)
          success = false;
      }
    });
  }
  for (auto& thread : threads)
    thread.join();
  EXPECT_TRUE(success);
}

TEST(HFChatTemplateTest, TemplateNewlinesMatchJinjaWithoutChangingMessages) {
  Directory directory;
  directory.write(
      "chat_template.jinja",
      "{{ 'first\r\nsecond\rthird' }}{{ messages[0]['content'] }}");
  HFChatTemplate renderer;
  ASSERT_EQ(renderer.load(directory.path.string()), Error::Ok);
  EXPECT_EQ(
      *renderer.apply(Json::array({{{"role", "user"}, {"content", "\r\n"}}})),
      "first\nsecond\nthird\r\n");
}

TEST(HFChatTemplateTest, ExactTransformersParityForPinnedRealModels) {
  const auto root = fs::path(std::getenv("RESOURCES_PATH")) / "chat_templates";
  std::ifstream input(root / "cases.json");
  const auto cases = Json::parse(input);
  ASSERT_GE(cases.size(), 30);
  for (size_t i = 0; i < cases.size(); ++i) {
    SCOPED_TRACE("HF parity case " + std::to_string(i));
    const auto& entry = cases[i];
    HFChatTemplate renderer;
    ASSERT_EQ(
        renderer.load((root / entry["model"].get<std::string>()).string()),
        Error::Ok);
    const auto actual = renderer.apply(
        entry["messages"],
        entry["add_generation_prompt"].get<bool>(),
        entry["kwargs"]);
    ASSERT_TRUE(actual.ok());
    EXPECT_EQ(*actual, entry["expected"].get<std::string>());
  }
}
} // namespace
} // namespace tokenizers
