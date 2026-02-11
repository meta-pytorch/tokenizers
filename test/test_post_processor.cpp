/*
 * Copyright (c) Software Mansion S.A. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/post_processor.h>
#include <vector>

using namespace tokenizers;
using json = nlohmann::json;

TEST(PostProcessorTest, TemplateProcessingSingleSequence) {
  // Setup: [CLS] $0 [SEP]
  Template single_template = {
      Piece::SpecialToken("[CLS]", 0),
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("[SEP]", 0)};
  Template pair_template; // Empty

  std::map<std::string, SpecialToken> special_tokens;
  special_tokens["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  special_tokens["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};

  auto processor = std::make_shared<TemplateProcessing>(
      single_template, pair_template, special_tokens);

  std::vector<uint64_t> input = {1, 2, 3};
  std::vector<uint64_t> expected = {101, 1, 2, 3, 102};

  auto output = processor->process(input);

  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, TemplateProcessingPairSequence) {
  // Setup: [CLS] $A [SEP] $B [SEP]
  Template single_template;
  Template pair_template = {
      Piece::SpecialToken("[CLS]", 0),
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("[SEP]", 0),
      Piece::Sequence(SequenceId::B, 1),
      Piece::SpecialToken("[SEP]", 0)};

  std::map<std::string, SpecialToken> special_tokens;
  special_tokens["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  special_tokens["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};

  auto processor = std::make_shared<TemplateProcessing>(
      single_template, pair_template, special_tokens);

  std::vector<uint64_t> input_a = {1, 2};
  std::vector<uint64_t> input_b = {3, 4};
  std::vector<uint64_t> expected = {101, 1, 2, 102, 3, 4, 102};

  auto output = processor->process(input_a, input_b);

  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, SequenceProcessing) {
  // Processor 1: Prepend 101
  Template t1 = {
      Piece::SpecialToken("[CLS]", 0), Piece::Sequence(SequenceId::A, 0)};
  std::map<std::string, SpecialToken> st1;
  st1["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  auto p1 = std::make_shared<TemplateProcessing>(t1, Template{}, st1);

  // Processor 2: Append 102
  Template t2 = {
      Piece::Sequence(SequenceId::A, 0), Piece::SpecialToken("[SEP]", 0)};
  std::map<std::string, SpecialToken> st2;
  st2["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};
  auto p2 = std::make_shared<TemplateProcessing>(t2, Template{}, st2);

  auto seq_processor =
      std::make_shared<Sequence>(std::vector<PostProcessor::Ptr>{p1, p2});

  std::vector<uint64_t> input = {5, 6};
  // p1 -> {101, 5, 6}
  // p2 -> {101, 5, 6, 102}
  std::vector<uint64_t> expected = {101, 5, 6, 102};

  auto output = seq_processor->process(input);

  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, ConfigParsing) {
  // Mimic parsing a simplified config
  json config_json = {
      {"type", "TemplateProcessing"},
      {"single", {"[CLS]", "$0", "[SEP]"}},
      {"pair", {"[CLS]", "$A:0", "[SEP]", "$B:1", "[SEP]"}},
      {"special_tokens",
       {{"[CLS]", {{"ids", {101}}}}, {"[SEP]", {{"ids", {102}}}}}}};

  auto processor = PostProcessorConfig().parse_json(config_json).create();
  ASSERT_NE(processor, nullptr);

  std::vector<uint64_t> input = {10, 20};
  std::vector<uint64_t> expected = {101, 10, 20, 102};

  auto output = processor->process(input);
  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, EmptyInput) {
  Template single_template = {
      Piece::SpecialToken("[CLS]", 0),
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("[SEP]", 0)};
  std::map<std::string, SpecialToken> special_tokens;
  special_tokens["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  special_tokens["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};

  auto processor = std::make_shared<TemplateProcessing>(
      single_template, Template{}, special_tokens);

  std::vector<uint64_t> input = {};
  std::vector<uint64_t> expected = {101, 102};

  auto output = processor->process(input);
  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, MultipleSequencesInTemplate) {
  // Setup: $0 [SEP] $0
  Template single_template = {
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("[SEP]", 0),
      Piece::Sequence(SequenceId::A, 0)};
  std::map<std::string, SpecialToken> special_tokens;
  special_tokens["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};

  auto processor = std::make_shared<TemplateProcessing>(
      single_template, Template{}, special_tokens);

  std::vector<uint64_t> input = {1};
  std::vector<uint64_t> expected = {1, 102, 1};

  auto output = processor->process(input);
  EXPECT_EQ(output, expected);
}

TEST(PostProcessorTest, NestedSequenceProcessing) {
  // p1: [101, $0]
  Template t1 = {
      Piece::SpecialToken("[CLS]", 0), Piece::Sequence(SequenceId::A, 0)};
  std::map<std::string, SpecialToken> st1;
  st1["[CLS]"] = {"[CLS]", {101}, {"[CLS]"}};
  auto p1 = std::make_shared<TemplateProcessing>(t1, Template{}, st1);

  // p2: [$0, 102]
  Template t2 = {
      Piece::Sequence(SequenceId::A, 0), Piece::SpecialToken("[SEP]", 0)};
  std::map<std::string, SpecialToken> st2;
  st2["[SEP]"] = {"[SEP]", {102}, {"[SEP]"}};
  auto p2 = std::make_shared<TemplateProcessing>(t2, Template{}, st2);

  auto inner_seq =
      std::make_shared<Sequence>(std::vector<PostProcessor::Ptr>{p1, p2});

  // p3: [200, $0, 201]
  Template t3 = {
      Piece::SpecialToken("200", 0),
      Piece::Sequence(SequenceId::A, 0),
      Piece::SpecialToken("201", 0)};
  std::map<std::string, SpecialToken> st3;
  st3["200"] = {"200", {200}, {"200"}};
  st3["201"] = {"201", {201}, {"201"}};
  auto p3 = std::make_shared<TemplateProcessing>(t3, Template{}, st3);

  auto outer_seq = std::make_shared<Sequence>(
      std::vector<PostProcessor::Ptr>{inner_seq, p3});

  std::vector<uint64_t> input = {5};
  // inner_seq(input) -> {101, 5, 102}
  // p3({101, 5, 102}) -> {200, 101, 5, 102, 201}
  std::vector<uint64_t> expected = {200, 101, 5, 102, 201};

  auto output = outer_seq->process(input);
  EXPECT_EQ(output, expected);
}
