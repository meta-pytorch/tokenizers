// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifdef TOKENIZERS_FB_BUCK
#include <TestResourceUtils/TestResourceUtils.h>
#endif
#include <gtest/gtest.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>

using namespace ::testing;

namespace tokenizers {

namespace {
// Test case based on llama2.c tokenizer
static inline std::string _get_resource_path(const std::string& name) {
#ifdef TOKENIZERS_FB_BUCK
  return facebook::xplat::testing::getPathForTestResource(
      "test/resources/" + name);
#else
  return std::getenv("RESOURCES_PATH") + std::string("/") + name;
#endif
}

} // namespace

class Llama2cTokenizerTest : public Test {
 public:
  void SetUp() override {
    tokenizer_ = std::make_unique<Llama2cTokenizer>();
    // Minimal llama2.c tokenizer fixture:
    // vocab_size=4 (0:"<unk>", 1:"<s>", 2:"</s>", 3:"<0x41>" -> 'A')
    // bos=1, eos=2.
    modelPath_ = _get_resource_path("test_llama2c_tokenizer.bin");
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  std::string modelPath_;
};

TEST_F(Llama2cTokenizerTest, EncodeWithoutLoadFails) {
  Result<std::vector<uint64_t>> res = tokenizer_->encode("hello world", 0, 0);
  EXPECT_EQ(res.error(), Error::Uninitialized);
}

TEST_F(Llama2cTokenizerTest, DecodeWithoutLoadFails) {
  auto result = tokenizer_->decode(0, 0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST_F(Llama2cTokenizerTest, IdToPieceWithoutLoadFails) {
  auto result = tokenizer_->id_to_piece(0);
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST_F(Llama2cTokenizerTest, DecodeOutOfRangeFails) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  auto result =
      tokenizer_->decode(0, static_cast<uint64_t>(tokenizer_->vocab_size()));
  EXPECT_EQ(result.error(), Error::OutOfRange);
}

TEST_F(Llama2cTokenizerTest, IdToPieceReturnsExpectedPieces) {
  Error res = tokenizer_->load(modelPath_);
  EXPECT_EQ(res, Error::Ok);

  auto unk = tokenizer_->id_to_piece(0);
  EXPECT_EQ(unk.error(), Error::Ok);
  EXPECT_EQ(unk.get(), "<unk>");

  auto bos = tokenizer_->id_to_piece(1);
  EXPECT_EQ(bos.error(), Error::Ok);
  EXPECT_EQ(bos.get(), "<s>");

  auto eos = tokenizer_->id_to_piece(2);
  EXPECT_EQ(eos.error(), Error::Ok);
  EXPECT_EQ(eos.get(), "</s>");

  auto byte_token = tokenizer_->id_to_piece(3);
  EXPECT_EQ(byte_token.error(), Error::Ok);
  EXPECT_EQ(byte_token.get(), "<0x41>");
}

TEST_F(Llama2cTokenizerTest, IdToPieceOutOfRangeFails) {
  Error res = tokenizer_->load(modelPath_);
  EXPECT_EQ(res, Error::Ok);

  auto result =
      tokenizer_->id_to_piece(static_cast<uint64_t>(tokenizer_->vocab_size()));
  EXPECT_EQ(result.error(), Error::OutOfRange);
}

TEST_F(Llama2cTokenizerTest, TokenizerMetadataIsExpected) {
  Error res = tokenizer_->load(modelPath_.c_str());
  EXPECT_EQ(res, Error::Ok);
  EXPECT_EQ(tokenizer_->vocab_size(), 4);
  EXPECT_EQ(tokenizer_->bos_tok(), 1);
  EXPECT_EQ(tokenizer_->eos_tok(), 2);
}

TEST_F(Llama2cTokenizerTest, SafeToDestruct) {
  // Safe to destruct initialized tokenizer.
  tokenizer_->load(modelPath_);
  tokenizer_.reset();

  // Safe to destruct uninitialized tokenizer.
  tokenizer_ = std::make_unique<Llama2cTokenizer>();
  tokenizer_.reset();
}

} // namespace tokenizers
