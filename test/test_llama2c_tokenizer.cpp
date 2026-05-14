// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifdef TOKENIZERS_FB_BUCK
#include <TestResourceUtils/TestResourceUtils.h>
#endif
#include <gtest/gtest.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

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

std::string make_temp_file() {
#ifdef _WIN32
  char tmp_dir[MAX_PATH];
  char tmp_file[MAX_PATH];
  GetTempPathA(MAX_PATH, tmp_dir);
  GetTempFileNameA(tmp_dir, "tok", 0, tmp_file);
  return std::string(tmp_file);
#else
  std::string tpl =
      std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
      "/tokenizer_test_XXXXXX";
  int fd = mkstemp(&tpl[0]);
  EXPECT_NE(fd, -1);
  close(fd);
  return tpl;
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

TEST_F(Llama2cTokenizerTest, PieceToIdWithoutLoadFails) {
  auto result = tokenizer_->piece_to_id("<s>");
  EXPECT_EQ(result.error(), Error::Uninitialized);
}

TEST_F(Llama2cTokenizerTest, PieceToIdReturnsExpectedIds) {
  Error res = tokenizer_->load(modelPath_);
  EXPECT_EQ(res, Error::Ok);

  auto unk = tokenizer_->piece_to_id("<unk>");
  EXPECT_EQ(unk.error(), Error::Ok);
  EXPECT_EQ(unk.get(), 0);

  auto bos = tokenizer_->piece_to_id("<s>");
  EXPECT_EQ(bos.error(), Error::Ok);
  EXPECT_EQ(bos.get(), 1);

  auto eos = tokenizer_->piece_to_id("</s>");
  EXPECT_EQ(eos.error(), Error::Ok);
  EXPECT_EQ(eos.get(), 2);

  auto byte_token = tokenizer_->piece_to_id("<0x41>");
  EXPECT_EQ(byte_token.error(), Error::Ok);
  EXPECT_EQ(byte_token.get(), 3);
}

TEST_F(Llama2cTokenizerTest, PieceToIdNotFoundFails) {
  Error res = tokenizer_->load(modelPath_);
  EXPECT_EQ(res, Error::Ok);

  auto result = tokenizer_->piece_to_id("not_a_real_piece");
  EXPECT_EQ(result.error(), Error::OutOfRange);
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

TEST_F(Llama2cTokenizerTest, TestDecodeSpecialTokens) {
  Error res = tokenizer_->load(modelPath_);
  EXPECT_EQ(res, Error::Ok);

  uint64_t bos = tokenizer_->bos_tok();

  // skip_special_tokens = false
  auto res_false = tokenizer_->decode(0, bos, false);
  EXPECT_TRUE(res_false.ok());
  EXPECT_EQ(res_false.get(), "<s>");

  // skip_special_tokens = true
  // Llama2cTokenizer ignores the skip_special_tokens flag and returns the
  // token.
  auto res_true = tokenizer_->decode(0, bos, true);
  EXPECT_TRUE(res_true.ok());
  EXPECT_EQ(res_true.get(), "<s>");
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

TEST_F(Llama2cTokenizerTest, DestructAfterFailedLoadIsSafe) {
  // Issue #1: Destructor must not crash after failed load
  Error res = tokenizer_->load("/nonexistent/path/tokenizer.bin");
  EXPECT_NE(res, Error::Ok);
  tokenizer_.reset(); // Must not crash
}

TEST_F(Llama2cTokenizerTest, DestructAfterCorruptFileIsSafe) {
  // Destructor must not crash after load() returns ParseFailure
  // with vocab_ partially populated.
  std::string tmp_path = make_temp_file();
  {
    FILE* f = fopen(tmp_path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    int32_t metadata[4] = {4, 1, 2, 10};
    fwrite(metadata, sizeof(int32_t), 4, f);
    float score = 1.0f;
    fwrite(&score, sizeof(float), 1, f);
    int32_t len = 5;
    fwrite(&len, sizeof(int32_t), 1, f);
    // Only write 2 of 5 bytes — fread of token content will fail
    fwrite("ab", 2, 1, f);
    fclose(f);
  }
  Error res = tokenizer_->load(tmp_path);
  EXPECT_NE(res, Error::Ok);
  tokenizer_.reset(); // Must not crash
  std::remove(tmp_path.c_str());
}

TEST_F(Llama2cTokenizerTest, LoadRejectsNegativeVocabSize) {
  // Issue #2: Reject invalid vocab_size from file
  std::string tmp_path = make_temp_file();
  {
    FILE* f = fopen(tmp_path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    int32_t metadata[4] = {-1, 1, 2, 10};
    fwrite(metadata, sizeof(int32_t), 4, f);
    fclose(f);
  }
  Error res = tokenizer_->load(tmp_path);
  EXPECT_EQ(res, Error::ParseFailure);
  tokenizer_.reset();
  std::remove(tmp_path.c_str());
}

TEST_F(Llama2cTokenizerTest, LoadRejectsHugeVocabSize) {
  // Issue #2: Reject impossibly large vocab_size
  std::string tmp_path = make_temp_file();
  {
    FILE* f = fopen(tmp_path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    int32_t metadata[4] = {2000000000, 1, 2, 10};
    fwrite(metadata, sizeof(int32_t), 4, f);
    fclose(f);
  }
  Error res = tokenizer_->load(tmp_path);
  EXPECT_EQ(res, Error::ParseFailure);
  tokenizer_.reset();
  std::remove(tmp_path.c_str());
}

TEST_F(Llama2cTokenizerTest, LoadRejectsNegativeMaxTokenLength) {
  // Issue #2: Reject negative max_token_length
  std::string tmp_path = make_temp_file();
  {
    FILE* f = fopen(tmp_path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    int32_t metadata[4] = {4, 1, 2, -5};
    fwrite(metadata, sizeof(int32_t), 4, f);
    fclose(f);
  }
  Error res = tokenizer_->load(tmp_path);
  EXPECT_EQ(res, Error::ParseFailure);
  tokenizer_.reset();
  std::remove(tmp_path.c_str());
}

TEST_F(Llama2cTokenizerTest, LoadRejectsInvalidTokenLength) {
  // Reject token length exceeding the hard safety limit (kMaxTokenLengthBytes)
  std::string tmp_path = make_temp_file();
  {
    FILE* f = fopen(tmp_path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    int32_t metadata[4] = {2, 0, 1, 10};
    fwrite(metadata, sizeof(int32_t), 4, f);
    float score = 1.0f;
    fwrite(&score, sizeof(float), 1, f);
    int32_t len = 1048577; // Exceeds kMaxTokenLengthBytes (1 MiB)
    fwrite(&len, sizeof(int32_t), 1, f);
    fclose(f);
  }
  Error res = tokenizer_->load(tmp_path);
  EXPECT_EQ(res, Error::ParseFailure);
  tokenizer_.reset();
  std::remove(tmp_path.c_str());
}

} // namespace tokenizers
