# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#
# Build the encode latency benchmark. Standalone: pulls in the parent
# tokenizers library via FetchContent (mirrors test/CMakeLists.txt) so it can be
# configured on its own and always rebuilds the library under test.
#
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)

project(TokenizersBenchmark)

include(FetchContent)
FetchContent_Declare(
  tokenizers SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. BUILD_ALWAYS ON
)
set(SUPPORT_REGEX_LOOKAHEAD ON)
FetchContent_MakeAvailable(tokenizers)

add_executable(hf_tokenizer_encode_latency hf_tokenizer_encode_latency.cpp)
target_link_libraries(hf_tokenizer_encode_latency PUBLIC tokenizers)
