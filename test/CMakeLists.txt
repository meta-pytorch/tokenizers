# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#
# Build tokenizers tests.
#
cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)

project(TokenizersTests)

# Include ExternalProject module
include(FindPackageHandleStandardArgs)
include(FetchContent)
include(ExternalProject)
FetchContent_Declare(
  tokenizers SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. BUILD_ALWAYS ON
)
set(SUPPORT_REGEX_LOOKAHEAD ON)
FetchContent_MakeAvailable(tokenizers)

# Build test
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE
)
FetchContent_MakeAvailable(googletest)

file(GLOB test_source_files test_*.cpp)

set(test_env "RESOURCES_PATH=${CMAKE_CURRENT_SOURCE_DIR}/resources")
enable_testing()
foreach(test_source_file ${test_source_files})
  get_filename_component(test_name ${test_source_file} NAME_WE)
  message(STATUS "Configuring unit test ${test_name}")
  add_executable(${test_name} ${test_source_file})
  target_include_directories(
    ${test_name} PRIVATE GTEST_INCLUDE_PATH ${TOKENIZERS_INCLUDE_DIRS}
  )
  target_link_libraries(${test_name} PUBLIC gtest_main GTest::gmock tokenizers)
  add_test(${test_name} "${test_name}")
  set_tests_properties(${test_name} PROPERTIES ENVIRONMENT ${test_env})
endforeach()

# The lookahead fallback registers itself from a static initializer, so the
# linker keeps it only if something references it. Every test above hides that,
# because linking the tokenizers target pulls regex_lookahead in with a
# whole-archive flag. This one names the archive files instead, so nothing is
# force loaded, which is the shape the Apple prebuilt frameworks have and the
# only shape that fails when the reference goes away.
add_executable(regex_lookahead_link_test regex_lookahead_link_test.cpp)
# Naming the archive files skips the usage requirements a target would carry, so
# the headers have to be compiled the same way the archive was.
target_include_directories(
  regex_lookahead_link_test
  PRIVATE $<TARGET_PROPERTY:tokenizers,INTERFACE_INCLUDE_DIRECTORIES>
)
target_compile_definitions(
  regex_lookahead_link_test
  PRIVATE $<TARGET_PROPERTY:tokenizers,INTERFACE_COMPILE_DEFINITIONS>
)
target_link_libraries(
  regex_lookahead_link_test
  PRIVATE gtest_main $<TARGET_FILE:tokenizers> $<TARGET_FILE:regex_lookahead>
          pcre2-8-static re2::re2 sentencepiece-static
)
add_dependencies(regex_lookahead_link_test tokenizers regex_lookahead)
add_test(regex_lookahead_link_test regex_lookahead_link_test)
set_tests_properties(
  regex_lookahead_link_test PROPERTIES ENVIRONMENT ${test_env}
)
