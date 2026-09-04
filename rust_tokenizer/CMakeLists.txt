# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(_rust_profile release)
if(TOKENIZERS_OPTIMIZE_SIZE)
  set(_rust_profile minsize)
endif()

set(_cargo_target_dir ${CMAKE_CURRENT_BINARY_DIR}/cargo-target)
string(
  CONCAT
  _rust_library_name
  ${CMAKE_STATIC_LIBRARY_PREFIX}
  tokenizers_hf_ffi
  ${CMAKE_STATIC_LIBRARY_SUFFIX}
)
set(_rust_library ${_cargo_target_dir}/${_rust_profile}/${_rust_library_name})
file(GLOB_RECURSE _rust_sources CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs
)

add_custom_command(
  OUTPUT ${_rust_library}
  COMMAND
    ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${_cargo_target_dir}
    CARGO_ENCODED_RUSTFLAGS=-Crelocation-model=pic
    ${CARGO_EXECUTABLE} build --locked --manifest-path
    ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml --profile ${_rust_profile}
  DEPENDS ${_rust_sources} ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
          ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
)

add_custom_target(tokenizers_hf_ffi_build DEPENDS ${_rust_library})
add_library(tokenizers_hf_ffi STATIC IMPORTED GLOBAL)
set_target_properties(
  tokenizers_hf_ffi PROPERTIES IMPORTED_LOCATION ${_rust_library}
)
add_dependencies(tokenizers_hf_ffi tokenizers_hf_ffi_build)

add_library(tokenizers_hf_rust_tokenizer STATIC ../src/rust_hf_tokenizer.cpp)
add_library(tokenizers::hf_rust_tokenizer ALIAS tokenizers_hf_rust_tokenizer)
set_target_properties(
  tokenizers_hf_rust_tokenizer PROPERTIES EXPORT_NAME hf_rust_tokenizer
)
target_include_directories(
  tokenizers_hf_rust_tokenizer
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(tokenizers_hf_rust_tokenizer PUBLIC cxx_std_17)
target_link_libraries(
  tokenizers_hf_rust_tokenizer
  PRIVATE tokenizers_hf_ffi
)

if(APPLE)
  target_link_libraries(tokenizers_hf_rust_tokenizer PRIVATE iconv)
elseif(UNIX)
  find_package(Threads REQUIRED)
  target_link_libraries(
    tokenizers_hf_rust_tokenizer
    PRIVATE Threads::Threads ${CMAKE_DL_LIBS} m rt util
  )
endif()

install(
  TARGETS tokenizers_hf_rust_tokenizer
  EXPORT tokenizers-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${_rust_library} DESTINATION ${CMAKE_INSTALL_LIBDIR})
