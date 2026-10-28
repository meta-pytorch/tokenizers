/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <vector>

#include <pytorch/tokenizers/pcre2_regex.h>

namespace tokenizers {

// Helper function to check if error code is a UTF-8 validation error
// PCRE2 UTF-8 error codes range from PCRE2_ERROR_UTF8_ERR1 (-3) to
// PCRE2_ERROR_UTF8_ERR21 (-23)
static bool is_utf8_error(int error_code) {
  return error_code >= PCRE2_ERROR_UTF8_ERR21 &&
      error_code <= PCRE2_ERROR_UTF8_ERR1;
}

// Whether offset is inside a multi-byte UTF-8 character, i.e. on one of its
// continuation bytes (10xxxxxx) rather than on a character boundary
static bool is_inside_utf8_char(
    PCRE2_SPTR subject,
    PCRE2_SIZE subject_length,
    PCRE2_SIZE offset) {
  return offset < subject_length && (subject[offset] & 0xC0) == 0x80;
}

Error Pcre2Regex::compile(const std::string& pattern) {
  int error_code;
  PCRE2_SIZE error_offset;

  // Compile the pattern with UTF-8 mode enabled first
  regex_ = pcre2_compile(
      reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
      pattern.length(),
      PCRE2_UCP | PCRE2_UTF, // Enable Unicode support and UTF-8 mode
      &error_code,
      &error_offset,
      nullptr);

  utf_ = regex_ != nullptr;

  if (regex_ == nullptr) {
    PCRE2_UCHAR error_buffer[256];
    pcre2_get_error_message(error_code, error_buffer, sizeof(error_buffer));

    // Check if this is a UTF-8 validation error
    if (is_utf8_error(error_code)) {
      TK_LOG(
          Info,
          "PCRE2 UTF-8 validation failed at offset %" PRId64
          ": %s. Retrying without UTF flags.",
          static_cast<int64_t>(error_offset),
          error_buffer);

      // Retry compilation without PCRE2_UTF flag
      regex_ = pcre2_compile(
          reinterpret_cast<PCRE2_SPTR>(pattern.c_str()),
          pattern.length(),
          PCRE2_UCP, // Enable Unicode properties but not UTF-8 validation
          &error_code,
          &error_offset,
          nullptr);

      if (regex_ == nullptr) {
        pcre2_get_error_message(error_code, error_buffer, sizeof(error_buffer));
        TK_LOG(
            Error,
            "PCRE2 compilation failed (without UTF) at offset %" PRId64 ": %s",
            static_cast<int64_t>(error_offset),
            error_buffer);
        return Error::RegexFailure;
      }
    } else {
      TK_LOG(
          Error,
          "PCRE2 compilation failed at offset %" PRId64 ": %s",
          static_cast<int64_t>(error_offset),
          error_buffer);
      return Error::RegexFailure;
    }
  }

  return Error::Ok;
}

Pcre2Regex::~Pcre2Regex() {
  if (regex_) {
    pcre2_code_free(regex_);
  }
}

std::vector<Match> Pcre2Regex::find_all(const std::string& text) const {
  std::vector<Match> result;

  if (!regex_) {
    TK_LOG(Error, "Regex is not compiled or invalid, run compile() first");
    return result;
  }

  // Allocate match_data per call rather than sharing it across calls.
  // PCRE2 writes match offsets into match_data during pcre2_match(), so a
  // shared buffer races under concurrent find_all() invocations: it silently
  // corrupts results and can trigger heap-buffer-overflow inside PCRE2.
  pcre2_match_data* match_data =
      pcre2_match_data_create_from_pattern(regex_, nullptr);
  if (match_data == nullptr) {
    TK_LOG(Error, "Failed to create PCRE2 match data");
    return result;
  }

  PCRE2_SIZE* ovector;
  PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(text.c_str());
  PCRE2_SIZE subject_length = text.length();
  PCRE2_SIZE offset = 0;
  uint32_t options = 0;

  while (offset < subject_length) {
    int rc = pcre2_match(
        regex_, subject, subject_length, offset, options, match_data, nullptr);

    if (rc < 0) {
      if (rc == PCRE2_ERROR_NOMATCH) {
        break; // No more matches
      } else {
        // Error occurred
        PCRE2_UCHAR error_buffer[256];
        pcre2_get_error_message(rc, error_buffer, sizeof(error_buffer));
        std::cerr << "PCRE2 matching error: " << error_buffer << std::endl;
        break;
      }
    }

    ovector = pcre2_get_ovector_pointer(match_data);

    // Add the match to the result
    result.push_back({ovector[0], ovector[1]});

    // Move to the next position after the match
    offset = ovector[1];

    // If the match was empty, move forward by one character to avoid infinite
    // loop. In UTF mode that character can be several bytes long, and a match
    // started inside it fails with PCRE2_ERROR_BADUTFOFFSET.
    if (ovector[0] == ovector[1]) {
      offset++;
      while (utf_ && is_inside_utf8_char(subject, subject_length, offset)) {
        offset++;
      }
    }

    // In UTF mode each call checks that the subject is valid UTF-8 from its
    // start offset to the end, which makes this loop quadratic in the length
    // of the text. The first call has checked all of it, so later calls can
    // skip the check while they start on a character boundary. (A match can
    // end inside a character only with \C; keep checking there so that case
    // still fails with PCRE2_ERROR_BADUTFOFFSET.)
    options = utf_ && is_inside_utf8_char(subject, subject_length, offset)
        ? 0
        : PCRE2_NO_UTF_CHECK;
  }

  pcre2_match_data_free(match_data);
  return result;
}

} // namespace tokenizers
