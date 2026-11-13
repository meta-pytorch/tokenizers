/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pytorch/tokenizers/re2_regex.h>

namespace tokenizers {

namespace {

// RE2's \s is the ASCII class [\t\n\f\r ], while the HuggingFace tokenizers
// library (Oniguruma) and the PCRE2 fallback (PCRE2_UCP) match every Unicode
// White_Space code point. These are the members of that set, spelled so RE2
// can use them both inside and outside a character class.
constexpr char kUnicodeSpaceMembers[] = R"(\t-\r\x{85}\p{Z})";

// Rewrite \s and \S so that a pattern like CLIP's `\s+` or
// `[^\s\p{L}\p{N}]+` treats U+00A0, U+3000 etc. as whitespace, as it does
// under the other engines.
std::string with_unicode_whitespace(const std::string& pattern) {
  std::string result;
  result.reserve(pattern.size());
  bool in_class = false;
  for (size_t i = 0; i < pattern.size(); ++i) {
    const char c = pattern[i];
    if (c == '\\' && i + 1 < pattern.size()) {
      const char next = pattern[++i];
      if (next == 's') {
        if (in_class) {
          result += kUnicodeSpaceMembers;
        } else {
          result += std::string("[") + kUnicodeSpaceMembers + "]";
        }
      } else if (next == 'S' && !in_class) {
        result += std::string("[^") + kUnicodeSpaceMembers + "]";
      } else {
        // \S inside a class has no RE2 spelling as a list of members, so it
        // keeps RE2's ASCII meaning.
        result += c;
        result += next;
      }
      continue;
    }
    result += c;
    if (!in_class && c == '[') {
      in_class = true;
      // A ']' right after '[' or '[^' is a member, not the end of the class.
      if (i + 1 < pattern.size() && pattern[i + 1] == '^') {
        result += pattern[++i];
      }
      if (i + 1 < pattern.size() && pattern[i + 1] == ']') {
        result += pattern[++i];
      }
    } else if (
        in_class && c == '[' && i + 1 < pattern.size() &&
        pattern[i + 1] == ':') {
      // Copy a POSIX class such as [:alpha:] so its ']' does not end the
      // enclosing class.
      const size_t end = pattern.find(":]", i + 2);
      if (end != std::string::npos) {
        result.append(pattern, i + 1, end + 1 - i);
        i = end + 1;
      }
    } else if (in_class && c == ']') {
      in_class = false;
    }
  }
  return result;
}

} // namespace

Error Re2Regex::compile(const std::string& pattern) {
  regex_ = std::make_unique<re2::RE2>(with_unicode_whitespace(pattern));
  // Warmup re2 as it is slow on the first run, void the return value as it's
  // not needed Refer to
  // https://github.com/google/re2/blob/6dcd83d60f7944926bfd308cc13979fc53dd69ca/re2/fuzzing/re2_fuzzer.cc#L136-L141
  (void)regex_->ReverseProgramSize();
  if (regex_->ok()) {
    return Error::Ok;
  } else {
    // It should log using Error level but it's too confusing.
    TK_LOG(
        Info,
        "Re2 failed to compile regex: %s, error: %s\nThis may be ok if a fallback regex is used.",
        pattern.c_str(),
        regex_->error().c_str());
    return Error::RegexFailure;
  }
}

std::vector<Match> Re2Regex::find_all(const std::string& text) const {
  if (!regex_ || !regex_->ok()) {
    TK_LOG(Error, "Regex is not compiled or invalid, run compile() first");
    return std::vector<Match>{};
  }
  std::vector<Match> result;
  re2::StringPiece input(text);
  re2::StringPiece piece;

  const char* base = input.data();

  while (RE2::FindAndConsume(&input, *regex_, &piece)) {
    size_t start = piece.data() - base;
    result.push_back({start, start + piece.size()});
  }

  return result;
}

} // namespace tokenizers
