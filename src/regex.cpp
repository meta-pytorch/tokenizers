/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Default implementation for create_regex, only using RE2 regex library.
// regex_lookahead.cpp has the implementation of create_regex with lookahead
// support, backed by PCRE2 and std::regex.

#include <pytorch/tokenizers/re2_regex.h>
#include <pytorch/tokenizers/regex.h>

namespace tokenizers {

// Default implementation that returns failure
static Result<std::unique_ptr<IRegex>> default_create_fallback_regex(
    const std::string& pattern) {
  (void)pattern;
  return tokenizers::Error::RegexFailure;
}

#ifdef SUPPORT_REGEX_LOOKAHEAD
// Declared here instead of in the public header because it exists only when
// regex_lookahead.cpp is built. Naming it is also what keeps that file in the
// link: a translation unit that only registers itself from a static
// initializer has nothing pointing at it, so the linker drops it out of a
// static archive unless the whole archive is force-loaded, and the Apple
// frameworks merge everything into one archive without that flag.
Result<std::unique_ptr<IRegex>> create_fallback_regex(
    const std::string& pattern);

FallbackRegexFn fallback_regex = create_fallback_regex;
#else
FallbackRegexFn fallback_regex = default_create_fallback_regex;
#endif

bool register_override_fallback_regex(FallbackRegexFn fn) {
  TK_LOG(Info, "Registering override fallback regex");
  fallback_regex = fn;
  return true;
}

FallbackRegexFn get_fallback_regex() {
  return fallback_regex;
}

std::string IRegex::escape(const std::string& input) {
  std::string result;
  result.reserve(input.size() * 2); // Reserve space for potential escaping

  for (char c : input) {
    // Escape regex special characters to treat them as literal strings
    if (c == '\\' || c == '^' || c == '$' || c == '.' || c == '|' || c == '?' ||
        c == '*' || c == '+' || c == '(' || c == ')' || c == '[' || c == ']' ||
        c == '{' || c == '}') {
      result += '\\';
    }
    result += c;
  }

  return result;
}

Result<std::unique_ptr<IRegex>> create_regex(const std::string& pattern) {
  // Try RE2 first
  auto re2 = std::make_unique<Re2Regex>();
  auto err = re2->compile("(" + pattern + ")");

  if (err == Error::Ok) {
    return static_cast<std::unique_ptr<IRegex>>(std::move(re2));
  }

  auto fallback = get_fallback_regex();
  auto res = fallback(pattern);
  if (res.ok()) {
    return res;
  }
  // Asking the pointer, not the build flags, keeps the advice right under both
  // build systems: only CMake defines SUPPORT_REGEX_LOOKAHEAD, while buck links
  // the same fallback in through its own whole-archive setting.
  if (fallback == default_create_fallback_regex) {
    TK_LOG(
        Error,
        "RE2 could not compile the pattern and no fallback regex engine is linked. Turn on SUPPORT_REGEX_LOOKAHEAD, or depend on the regex_lookahead target, to support patterns such as lookahead.");
  } else {
    TK_LOG(Error, "No available regex engine could compile the pattern.");
  }

  return tokenizers::Error::RegexFailure;
}
} // namespace tokenizers
