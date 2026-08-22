/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Built by a target that links the static archives directly, with no
// whole-archive flag, so that it fails if regex.cpp stops referencing the
// lookahead fallback. See the comment on that target in CMakeLists.txt.

#include <gtest/gtest.h>
#include <pytorch/tokenizers/regex.h>

namespace tokenizers {

TEST(RegexLookaheadLinkTest, LookaheadPatternCompiles) {
  auto regex = create_regex(R"(\s+(?!\S))");
  EXPECT_TRUE(regex.ok());
}

} // namespace tokenizers
