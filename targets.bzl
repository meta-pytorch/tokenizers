load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_executorch_supported_platforms", "runtime")
load("@fbsource//xplat/executorch/third-party:glob_defs.bzl", "subdir_glob")

PLATFORMS = get_executorch_supported_platforms()

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "headers",
        exported_headers = subdir_glob([
            ("include", "pytorch/tokenizers/*.h"),
        ]),
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        header_namespace = "",
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "regex",
        srcs = ["src/regex.cpp"] + glob([
            "src/*_regex.cpp",
        ]),
        exported_headers = subdir_glob([
            ("include", "pytorch/tokenizers/regex.h"),
            ("include", "pytorch/tokenizers/*_regex.h"),
        ]),
        exported_deps = [
            ":headers",
        ],
        exported_external_deps = [
            "pcre2",
            "re2",
        ],
        visibility = ["//pytorch/tokenizers/..."],
        header_namespace = "",
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "sentencepiece",
        srcs = [
            "src/sentencepiece.cpp",
        ],
        deps = [
            ":regex",
        ],
        exported_deps = [
            ":headers",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        external_deps = [
            "sentencepiece",
            "abseil-cpp",
        ],
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "tiktoken",
        srcs = [
            "src/tiktoken.cpp",
            "src/bpe_tokenizer_base.cpp",
        ],
        deps = [
            ":regex",
        ],
        exported_deps = [
            ":headers",
        ],
        exported_external_deps = [
            "re2",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "hf_tokenizer",
        srcs = [
            "src/hf_tokenizer.cpp",
            "src/bpe_tokenizer_base.cpp",
            "src/pre_tokenizer.cpp",
            "src/token_decoder.cpp",
        ],
        deps = [
            ":regex",
        ],
        exported_deps = [
            ":headers",
            "//pytorch/tokenizers/third-party:unicode",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        exported_external_deps = [
            "re2",
            "nlohmann_json",
        ],
        platforms = PLATFORMS,
    )

    runtime.cxx_library(
        name = "llama2c_tokenizer",
        srcs = [
            "src/llama2c_tokenizer.cpp",
        ],
        exported_deps = [
            ":headers",
        ],
        visibility = [
            "@EXECUTORCH_CLIENTS",
            "//pytorch/tokenizers/...",
        ],
        platforms = PLATFORMS,
    )
