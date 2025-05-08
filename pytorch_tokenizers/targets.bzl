load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.python_library(
        name = "tokenizers",
        srcs = [
            "__init__.py",
            "constants.py",
            "llama2c.py",
            "tiktoken.py",
            "hf_tokenizer.py",
        ],
        base_module = "pytorch_tokenizers",
        visibility = ["PUBLIC"],
        _is_external_target = True,
        external_deps = [
            "sentencepiece-py",
        ],
        deps = [
            "fbsource//third-party/pypi/blobfile:blobfile",
            "fbsource//third-party/pypi/tiktoken:tiktoken",
            "fbsource//third-party/pypi/tokenizers:tokenizers",
        ],
    )
