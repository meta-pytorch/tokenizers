# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT

"""
Test script for hf tokenizers.
"""

import unittest
from tempfile import TemporaryDirectory

import pytest
from pytorch_tokenizers import CppHFTokenizer
from transformers import AutoTokenizer

PROMPT = "What is the capital of France?"


@pytest.mark.parametrize("model_id", ["HuggingFaceTB/SmolLM3-3B", "Qwen/Qwen2.5-0.5B"])
def test_models(model_id: str) -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer_path = tokenizer.save_pretrained(temp_dir)[-1]

        cpp_tokenizer = CppHFTokenizer()
        cpp_tokenizer.load(tokenizer_path)

        tokens = tokenizer.encode(PROMPT)
        cpp_tokens = cpp_tokenizer.encode(PROMPT)
        assert tokens == cpp_tokens


CLIP_PROMPTS = [
    "A photo of a CAT",
    "Ünïcode Straße İstanbul ΣΑΣ",
    "it's a 1024x768 image!!!",
    "  lots   of\twhitespace  ",
    "no-break\u00a0and ideographic\u3000spaces",
    "a cat <|endoftext|> a dog",
    "a " * 100,
]


@pytest.mark.parametrize("prompt", CLIP_PROMPTS)
def test_clip(prompt: str) -> None:
    with TemporaryDirectory() as temp_dir:
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer_path = tokenizer.save_pretrained(temp_dir)[-1]

        cpp_tokenizer = CppHFTokenizer()
        cpp_tokenizer.load(tokenizer_path)

        assert tokenizer.encode(prompt) == cpp_tokenizer.encode(prompt, bos=1, eos=1)


class TestHfTokenizer(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        super().setUp()

    def test_llama3_2_1b(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        tokenizer_path = tokenizer.save_pretrained(self.temp_dir.name)[-1]

        cpp_tokenizer = CppHFTokenizer()
        cpp_tokenizer.load(tokenizer_path)

        tokens = tokenizer.encode(PROMPT)
        cpp_tokens = cpp_tokenizer.encode(PROMPT, bos=1)
        self.assertEqual(tokens, cpp_tokens)

    def test_llama3_2_1b_special_toks(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        tokenizer.save_pretrained(self.temp_dir.name)

        cpp_tokenizer = CppHFTokenizer()
        cpp_tokenizer.load(self.temp_dir.name)

        tokens = tokenizer.encode(PROMPT)
        cpp_tokens = cpp_tokenizer.encode(PROMPT, bos=1)
        self.assertEqual(tokens, cpp_tokens)

        bos_id = tokenizer.convert_tokens_to_ids(
            tokenizer.special_tokens_map["bos_token"]
        )
        eos_id = tokenizer.convert_tokens_to_ids(
            tokenizer.special_tokens_map["eos_token"]
        )
        self.assertEqual(cpp_tokenizer.bos_tok(), bos_id)
        self.assertEqual(cpp_tokenizer.eos_tok(), eos_id)

    def test_phi_4_mini(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
        tokenizer_path = tokenizer.save_pretrained(self.temp_dir.name)[-1]

        cpp_tokenizer = CppHFTokenizer()
        cpp_tokenizer.load(tokenizer_path)

        tokens = tokenizer.encode(PROMPT)
        cpp_tokens = cpp_tokenizer.encode(PROMPT)
        self.assertEqual(tokens, cpp_tokens)

    def test_decode_batch(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        tokenizer_path = tokenizer.save_pretrained(self.temp_dir.name)[-1]

        cpp_tokenizer = CppHFTokenizer()
        cpp_tokenizer.load(tokenizer_path)

        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded_text = cpp_tokenizer.decode_batch(tokens, skip_special_tokens=True)
        # We use skip_special_tokens=True to match tokenizer.encode output if it includes special tokens
        # but tokenizer.decode(tokens, skip_special_tokens=True) is better comparison
        ref_decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertEqual(decoded_text, ref_decoded)
