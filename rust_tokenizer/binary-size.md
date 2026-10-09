# Hugging Face Rust tokenizer backend binary size

Measured on Apple arm64 with Rust 1.98.1, `CMAKE_BUILD_TYPE=Release`,
`TOKENIZERS_OPTIMIZE_SIZE=ON`, dead stripping, and `gzip -9`. The smoke binary
loads a GPT-2 `tokenizer.json`, encodes `Hello world`, performs vocabulary
lookups, and decodes both output tokens.

| Configuration | Stripped | Gzipped |
|---|---:|---:|
| Default (`TOKENIZERS_BUILD_HF_RUST_TOKENIZER=OFF`) | 0 B added | 0 B added |
| JSON backend enabled | 1,921,872 B | 901,688 B |

This is a complete smoke executable that loads and round-trips GPT-2, not an
archive size. The OFF configuration exposes no Rust CMake target and produces
no Cargo build directory.
