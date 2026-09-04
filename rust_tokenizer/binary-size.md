# Hugging Face `.tok` backend binary size

Measured on Apple arm64 with Rust 1.98.1, `CMAKE_BUILD_TYPE=Release`,
`TOKENIZERS_OPTIMIZE_SIZE=ON`, dead stripping, and `gzip -9`. The smoke binary
loads a GPT-2 `.tok`, encodes `Hello world`, performs vocabulary lookups, and
decodes both output tokens.

| Configuration | Stripped | Gzipped |
|---|---:|---:|
| Default (`TOKENIZERS_BUILD_HF_RUST_TOKENIZER=OFF`) | 0 B added | 0 B added |
| Opt-in `.tok` backend | 591,040 B | 298,961 B |

The OFF configuration exposes no Rust CMake target and produces no Cargo build
directory.
