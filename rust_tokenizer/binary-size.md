# Hugging Face Rust tokenizer backend binary size

Measured on Apple arm64 with Rust 1.98.1, `CMAKE_BUILD_TYPE=Release`,
`TOKENIZERS_OPTIMIZE_SIZE=ON`, dead stripping, and `gzip -9`. Each smoke binary
loads the corresponding GPT-2 artifact, encodes `Hello world`, performs
vocabulary lookups, and decodes both output tokens.

| Configuration | Stripped | Gzipped |
|---|---:|---:|
| Default (`TOKENIZERS_BUILD_HF_RUST_TOKENIZER=OFF`) | 0 B added | 0 B added |
| `.tok` only | 694,768 B | 353,651 B |
| JSON only | 1,921,872 B | 901,688 B |
| JSON and `.tok` | 2,105,056 B | 1,006,186 B |

The `.tok` row uses `TOKENIZERS_HF_RUST_FORMATS=tok`; it does not link the JSON
reader or canonicalizer. The JSON row uses the published v1.0.0-rc.2 crates.
These are complete smoke executables that load and round-trip GPT-2, not archive
sizes. The OFF configuration exposes no Rust CMake target and produces no Cargo
build directory.
