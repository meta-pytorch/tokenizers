# Hugging Face Rust tokenizer backend

This backend is opt-in and disabled by default. It loads standard Hugging Face
`tokenizer.json` files through the published v1 release-candidate crates. Enable
it with `-DTOKENIZERS_BUILD_HF_RUST_TOKENIZER=ON`. When disabled, CMake does not
invoke Cargo or compile the C++ bridge, so the default tokenizer library is
unchanged. ExecuTorch users also set
`-DEXECUTORCH_BUILD_EXTENSION_LLM=ON`.

Use `-DTOKENIZERS_OPTIMIZE_SIZE=ON` for the Rust `minsize` profile; ExecuTorch
forwards `EXECUTORCH_OPTIMIZE_SIZE` to this option.

The backend uses Hugging Face tokenizers v1.0.0-rc.2's `tk-encode`,
`tk-serialize`, and `tk-convert` crates. It accepts normal Hub
`tokenizer.json` files, canonicalizes legacy v1 JSON in memory, and uses the
upstream decoder chain. If a directory is passed, it reads `tokenizer.json` and
uses sibling `tokenizer_config.json` or `special_tokens_map.json` for BOS/EOS
metadata when available.

The adapter requires identifiable BOS and EOS IDs because the ExecuTorch
tokenizer interface cannot represent either value as absent. If the Rust
backend cannot load a JSON file, the LLM runner retains its existing C++
`HFTokenizer` fallback.

This integration currently supports host CMake builds. Android, Apple
framework, WASM, and Buck packaging still need explicit Rust target/toolchain
integration. The build requires Cargo and fetches its pinned crates.io
dependencies on the first run.

Hugging Face has announced inference-only C and C++ bindings for ExecuTorch and
llama.cpp on the v1 roadmap. The local C ABI is deliberately small so the
implementation can move to that upstream binding when it stabilizes.
