# Experimental Hugging Face Rust tokenizer backend

This backend is opt-in and disabled by default. Enable it with
`-DTOKENIZERS_BUILD_HF_RUST_TOKENIZER=ON`. When disabled, CMake does not enter
this directory, invoke Cargo, build the Rust archive, or compile the C++
bridge, so the default tokenizer library is unchanged. ExecuTorch users can
enable the same target with both `-DEXECUTORCH_BUILD_EXTENSION_LLM=ON` and
`-DEXECUTORCH_BUILD_HF_RUST_TOKENIZER=ON`.
Use `-DTOKENIZERS_OPTIMIZE_SIZE=ON` for the Rust `minsize` profile; ExecuTorch
forwards `EXECUTORCH_OPTIMIZE_SIZE` to this option.

When enabled, the LLM runner uses Hugging Face's parser-free inference pipeline
for `.tok` files while retaining the existing ExecuTorch `Tokenizer` interface.
A directory input uses `<directory>/tokenizer.tok`. JSON tokenizers continue
through the existing C++ `HFTokenizer`; all other tokenizer fallbacks are
unchanged.

Create the artifact offline with the `tk-convert` tool from Hugging Face's
[`feat/tok-format`](https://github.com/huggingface/tokenizers/tree/feat/tok-format)
branch:

```sh
cargo run --release --manifest-path tokenizers/Cargo.toml -p tk-convert -- \
  /path/to/tokenizer.json
```

This writes `/path/to/tokenizer.tok`. The v1 container supports BPE, Unigram,
WordPiece, and WordLevel models, but intentionally supports only the
normalizers and pre-tokenizers represented by the format. Conversion fails
instead of silently dropping unsupported behavior.

This experiment currently supports host CMake builds. Android, Apple framework,
WASM, and Buck packaging still need explicit Rust target/toolchain integration.
The build requires Cargo and fetches the pinned Rust dependencies on its first
run.

The Rust dependencies are pinned to Hugging Face tokenizers commit
`054bdf469b2cf416c0da953923d88c82f4d765b5` from `feat/tok-format`. Only
`tk-encode` and the zero-dependency `.tok` reader are linked; JSON conversion,
serde, training, and progress-bar code stay out of the runtime binary. The
Rust pipeline owns encoding. The C++ compatibility layer reads vocabulary and
special-token metadata from the same `.tok` image and applies byte-level
decoding when the format marks the model as byte-level. Other decoder chains
are not represented by `.tok` v1 and therefore retain raw-piece incremental
decode behavior.
