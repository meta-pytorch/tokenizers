# Hugging Face Rust tokenizer backend

This backend is opt-in and disabled by default. It can load the standard
Hugging Face `tokenizer.json` format through the v1 release candidate, the
experimental `.tok` format, or both. Enable it with
`-DTOKENIZERS_BUILD_HF_RUST_TOKENIZER=ON`. When disabled, CMake does not invoke
Cargo or compile the C++ bridge, so the default tokenizer library is unchanged.
ExecuTorch users also set `-DEXECUTORCH_BUILD_EXTENSION_LLM=ON`.

Select the linked formats with `TOKENIZERS_HF_RUST_FORMATS` (or
`EXECUTORCH_HF_RUST_FORMATS` from the ExecuTorch root):

| Value | Runtime input | Dependency source |
|---|---|---|
| `json` | `tokenizer.json` | published Hugging Face v1 RC crates |
| `tok` | `tokenizer.tok` | pinned experimental `.tok` branch |
| `json,tok` | both; JSON is preferred for directory inputs | both stacks |

The default is `json`, keeping the supported RC path independent of the
experimental branch. A size-focused `.tok`-only build is:

```sh
cmake -S . -B build \
  -DTOKENIZERS_BUILD_HF_RUST_TOKENIZER=ON \
  -DTOKENIZERS_HF_RUST_FORMATS=tok \
  -DTOKENIZERS_OPTIMIZE_SIZE=ON
```

Use `-DTOKENIZERS_OPTIMIZE_SIZE=ON` for the Rust `minsize` profile; ExecuTorch
forwards `EXECUTORCH_OPTIMIZE_SIZE` to this option.

The JSON path uses Hugging Face `tokenizers` v1.0.0-rc.2's `tk-encode`,
`tk-serialize`, and `tk-convert` crates. It accepts normal Hub
`tokenizer.json` files, canonicalizes legacy v1 JSON in memory, and uses the
upstream decoder chain. If a directory is passed, it reads `tokenizer.json` and
uses sibling `tokenizer_config.json` or `special_tokens_map.json` for BOS/EOS
metadata when available.

The `.tok` path stays available independently. Create its artifact offline with
the `tk-convert` tool from Hugging Face's
[`feat/tok-format`](https://github.com/huggingface/tokenizers/tree/feat/tok-format)
branch:

```sh
cargo run --release --manifest-path tokenizers/Cargo.toml -p tk-convert -- \
  /path/to/tokenizer.json
```

This writes `/path/to/tokenizer.tok`. The v1 container's encoder supports BPE,
Unigram, WordPiece, and WordLevel models, but it does not yet serialize decoder
configuration. The ExecuTorch adapter therefore rejects non-byte-level files
instead of returning raw, incorrectly decoded vocabulary pieces. Conversion
fails instead of silently dropping unsupported normalizer or pre-tokenizer
behavior.

The adapter also requires identifiable BOS and EOS IDs because the ExecuTorch
tokenizer interface cannot represent either value as absent.

An explicit file path always selects its format. For a directory containing
both files, JSON is preferred. If the selected Rust backend cannot load a JSON
file, the LLM runner retains its existing C++ `HFTokenizer` fallback.

This integration currently supports host CMake builds. Android, Apple
framework, WASM, and Buck packaging still need explicit Rust target/toolchain
integration. The build requires Cargo and fetches its pinned dependencies on
the first run.

The Rust dependencies are pinned to Hugging Face tokenizers commit
`054bdf469b2cf416c0da953923d88c82f4d765b5` from `feat/tok-format`. A `tok`
build links only that pipeline and reader: the released JSON reader,
canonicalizer, serde, training, and progress-bar code stay out of the binary.
The `.tok` format still lacks decoder configuration, so only byte-level files
are accepted. The JSON path does not have that limitation.

Hugging Face has announced inference-only C and C++ bindings for ExecuTorch and
llama.cpp on the v1 roadmap. The local C ABI is deliberately small so the JSON
implementation can move to that upstream binding when it stabilizes.
