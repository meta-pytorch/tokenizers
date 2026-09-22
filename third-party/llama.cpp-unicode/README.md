# llama.cpp Unicode

The original import of `unicode.cpp`, `unicode.h`, `unicode-data.cpp`, and
`unicode-data.h` comes from [llama.cpp commit
`54ef9cfc726a799e6f454ac22c4815d037716eda`](https://github.com/ggml-org/llama.cpp/tree/54ef9cfc726a799e6f454ac22c4815d037716eda/src).

This copy includes local tokenizer changes, including regex caching and
`unicode_cpts_normalize_nfc` in `unicode.cpp`.

The NFC tables in `include/unicode-nfc-data.h` are generated locally by
`generate_nfc_data.py` using `unicodedata2==17.0.0`. They are separate from the
imported `unicode-data` files and carry the Unicode License V3 in their header.
