use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

#[cfg(feature = "json")]
use serde_json::Value;
#[cfg(feature = "json")]
use tk_encode_json::pipeline::{
    EncodeOptions, Override, PipelineModel, PipelineTokenizer as JsonTokenizer,
};
#[cfg(feature = "tok")]
use tk_encode_tok::pipeline::PipelineTokenizer as TokTokenizer;
#[cfg(feature = "tok")]
use tk_serialization::{AddedEntry, Entry, TokFile, added_flag, kind};

#[cfg(not(any(feature = "json", feature = "tok")))]
compile_error!("enable at least one of the `json` or `tok` features");

#[cfg(feature = "json")]
const BYTE_LEVEL_FLAG: u32 = 1 << 2;

struct TokenRecord {
    id: u32,
    text: String,
    added: bool,
    special: bool,
}

struct Metadata {
    records: Vec<TokenRecord>,
    bos: Option<u32>,
    eos: Option<u32>,
    flags: u32,
    native_decoder: bool,
}

enum Backend {
    #[cfg(feature = "json")]
    Json(JsonTokenizer),
    #[cfg(feature = "tok")]
    Tok {
        tokenizer: Box<TokTokenizer>,
        _file: TokFile,
    },
}

struct Handle {
    backend: Backend,
    metadata: Metadata,
}

impl Handle {
    fn load(path: &Path) -> Result<Self, ()> {
        match path.extension().and_then(|extension| extension.to_str()) {
            #[cfg(feature = "json")]
            Some("json") => Self::load_json(path),
            #[cfg(feature = "tok")]
            Some("tok") => Self::load_tok(path),
            _ => Err(()),
        }
    }

    #[cfg(feature = "json")]
    fn load_json(path: &Path) -> Result<Self, ()> {
        let canonical = tk_convert_json::canonicalize_file(path).map_err(|_| ())?;
        let tokenizer = tk_serialize_json::from_json(&canonical).map_err(|_| ())?;
        let mut metadata = json_metadata(&canonical)?;

        let post_processor = tokenizer.get_post_processor();
        let post_bos = post_processor
            .single
            .prefix
            .first()
            .map(|(token, _)| token.id());
        let post_eos = post_processor
            .single
            .suffix
            .last()
            .map(|(token, _)| token.id());
        let role_bos = tokenizer
            .get_token_for_role("bos_token")
            .and_then(|token| token_id(&metadata.records, token));
        let role_eos = tokenizer
            .get_token_for_role("eos_token")
            .and_then(|token| token_id(&metadata.records, token));
        let sidecar = sidecar_special_tokens(path);
        metadata.bos = role_bos
            .or_else(|| {
                sidecar
                    .0
                    .and_then(|token| token_id(&metadata.records, &token))
            })
            .or(post_bos);
        metadata.eos = role_eos
            .or_else(|| {
                sidecar
                    .1
                    .and_then(|token| token_id(&metadata.records, &token))
            })
            .or(post_eos);
        if matches!(
            tokenizer.get_model(),
            PipelineModel::BPE(model) if model.is_byte_level()
        ) {
            metadata.flags |= BYTE_LEVEL_FLAG;
        }
        Ok(Self {
            backend: Backend::Json(tokenizer),
            metadata,
        })
    }

    #[cfg(feature = "tok")]
    fn load_tok(path: &Path) -> Result<Self, ()> {
        let file = TokFile::open(path).map_err(|_| ())?;
        let tokenizer = TokTokenizer::from_tok(file.bytes()).map_err(|_| ())?;
        let reader = file.reader().map_err(|_| ())?;
        let vocab = reader.require::<Entry>(kind::VOCAB_ENTRY).map_err(|_| ())?;
        let vocab_slab = reader.require::<u8>(kind::VOCAB_SLAB).map_err(|_| ())?;
        let added = reader
            .section::<AddedEntry>(kind::ADDED_ENTRY)
            .map_err(|_| ())?;
        let added_slab = reader.section::<u8>(kind::ADDED_SLAB).map_err(|_| ())?;
        let mut records = Vec::with_capacity(vocab.len() + added.len());
        for entry in vocab {
            records.push(TokenRecord {
                id: entry.id,
                text: slab_string(vocab_slab, entry.start, entry.len)?,
                added: false,
                special: false,
            });
        }
        for entry in added {
            records.push(TokenRecord {
                id: entry.id,
                text: slab_string(added_slab, entry.start, entry.len)?,
                added: true,
                special: entry.flags & added_flag::SPECIAL != 0,
            });
        }
        let prefix = reader
            .section::<u32>(kind::POST_PREFIX)
            .map_err(|_| ())?
            .first()
            .copied();
        let suffix = reader
            .section::<u32>(kind::POST_SUFFIX)
            .map_err(|_| ())?
            .last()
            .copied();
        let flags = reader.config.flags;
        Ok(Self {
            backend: Backend::Tok {
                tokenizer: Box::new(tokenizer),
                _file: file,
            },
            metadata: Metadata {
                records,
                bos: prefix,
                eos: suffix,
                flags,
                native_decoder: false,
            },
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, ()> {
        match &self.backend {
            #[cfg(feature = "json")]
            Backend::Json(tokenizer) => {
                let options = EncodeOptions {
                    padding: Override::Off,
                    truncation: Override::Off,
                    ..EncodeOptions::no_specials()
                };
                let mut encodings = tokenizer.encode(text, &options).wait().map_err(|_| ())?;
                if encodings.len() != 1 {
                    return Err(());
                }
                Ok(encodings
                    .pop()
                    .expect("one encoding")
                    .ids()
                    .iter()
                    .map(|token| token.id())
                    .collect())
            }
            #[cfg(feature = "tok")]
            Backend::Tok { tokenizer, .. } => tokenizer
                .encode(text, false)
                .map(|tokens| tokens.into_iter().map(|token| token.id).collect())
                .map_err(|_| ()),
        }
    }

    fn decode(&self, previous: Option<u32>, token: u32, skip_special: bool) -> Result<String, ()> {
        #[cfg(not(feature = "json"))]
        let _ = (previous, token, skip_special);
        match &self.backend {
            #[cfg(feature = "json")]
            Backend::Json(tokenizer) => {
                let current = tokenizer.decode(&[token], skip_special).map_err(|_| ())?;
                let Some(previous) = previous else {
                    return Ok(current);
                };
                let prefix = tokenizer
                    .decode(&[previous], skip_special)
                    .map_err(|_| ())?;
                let pair = tokenizer
                    .decode(&[previous, token], skip_special)
                    .map_err(|_| ())?;
                Ok(pair.strip_prefix(&prefix).unwrap_or(&current).to_string())
            }
            #[cfg(feature = "tok")]
            Backend::Tok { .. } => Err(()),
        }
    }
}

#[cfg(feature = "tok")]
fn slab_string(slab: &[u8], start: u32, len: u32) -> Result<String, ()> {
    let start = start as usize;
    let end = start.checked_add(len as usize).ok_or(())?;
    std::str::from_utf8(slab.get(start..end).ok_or(())?)
        .map(str::to_owned)
        .map_err(|_| ())
}

#[cfg(feature = "json")]
fn token_id(records: &[TokenRecord], text: &str) -> Option<u32> {
    records
        .iter()
        .find(|record| record.text == text)
        .map(|record| record.id)
}

#[cfg(feature = "json")]
fn json_metadata(text: &str) -> Result<Metadata, ()> {
    let document: Value = serde_json::from_str(text).map_err(|_| ())?;
    let model = document.get("model").and_then(Value::as_object).ok_or(())?;
    let vocab = model.get("vocab").ok_or(())?;
    let mut records = Vec::new();
    if let Some(entries) = vocab.as_object() {
        for (token, id) in entries {
            records.push(TokenRecord {
                id: value_u32(id)?,
                text: token.clone(),
                added: false,
                special: false,
            });
        }
    } else if let Some(entries) = vocab.as_array() {
        for (id, entry) in entries.iter().enumerate() {
            let token = entry
                .as_array()
                .and_then(|fields| fields.first())
                .and_then(Value::as_str)
                .ok_or(())?;
            records.push(TokenRecord {
                id: u32::try_from(id).map_err(|_| ())?,
                text: token.to_owned(),
                added: false,
                special: false,
            });
        }
    } else {
        return Err(());
    }

    if let Some(added_tokens) = document.get("added_tokens").and_then(Value::as_array) {
        for token in added_tokens {
            records.push(TokenRecord {
                id: value_u32(token.get("id").ok_or(())?)?,
                text: token
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or(())?
                    .to_owned(),
                added: true,
                special: token
                    .get("special")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            });
        }
    }

    let roles = document.get("role_to_token").and_then(Value::as_object);
    let bos = roles
        .and_then(|roles| roles.get("bos_token"))
        .and_then(special_token_text)
        .and_then(|token| token_id(&records, token));
    let eos = roles
        .and_then(|roles| roles.get("eos_token"))
        .and_then(special_token_text)
        .and_then(|token| token_id(&records, token));
    Ok(Metadata {
        records,
        bos,
        eos,
        flags: 0,
        native_decoder: true,
    })
}

#[cfg(feature = "json")]
fn value_u32(value: &Value) -> Result<u32, ()> {
    value
        .as_u64()
        .and_then(|id| u32::try_from(id).ok())
        .ok_or(())
}

#[cfg(feature = "json")]
fn special_token_text(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}

#[cfg(feature = "json")]
fn sidecar_special_tokens(path: &Path) -> (Option<String>, Option<String>) {
    let mut bos = None;
    let mut eos = None;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    for filename in ["tokenizer_config.json", "special_tokens_map.json"] {
        let Ok(text) = std::fs::read_to_string(parent.join(filename)) else {
            continue;
        };
        let Ok(document) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if bos.is_none() {
            bos = document
                .get("bos_token")
                .and_then(special_token_text)
                .map(str::to_owned);
        }
        if eos.is_none() {
            eos = document
                .get("eos_token")
                .and_then(special_token_text)
                .map(str::to_owned);
        }
    }
    (bos, eos)
}

fn candidate_paths(path: &Path) -> Vec<PathBuf> {
    if !path.is_dir() {
        return vec![path.to_owned()];
    }
    vec![path.join("tokenizer.json"), path.join("tokenizer.tok")]
}

/// Create a tokenizer from a `tokenizer.json`, `.tok`, or directory containing one.
///
/// # Safety
///
/// `path` must be null or point to a valid NUL-terminated C string for the duration of this call.
/// A non-null return value must eventually be passed exactly once to [`tokenizers_hf_destroy`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_create(path: *const c_char) -> *mut c_void {
    catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        let Ok(path) = unsafe { CStr::from_ptr(path) }.to_str() else {
            return std::ptr::null_mut();
        };
        for candidate in candidate_paths(Path::new(path)) {
            if let Ok(handle) = Handle::load(&candidate) {
                return Box::into_raw(Box::new(handle)).cast();
            }
        }
        std::ptr::null_mut()
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Encode one UTF-8 string into caller-owned token storage.
///
/// # Safety
///
/// `opaque` must be a live handle returned by [`tokenizers_hf_create`]. `text` must address
/// `text_len` readable bytes unless the length is zero. `output` must address `output_capacity`
/// writable `u32` values unless the capacity is zero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_encode(
    opaque: *const c_void,
    text: *const u8,
    text_len: usize,
    output: *mut u32,
    output_capacity: usize,
) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() || (text.is_null() && text_len != 0) {
            return -1;
        }
        let bytes = if text_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(text, text_len) }
        };
        let Ok(text) = std::str::from_utf8(bytes) else {
            return -1;
        };
        let Ok(tokens) = (unsafe { &*opaque.cast::<Handle>() }).encode(text) else {
            return -1;
        };
        let Ok(token_count) = isize::try_from(tokens.len()) else {
            return -1;
        };
        if tokens.len() > output_capacity {
            return token_count;
        }
        if !tokens.is_empty() && output.is_null() {
            return -1;
        }
        if !tokens.is_empty() {
            unsafe { std::ptr::copy_nonoverlapping(tokens.as_ptr(), output, tokens.len()) };
        }
        token_count
    }))
    .unwrap_or(-1)
}

/// Decode one token with optional previous-token context into caller-owned byte storage.
///
/// `has_previous == 0` ignores `previous`. The return value is the required byte count, or -1
/// when this backend has no serialized decoder (the `.tok` format) or decoding fails.
///
/// # Safety
///
/// `opaque` must be a live handle. `output` must address `output_capacity` writable bytes unless
/// the capacity is zero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_decode(
    opaque: *const c_void,
    previous: u32,
    has_previous: u8,
    token: u32,
    skip_special: u8,
    output: *mut u8,
    output_capacity: usize,
) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() {
            return -1;
        }
        let previous = (has_previous != 0).then_some(previous);
        let Ok(decoded) =
            (unsafe { &*opaque.cast::<Handle>() }).decode(previous, token, skip_special != 0)
        else {
            return -1;
        };
        let Ok(byte_count) = isize::try_from(decoded.len()) else {
            return -1;
        };
        if decoded.len() > output_capacity {
            return byte_count;
        }
        if !decoded.is_empty() && output.is_null() {
            return -1;
        }
        if !decoded.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(decoded.as_ptr(), output, decoded.len());
            }
        }
        byte_count
    }))
    .unwrap_or(-1)
}

/// Return the number of vocabulary and added-token records.
///
/// # Safety
///
/// `opaque` must be null or a live handle returned by [`tokenizers_hf_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_token_count(opaque: *const c_void) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() {
            return -1;
        }
        isize::try_from(
            (unsafe { &*opaque.cast::<Handle>() })
                .metadata
                .records
                .len(),
        )
        .unwrap_or(-1)
    }))
    .unwrap_or(-1)
}

/// Read one vocabulary record. Returned string bytes borrow `opaque` and remain valid until the
/// handle is destroyed.
///
/// # Safety
///
/// `opaque` must be a live handle. Every output argument must point to writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_token_at(
    opaque: *const c_void,
    index: usize,
    id: *mut u32,
    text: *mut *const u8,
    text_len: *mut usize,
    is_added: *mut u8,
    is_special: *mut u8,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null()
            || id.is_null()
            || text.is_null()
            || text_len.is_null()
            || is_added.is_null()
            || is_special.is_null()
        {
            return -1;
        }
        let handle = unsafe { &*opaque.cast::<Handle>() };
        let Some(record) = handle.metadata.records.get(index) else {
            return -1;
        };
        unsafe {
            id.write(record.id);
            text.write(record.text.as_ptr());
            text_len.write(record.text.len());
            is_added.write(u8::from(record.added));
            is_special.write(u8::from(record.special));
        }
        0
    }))
    .unwrap_or(-1)
}

/// Return the configured BOS (`suffix == 0`) or EOS (`suffix != 0`) token.
///
/// # Safety
///
/// `opaque` must be a live handle and `token` must point to writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_post_token(
    opaque: *const c_void,
    suffix: u8,
    token: *mut u32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() || token.is_null() {
            return -1;
        }
        let metadata = &(unsafe { &*opaque.cast::<Handle>() }).metadata;
        let value = if suffix == 0 {
            metadata.bos
        } else {
            metadata.eos
        };
        let Some(value) = value else {
            return 1;
        };
        unsafe { token.write(value) };
        0
    }))
    .unwrap_or(-1)
}

/// Return backend flags, or `u32::MAX` for an invalid handle.
///
/// # Safety
///
/// `opaque` must be null or a live handle returned by [`tokenizers_hf_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_config_flags(opaque: *const c_void) -> u32 {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() {
            return u32::MAX;
        }
        (unsafe { &*opaque.cast::<Handle>() }).metadata.flags
    }))
    .unwrap_or(u32::MAX)
}

/// Return whether the loaded format carries a decoder, or -1 for an invalid handle.
///
/// # Safety
///
/// `opaque` must be null or a live handle returned by [`tokenizers_hf_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_has_native_decoder(opaque: *const c_void) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() {
            return -1;
        }
        i32::from(
            (unsafe { &*opaque.cast::<Handle>() })
                .metadata
                .native_decoder,
        )
    }))
    .unwrap_or(-1)
}

/// Destroy a tokenizer handle.
///
/// # Safety
///
/// `opaque` must be null or a live handle returned by [`tokenizers_hf_create`] that has not
/// previously been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_destroy(opaque: *mut c_void) {
    if !opaque.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(opaque.cast::<Handle>()) });
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn temp_path(extension: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "pytorch-tokenizers-hf-{}-{}.{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test"),
            extension
        ))
    }

    fn with_handle(path: &Path, test: impl FnOnce(*mut c_void)) {
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let handle = unsafe { tokenizers_hf_create(c_path.as_ptr()) };
        assert!(!handle.is_null());
        test(handle);
        unsafe { tokenizers_hf_destroy(handle) };
        std::fs::remove_file(path).unwrap();
    }

    fn encoded(handle: *const c_void, text: &[u8]) -> Vec<u32> {
        let required = unsafe {
            tokenizers_hf_encode(handle, text.as_ptr(), text.len(), std::ptr::null_mut(), 0)
        };
        assert!(required >= 0);
        let mut output = vec![0; required as usize];
        let written = unsafe {
            tokenizers_hf_encode(
                handle,
                text.as_ptr(),
                text.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        assert_eq!(written, required);
        output
    }

    #[cfg(feature = "json")]
    const TEST_JSON: &str = r#"{
      "version": "2.0",
      "role_to_token": {"bos_token": "<s>", "eos_token": "</s>"},
      "added_tokens": [
        {"id": 4, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 5, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null,
      "pre_tokenizer": null,
      "post_processor": null,
      "decoder": null,
      "model": {"type": "BPE", "byte_level": false, "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3}, "merges": [["a", "b"], ["ab", "ab"]]},
      "padding": null,
      "truncation": null
    }"#;

    #[cfg(feature = "json")]
    #[test]
    fn loads_encodes_and_decodes_json() {
        let path = temp_path("json");
        std::fs::write(&path, TEST_JSON).unwrap();
        with_handle(&path, |handle| {
            assert_eq!(encoded(handle, b"abab"), [3]);
            assert_eq!(unsafe { tokenizers_hf_token_count(handle) }, 6);
            assert_eq!(unsafe { tokenizers_hf_config_flags(handle) }, 0);
            assert_eq!(unsafe { tokenizers_hf_has_native_decoder(handle) }, 1);

            let mut token = 0;
            assert_eq!(
                unsafe { tokenizers_hf_post_token(handle, 0, &mut token) },
                0
            );
            assert_eq!(token, 4);
            assert_eq!(
                unsafe { tokenizers_hf_post_token(handle, 1, &mut token) },
                0
            );
            assert_eq!(token, 5);

            let required =
                unsafe { tokenizers_hf_decode(handle, 0, 0, 3, 0, std::ptr::null_mut(), 0) };
            assert_eq!(required, 4);
            let mut output = vec![0; required as usize];
            assert_eq!(
                unsafe {
                    tokenizers_hf_decode(handle, 0, 0, 3, 0, output.as_mut_ptr(), output.len())
                },
                required
            );
            assert_eq!(output, b"abab");
        });
    }

    #[cfg(feature = "tok")]
    fn test_tok_image() -> Vec<u8> {
        use tk_serialization::{Config, Writer, model, pretok, strings};

        let config = Config {
            model: model::WORDLEVEL,
            model_param: 0,
            pretok: pretok::NONE,
            pretok_param: 0,
            flags: 0,
            _pad0: 0,
            added_first: [1 << (b'<' & 63), 0, 0, 0],
        };
        let vocab_slab = b"<unk>hello";
        let vocab = [
            Entry {
                start: 0,
                len: 5,
                id: 0,
            },
            Entry {
                start: 5,
                len: 5,
                id: 1,
            },
        ];
        let added_slab = b"<s>";
        let added = [AddedEntry {
            start: 0,
            len: 3,
            id: 2,
            flags: added_flag::SPECIAL,
        }];
        let mut model_strings = Vec::new();
        strings::push(&mut model_strings, "<unk>");
        strings::push(&mut model_strings, "");
        strings::push(&mut model_strings, "");

        let mut writer = Writer::new();
        writer.push_one(kind::CONFIG, &config);
        writer.push(kind::VOCAB_SLAB, vocab_slab);
        writer.push(kind::VOCAB_ENTRY, &vocab);
        writer.push(kind::ADDED_SLAB, added_slab);
        writer.push(kind::ADDED_ENTRY, &added);
        writer.push(kind::POST_PREFIX, &[2u32]);
        writer.push(kind::POST_SUFFIX, &[2u32]);
        writer.push(kind::MODEL_STRINGS, &model_strings);
        writer.finish()
    }

    #[cfg(feature = "tok")]
    #[test]
    fn loads_and_encodes_tok() {
        let path = temp_path("tok");
        std::fs::write(&path, test_tok_image()).unwrap();
        with_handle(&path, |handle| {
            assert_eq!(encoded(handle, b"hello"), [1]);
            assert_eq!(unsafe { tokenizers_hf_token_count(handle) }, 3);
            assert_eq!(unsafe { tokenizers_hf_has_native_decoder(handle) }, 0);
            assert_eq!(
                unsafe { tokenizers_hf_decode(handle, 0, 0, 1, 0, std::ptr::null_mut(), 0) },
                -1
            );
        });
    }

    #[test]
    fn rejects_unknown_input() {
        let path = temp_path("bin");
        std::fs::write(&path, b"not a tokenizer").unwrap();
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        assert!(unsafe { tokenizers_hf_create(c_path.as_ptr()) }.is_null());
        std::fs::remove_file(path).unwrap();
    }
}
