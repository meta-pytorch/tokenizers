use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};

use tk_encode::pipeline::PipelineTokenizer;
use tk_serialization::{AddedEntry, Entry, TokFile, added_flag, kind};

struct Handle {
    tokenizer: PipelineTokenizer,
    // Keep the aligned `.tok` image alive. `PipelineTokenizer::from_tok`
    // currently owns the structures it builds, but retaining the image makes
    // the FFI safe if the upstream reader starts borrowing sections later.
    file: TokFile,
}

/// Create a tokenizer from a `.tok` path.
///
/// # Safety
///
/// `path` must be null or point to a valid NUL-terminated C string for the
/// duration of this call. A non-null return value must eventually be passed
/// exactly once to [`tokenizers_hf_destroy`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_create(path: *const c_char) -> *mut c_void {
    catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        let Ok(path) = unsafe { CStr::from_ptr(path) }.to_str() else {
            return std::ptr::null_mut();
        };
        let Ok(file) = TokFile::open(path) else {
            return std::ptr::null_mut();
        };
        let Ok(tokenizer) = PipelineTokenizer::from_tok(file.bytes()) else {
            return std::ptr::null_mut();
        };
        Box::into_raw(Box::new(Handle { tokenizer, file })).cast()
    }))
    .unwrap_or(std::ptr::null_mut())
}

/// Encode one UTF-8 string into caller-owned token storage.
///
/// # Safety
///
/// `opaque` must be a live handle returned by
/// [`tokenizers_hf_create`]. `text` must address `text_len` readable
/// bytes unless `text_len` is zero. `output` must address `output_capacity`
/// writable `u32` values unless the capacity is zero.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_encode(
    opaque: *const c_void,
    text: *const u8,
    text_len: usize,
    add_special_tokens: u8,
    output: *mut u32,
    output_capacity: usize,
) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() || (text.is_null() && text_len != 0) {
            return -1;
        }
        let handle = unsafe { &*opaque.cast::<Handle>() };
        let bytes = if text_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(text, text_len) }
        };
        let Ok(text) = std::str::from_utf8(bytes) else {
            return -1;
        };

        let Ok(tokens) = handle.tokenizer.encode(text, add_special_tokens != 0) else {
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
        for (index, token) in tokens.iter().enumerate() {
            unsafe { output.add(index).write(token.id) };
        }
        token_count
    }))
    .unwrap_or(-1)
}

/// Return the number of vocabulary and added-token records in the `.tok`.
///
/// This is a record count rather than a vocabulary size because an added token
/// can deliberately reuse a model-vocabulary id. The C++ compatibility layer
/// removes those duplicates while constructing its lookup maps.
///
/// # Safety
///
/// `opaque` must be null or a live handle returned by
/// [`tokenizers_hf_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_token_count(opaque: *const c_void) -> isize {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() {
            return -1;
        }
        let handle = unsafe { &*opaque.cast::<Handle>() };
        let Ok(reader) = handle.file.reader() else {
            return -1;
        };
        let Ok(vocab) = reader.require::<Entry>(kind::VOCAB_ENTRY) else {
            return -1;
        };
        let Ok(added) = reader.section::<AddedEntry>(kind::ADDED_ENTRY) else {
            return -1;
        };
        vocab
            .len()
            .checked_add(added.len())
            .and_then(|count| isize::try_from(count).ok())
            .unwrap_or(-1)
    }))
    .unwrap_or(-1)
}

/// Read one vocabulary record. Returned string bytes borrow `opaque` and stay
/// valid until the handle is destroyed.
///
/// # Safety
///
/// `opaque` must be a live handle returned by
/// [`tokenizers_hf_create`]. Every output argument must point to
/// writable storage of its declared type.
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
        let Ok(reader) = handle.file.reader() else {
            return -1;
        };
        let Ok(vocab) = reader.require::<Entry>(kind::VOCAB_ENTRY) else {
            return -1;
        };
        let (slab, start, len, token_id, added, special) = if index < vocab.len() {
            let Ok(slab) = reader.require::<u8>(kind::VOCAB_SLAB) else {
                return -1;
            };
            let entry = vocab[index];
            (slab, entry.start, entry.len, entry.id, false, false)
        } else {
            let Ok(entries) = reader.section::<AddedEntry>(kind::ADDED_ENTRY) else {
                return -1;
            };
            let Some(entry) = entries.get(index - vocab.len()).copied() else {
                return -1;
            };
            let Ok(slab) = reader.section::<u8>(kind::ADDED_SLAB) else {
                return -1;
            };
            (
                slab,
                entry.start,
                entry.len,
                entry.id,
                true,
                entry.flags & added_flag::SPECIAL != 0,
            )
        };
        let Some(end) = (start as usize).checked_add(len as usize) else {
            return -1;
        };
        let Some(bytes) = slab.get(start as usize..end) else {
            return -1;
        };
        if std::str::from_utf8(bytes).is_err() {
            return -1;
        }
        unsafe {
            id.write(token_id);
            text.write(bytes.as_ptr());
            text_len.write(bytes.len());
            is_added.write(u8::from(added));
            is_special.write(u8::from(special));
        }
        0
    }))
    .unwrap_or(-1)
}

/// Return the first prefix token (`suffix == 0`) or last suffix token
/// (`suffix != 0`) from the `.tok` post-processor. Returns 0 when present, 1
/// when the requested side is empty, and -1 on error.
///
/// # Safety
///
/// `opaque` must be a live handle returned by
/// [`tokenizers_hf_create`] and `token` must point to a writable
/// `u32`.
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
        let handle = unsafe { &*opaque.cast::<Handle>() };
        let Ok(reader) = handle.file.reader() else {
            return -1;
        };
        let Ok(tokens) = reader.section::<u32>(if suffix == 0 {
            kind::POST_PREFIX
        } else {
            kind::POST_SUFFIX
        }) else {
            return -1;
        };
        let value = if suffix == 0 {
            tokens.first()
        } else {
            tokens.last()
        };
        let Some(value) = value else {
            return 1;
        };
        unsafe { token.write(*value) };
        0
    }))
    .unwrap_or(-1)
}

/// Return `.tok` config flags, or `u32::MAX` for an invalid handle/image.
///
/// # Safety
///
/// `opaque` must be null or a live handle returned by
/// [`tokenizers_hf_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tokenizers_hf_config_flags(opaque: *const c_void) -> u32 {
    catch_unwind(AssertUnwindSafe(|| {
        if opaque.is_null() {
            return u32::MAX;
        }
        let handle = unsafe { &*opaque.cast::<Handle>() };
        handle
            .file
            .reader()
            .map(|reader| reader.config.flags)
            .unwrap_or(u32::MAX)
    }))
    .unwrap_or(u32::MAX)
}

/// Destroy a tokenizer handle.
///
/// # Safety
///
/// `opaque` must be null or a live handle returned by
/// [`tokenizers_hf_create`] that has not previously been destroyed.
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
    use tk_serialization::{Config, Writer, model, pretok, strings};

    fn test_image() -> Vec<u8> {
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

    fn with_handle(test: impl FnOnce(*mut c_void)) {
        let path = std::env::temp_dir().join(format!(
            "pytorch-tokenizers-hf-{}-{}.tok",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, test_image()).unwrap();
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let handle = unsafe { tokenizers_hf_create(c_path.as_ptr()) };
        assert!(!handle.is_null());
        test(handle);
        unsafe { tokenizers_hf_destroy(handle) };
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn loads_and_encodes_tok() {
        with_handle(|handle| {
            let text = b"hello";
            let required = unsafe {
                tokenizers_hf_encode(
                    handle,
                    text.as_ptr(),
                    text.len(),
                    1,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(required, 3);
            let mut output = [0u32; 3];
            let written = unsafe {
                tokenizers_hf_encode(
                    handle,
                    text.as_ptr(),
                    text.len(),
                    1,
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            assert_eq!(written, 3);
            assert_eq!(output, [2, 1, 2]);
        });
    }

    #[test]
    fn exposes_tok_metadata() {
        with_handle(|handle| {
            assert_eq!(unsafe { tokenizers_hf_token_count(handle) }, 3);

            let mut id = 0;
            let mut text = std::ptr::null();
            let mut len = 0;
            let mut added = 0;
            let mut special = 0;
            assert_eq!(
                unsafe {
                    tokenizers_hf_token_at(
                        handle,
                        2,
                        &mut id,
                        &mut text,
                        &mut len,
                        &mut added,
                        &mut special,
                    )
                },
                0
            );
            assert_eq!(id, 2);
            assert_eq!(unsafe { std::slice::from_raw_parts(text, len) }, b"<s>");
            assert_eq!((added, special), (1, 1));

            let mut token = 0;
            assert_eq!(
                unsafe { tokenizers_hf_post_token(handle, 0, &mut token) },
                0
            );
            assert_eq!(token, 2);
            assert_eq!(
                unsafe { tokenizers_hf_post_token(handle, 1, &mut token) },
                0
            );
            assert_eq!(token, 2);
            assert_eq!(unsafe { tokenizers_hf_config_flags(handle) }, 0);
        });
    }

    #[test]
    fn rejects_non_tok_input() {
        let path = std::env::temp_dir().join(format!(
            "pytorch-tokenizers-hf-invalid-{}.tok",
            std::process::id()
        ));
        std::fs::write(&path, b"not a tok").unwrap();
        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        assert!(unsafe { tokenizers_hf_create(c_path.as_ptr()) }.is_null());
        std::fs::remove_file(path).unwrap();
    }
}
