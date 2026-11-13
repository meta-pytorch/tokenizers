# HF chat template rendering

Enable `TOKENIZERS_BUILD_CHAT_TEMPLATE` and `TOKENIZERS_BUILD_TOOLS`, then build
`chat_template_tool`. This optional target uses the pinned Minja Jinja renderer;
the default tokenizer build does not compile it.

```bash
cmake -S . -B build -DTOKENIZERS_BUILD_CHAT_TEMPLATE=ON -DTOKENIZERS_BUILD_TOOLS=ON
cmake --build build --target chat_template_tool
build/examples/chat_template_tool/chat_template_tool model_dir messages.json kwargs.json
```

`messages.json` uses HF's role/content conversation structure, for example:

```json
[{"role": "user", "content": "Give me a short introduction to large language models."}]
```

`kwargs.json` can contain `{"enable_thinking": false}` or other variables used by
the model template. A final `0` or `1` argument controls `add_generation_prompt`
(default `1`). Output is the rendered prompt, with no additional newline.

C++ runners can link `tokenizers::chat_template` and use the same API:

```cpp
tokenizers::HFChatTemplate renderer;
if (renderer.load(model_directory) != tokenizers::Error::Ok) {
  // Report a missing/unsupported template or choose an explicit app fallback.
}
auto prompt = renderer.apply(messages, true, {{"enable_thinking", false}});
if (prompt.ok()) {
  auto ids = tokenizer.encode(*prompt, 0, 0);
}
```

Rendering is separate from tokenization and generation. Encode without adding
BOS/EOS again because the model's template supplies the required special tokens.
`enable_thinking` changes only templates that reference that variable.

The loader reads `chat_template.jinja`, legacy `tokenizer_config.json` templates
or legacy processor `chat_template.json`, plus string/object special-token
metadata. Standalone Jinja overrides legacy templates. Named templates can be
selected with `load(directory, "tool_use")`; automatic tool-template selection is
not performed. Named standalone files live in `additional_chat_templates`.

Missing artifacts return `LoadFailure`; malformed/unsupported templates and
rendering failures return `ParseFailure`. Failed reloads preserve the old template.
Each artifact is limited to 1 MiB. Load before sharing a renderer; concurrent
`apply` calls have independent context, but concurrent reloads are not supported.
Use trusted model templates: Minja implements a subset of Jinja for chat models,
not every Python/Jinja feature. This first library/example change does not alter
existing ExecuTorch app prompts or add a Kotlin/JNI binding for this new API.
The extra `capitalize` filter covers ASCII role names; Unicode case expansion is
unsupported and returns a render error. Multimodal messages must use the content
block structure expected by their processor template.

Fixtures pin Qwen3 and SmolVLM templates and contain exact reference output from
Transformers, covering Unicode, multiple turns, image content, generation prompts
and thinking enabled/disabled/default. Source revisions and hashes are in
`test/resources/chat_templates/manifest.json`. Fixture templates are Apache-2.0;
Minja is MIT. The normal project tests include the renderer when the option is on.
