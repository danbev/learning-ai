## Embedding presets
This is a task to enable presets for llama-embeddings which can be run
like this:
```console
build/bin/llama-embedding -m models/bge-small-en-v1.5-f16.gguf \
    --pooling none -c 512 -p "Hello, how are you?" --embd-normalize 2 \
    --verbose-prompt --no-warmup
```
This is quite verbose and cubersome to remember. There has been a suggestion
to add preset values to `llama-tts` (Text To Speach).
```cpp
    add_opt(common_arg(
        {"--embd-bge-small-en-default"},
        string_format("use default FlagEmbeddings models (note: can download weights from the internet)"),
        [](common_params & params) {
            params.hf_repo = "CompendiumLabs/bge-small-en-v1.5-gguf";
            params.hf_file = "bge-small-en-v1.5-f16.gguf";
            params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            params.embd_normalize = 2;
            params.n_ctx = 512;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
```
This will try to download the model from Huggingface is it is not found in the
local cache which on my machine is `~/.cache/llama.cpp`. I was not sure how
this cache directory was "chosen", but this is how it is determined:
```cpp
int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }
    ...
```
```c++
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    auto ctx_arg = common_params_parser_init(params, ex, print_usage);
    const common_params params_org = ctx_arg.params; // the example can modify the default params

    try {
        if (!common_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
```
```cpp
static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {
    ...
    common_params_handle_model_default(params.model,
                                       params.model_url,
                                       params.hf_repo,
                                       params.hf_file,
                                       params.hf_token,
                                       DEFAULT_MODEL_PATH);
```
```cpp
        if (model.empty()) {
            // this is to avoid different repo having same file name, or same file name in different subdirs
            std::string filename = hf_repo + "_" + hf_file;
            // to make sure we don't have any slashes in the filename
            string_replace_all(filename, "/", "_");
            model = fs_get_cache_file(filename);
```
```cpp
std::string fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    std::string cache_directory = fs_get_cache_directory();
    const bool success = fs_create_directory_with_parents(cache_directory);
    if (!success) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory);
    }
    return cache_directory + filename;
}
```
```cpp
std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#ifdef __linux__
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#endif // __linux__
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}
```
So on linux it is the environment variable `LLAMA_CACHE` if set, or
`XDG_CACHE_HOME` if set, or `HOME/.cache/llama.cpp`.

The following PR has been opened for this work:
https://github.com/ggerganov/llama.cpp/pull/11677

### HuggingFace org curated models
As part of the feedback on the above linked PR it was suggested that we add
the models that we use to the [ggml-org](https://huggingface.co/ggml-org) in a
new collection named `llama.cpp presets`. And then we would upload/push (not
sure how this works get) the models that we use. This allows us to control the
models and avoid the issue that the model repository are subject to change or
malicious intervention.

For example, lets take this model (from common/arg.cpp):
```cpp
    add_opt(common_arg(
        {"--embd-gte-small-default"},
        string_format("use default gte-small model (note: can download weights from the internet)"),
        [](common_params & params) {
            params.hf_repo = "ChristianAzinn/gte-small-gguf";
            params.hf_file = "gte-small.Q8_0.gguf";
            params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            params.embd_normalize = 2;
            params.n_ctx = 512;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER}));
```
We can use [gguf-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo)
to create a gguf model from the original model `thenlper/gte-small` as the
`Hub Model ID`. This will create a model in your user account and show a link
to it. For example, this is the [repo](https://huggingface.co/danbev/gte-small-Q8_0-GGUF)
that I created for this model.

We want to transfer this model to the `ggml-org` organization and this can
be done by clicking on [settings](https://huggingface.co/danbev/gte-small-Q8_0-GGUF/settings)
and then in the `Rename or transfer this model` section we can transfer the
model to the `ggml-org` organization.

To create a new collection we can use the `+ New` button (Next to `Activity Feed`
at the same level as `ggml.ai`).

After the collection has been created, or if it already exists then we can
click on the `Add to collection` button to add the model to this collection.
In this case the model name is `ggml-org/gte-small-Q8_0-GGUF`.

After this we should be able to update our preset to use the ggml-org model
instead:
```cpp
    add_opt(common_arg(
        {"--embd-gte-small-default"},
        string_format("use default gte-small model (note: can download weights from the internet)"),
        [](common_params & params) {
            params.hf_repo = "ggml-org/gte-small-Q8_0-GGUF";
            params.hf_file = "gte-small-q8_0.gguf";
            params.pooling_type = LLAMA_POOLING_TYPE_NONE;
            params.embd_normalize = 2;
            params.n_ctx = 512;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER}));
```

```

