### CUDA warnings
#### llama.cpp
```console
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu: In function ‘bool ggml_backend_cuda_register_host_buffer(void*, size_t)’:
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu:2991:51: warning: unused parameter ‘buffer’ [-Wunused-parameter]
 2991 | GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
      |                                             ~~~~~~^~~~~~
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu:2991:66: warning: unused parameter ‘size’ [-Wunused-parameter]
 2991 | GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
```

#### llm-engine:
```console
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu: In function ‘ggml_backend_buffer* ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t, size_t, std::string&)’:
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu:552:132: warning: unused parameter ‘error’ [-Wunused-parameter]
  552 | GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, std::string& error) {
      |                                                                                                                       ~~~~~~~~~~~~~^~~~~
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu: In function ‘ggml_backend_buffer* ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t, size_t, std::string&)’:
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu:864:138: warning: unused parameter ‘error’ [-Wunused-parameter]
  864 | GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, std::string& error) {
      |                                                                                                                             ~~~~~~~~~~~~~^~~~~
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu: In function ‘ggml_backend_buffer* ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t, size_t, std::string&)’:
/home/danbev/work/lmstudio/llmster/electron/vendor/llm-engine/llama.cpp/ggml/src/ggml-cuda.cu:994:137: warning: unused parameter ‘error’ [-Wunused-parameter]
  994 | GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, std::string& error) {
      |                                                                                                                            ~~~~~~~~~~~~~^~~~~
```
