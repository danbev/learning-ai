## Memory 
The is a `llama_memory__i` struct that represents an interface to memory in
llama.cpp. Memory is used for models that need to preserve state across multiple
inference calls, such as chat models the use decode. This is not required for
non-causal models like embedding models and they don't perform repeated calls, the
just do a single inference call.

### Types of memory in llama.cpp
* `llama_kv_cache_unified`
* `llama_kv_cache_unified_iswa`
* `llama_memory_hybrid `
* `llama_memory_recurrent`
