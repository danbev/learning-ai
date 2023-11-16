### llm-chain-llama Embeddings
I've run into an issue where I'm trying to add Embeddings for llama.app to
llm-chain and a simple example works fine but when trying this with a
VectorStore, I'm using Qdrant, I'm getting the following error:
```console
llama_get_embeddings
len: 4096
ptr: 0x562f22554b40
Embedding query: "openssl-libs-1:1.0.2k-21.el7_9.x86_64","product":{"name":"openssl-libs-1:1.0.2k-21.el7_9.x86_64","product_id":"openssl-libs-1:1.0.2k-21.el7_9.x86_64"}},{"category":"product_version","name":"openssl-perl-1:1.0.2k-21.el7_9.x86_64","product":{"name":"openssl-perl-1:1.0.2k-21.el7_9.x86_64","product_id":"openssl-perl-1:1.0.2k-21.el7_9.x86_64"}},{"category":"product_version","name":"openssl-static-1:1.0.2k-21.el7_9.x86_64","product":{"name":"openssl-static-1:1.0.2k-21.el7_9.x86_64","product_id":"openssl-static-1:1.0.2k-21.el7_9.x86_64"}}],"category":"architecture","name":"x86_64"},{"branches":[{"category":"product_version","name":"openssl-debuginfo-1:1.0.2k-21.el7_9.i686","product":{"name":"openssl-debuginfo-1:1.0.2k-21.el7_9.i686","product_id":"openssl-debuginfo-1:1.0.2k-21.el7_9.i686"}},{"category":"product_version","name":"openssl-devel-1:1.0.2k-21.el7_9.i686","product":{"name":"openssl-devel-1:1.0.2k-21.el7_9.i686","product_id":"openssl-devel-1:1.0.2k-21.el7_9.i686"}},{"category":"
eval allocating 2152 bytes
ggml_allocr_alloc: not enough space in the buffer (needed 68864000, largest block available 65110032)
GGML_ASSERT: /home/danielbevenius/work/ai/llm-chain/crates/llm-chain-llama-sys/llama.cpp/ggml-alloc.c:131: !"not enough space in the buffer"
Aborted (core dumped)
```
Lets try running this in gdb:
```console
$ rust-gdb --args target/debug/llama
(gdb) r
...

Embedding query: "openssl-libs-1:1.0.2k-21.el7_9.x86_64","product":{"name":"openssl-libs-1:1.0.2k-21.el7_9.x86_64","product_id":"openssl-libs-1:1.0.2k-21.el7_9.x86_64"}},{"category":"product_version","name":"openssl-perl-1:1.0.2k-21.el7_9.x86_64","product":{"name":"openssl-perl-1:1.0.2k-21.el7_9.x86_64","product_id":"openssl-perl-1:1.0.2k-21.el7_9.x86_64"}},{"category":"product_version","name":"openssl-static-1:1.0.2k-21.el7_9.x86_64","product":{"name":"openssl-static-1:1.0.2k-21.el7_9.x86_64","product_id":"openssl-static-1:1.0.2k-21.el7_9.x86_64"}}],"category":"architecture","name":"x86_64"},{"branches":[{"category":"product_version","name":"openssl-debuginfo-1:1.0.2k-21.el7_9.i686","product":{"name":"openssl-debuginfo-1:1.0.2k-21.el7_9.i686","product_id":"openssl-debuginfo-1:1.0.2k-21.el7_9.i686"}},{"category":"product_version","name":"openssl-devel-1:1.0.2k-21.el7_9.i686","product":{"name":"openssl-devel-1:1.0.2k-21.el7_9.i686","product_id":"openssl-devel-1:1.0.2k-21.el7_9.i686"}},{"category":"
eval allocating 2152 bytes
ggml_allocr_alloc: not enough space in the buffer (needed 68864000, largest block available 65110032)
GGML_ASSERT: /home/danielbevenius/work/ai/llm-chain/crates/llm-chain-llama-sys/llama.cpp/ggml-alloc.c:131: !"not enough space in the buffer"

Thread 2 "tokio-runtime-w" received signal SIGABRT, Aborted.
[Switching to Thread 0x7ffff74226c0 (LWP 241021)]

(gdb) bt 10
#0  0x00007ffff74afecc in __pthread_kill_implementation () from /lib64/libc.so.6
#1  0x00007ffff745fab6 in raise () from /lib64/libc.so.6
#2  0x00007ffff74497fc in abort () from /lib64/libc.so.6
#3  0x00005555558bd09e in ggml_allocr_alloc ()
#4  0x00005555558bd67e in ggml_allocr_alloc_graph ()
#5  0x0000555555874d18 in llama_eval_internal(llama_context&, int const*, float const*, int, int, int, char const*)
    ()
#6  0x0000555555875049 in llama_eval ()
#7  0x00005555558510ad in llm_chain_llama::context::LLamaContext::llama_eval (self=0x555556c23068, 
    tokens=&[i32](size=538) = {...}, n_tokens=538, n_past=0, input=0x7ffff7420f78) at src/context.rs:286
#8  0x000055555584b02b in llm_chain_llama::embeddings::{impl#0}::get_embeddings::{async_fn#0}::{closure#0} ()
    at src/embeddings.rs:92
#9  0x000055555585cc4f in tokio::runtime::blocking::task::{impl#2}::poll<llm_chain_llama::embeddings::{impl#0}::get_embeddings::{async_fn#0}::{closure_env#0}, alloc::vec::Vec<f32, alloc::alloc::Global>> (self=..., _cx=0x7ffff7421140)
    at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/blocking/task.rs:42
(More stack frames follow...)
```
_work in progress_
