## PHI 4 Mini XCFramework issue

This [issue](Isssue: https://github.com/ggml-org/llama.cpp/issues/12232) is about
loading the model `bartowski/microsoft_Phi-4-mini-instruct-GGUF` in combination with
using the new llama.cpp XCFramework.

The error message reported was the following:
```console
totalRAM 7.4770813, freeRAM 0.15818787, modelSize 2.64
RAMPressue normal

llama_model_load_from_file_impl: using device Metal (Apple A18 Pro GPU) - 5461 MiB free
llama_model_loader: loaded meta data with 40 key-value pairs and 196 tensors from /var/mobile/Containers/Data/Application/46F3617C-4A90-4449-8323-6961420AB5FC/Documents/Phi-4-mini-instruct-Q4_K_L.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = phi3
llama_model_loader: - kv   1:              phi3.rope.scaling.attn_factor f32              = 1.190238
llama_model_loader: - kv   2:                               general.type str              = model
llama_model_loader: - kv   3:                               general.name str              = Phi 4 Mini Instruct
llama_model_loader: - kv   4:                           general.finetune str              = instruct
llama_model_loader: - kv   5:                           general.basename str              = Phi-4
llama_model_loader: - kv   6:                         general.size_label str              = mini
llama_model_loader: - kv   7:                            general.license str              = mit
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/microsoft/Phi-...
llama_model_loader: - kv   9:                               general.tags arr[str,3]       = ["nlp", "code", "text-generation"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["multilingual"]
llama_model_loader: - kv  11:                        phi3.context_length u32              = 131072
llama_model_loader: - kv  12:  phi3.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  13:                      phi3.embedding_length u32              = 3072
llama_model_loader: - kv  14:                   phi3.feed_forward_length u32              = 8192
llama_model_loader: - kv  15:                           phi3.block_count u32              = 32
llama_model_loader: - kv  16:                  phi3.attention.head_count u32              = 24
llama_model_loader: - kv  17:               phi3.attention.head_count_kv u32              = 8
llama_model_loader: - kv  18:      phi3.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  19:                  phi3.rope.dimension_count u32              = 96
llama_model_loader: - kv  20:                        phi3.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  21:              phi3.attention.sliding_window u32              = 262144
llama_model_loader: - kv  22:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  23:                         tokenizer.ggml.pre str              = gpt-4o
llama_model_loader: - kv  24:                      tokenizer.ggml.tokens arr[str,200064]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  25:                  tokenizer.ggml.token_type arr[i32,200064]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  26:                      tokenizer.ggml.merges arr[str,199742]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "e r", ...
llama_model_loader: - kv  27:                tokenizer.ggml.bos_token_id u32              = 199999
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 199999
llama_model_loader: - kv  29:            tokenizer.ggml.unknown_token_id u32              = 199999
llama_model_loader: - kv  30:            tokenizer.ggml.padding_token_id u32              = 199999
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  33:                    tokenizer.chat_template str              = {% for message in messages %}{% if me...
llama_model_loader: - kv  34:               general.quantization_version u32              = 2
llama_model_loader: - kv  35:                          general.file_type u32              = 15
llama_model_loader: - kv  36:                      quantize.imatrix.file str              = /models_out/Phi-4-mini-instruct-GGUF/...
llama_model_loader: - kv  37:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  38:             quantize.imatrix.entries_count i32              = 128
llama_model_loader: - kv  39:              quantize.imatrix.chunks_count i32              = 123
llama_model_loader: - type  f32:   67 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q4_K:   80 tensors
llama_model_loader: - type q5_K:   32 tensors
llama_model_loader: - type q6_K:   16 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 2.45 GiB (5.49 BPW) 
init_tokenizer: initializing tokenizer for type 2
load: control token: 200028 '<|tag|>' is not marked as EOG
load: control token: 200027 '<|tool_response|>' is not marked as EOG
load: control token: 200026 '<|/tool_call|>' is not marked as EOG
load: control token: 200025 '<|tool_call|>' is not marked as EOG
load: control token: 200022 '<|system|>' is not marked as EOG
load: control token: 200018 '<|endofprompt|>' is not marked as EOG
load: control token: 200021 '<|user|>' is not marked as EOG
load: control token: 200024 '<|/tool|>' is not marked as EOG
load: control token: 200023 '<|tool|>' is not marked as EOG
load: control token: 200019 '<|assistant|>' is not marked as EOG
load: special tokens cache size = 12
load: token to piece cache size = 1.3333 MB
print_info: arch             = phi3
print_info: vocab_only       = 0
print_info: n_ctx_train      = 131072
print_info: n_embd           = 3072
print_info: n_layer          = 32
print_info: n_head           = 24
print_info: n_head_kv        = 8
print_info: n_rot            = 96
print_info: n_swa            = 262144
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 3
print_info: n_embd_k_gqa     = 1024
print_info: n_embd_v_gqa     = 1024
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: n_ff             = 8192
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 4096
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 3B
print_info: model params     = 3.84 B
print_info: general.name     = Phi 4 Mini Instruct
print_info: vocab type       = BPE
print_info: n_vocab          = 200064
print_info: n_merges         = 199742
print_info: BOS token        = 199999 '<|endoftext|>'
print_info: EOS token        = 199999 '<|endoftext|>'
print_info: EOT token        = 200020 '<|end|>'
print_info: UNK token        = 199999 '<|endoftext|>'
print_info: PAD token        = 199999 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: EOG token        = 199999 '<|endoftext|>'
print_info: EOG token        = 200020 '<|end|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: layer   0 assigned to device Metal
load_tensors: layer   1 assigned to device Metal
load_tensors: layer   2 assigned to device Metal
load_tensors: layer   3 assigned to device Metal
load_tensors: layer   4 assigned to device Metal
load_tensors: layer   5 assigned to device Metal
load_tensors: layer   6 assigned to device Metal
load_tensors: layer   7 assigned to device Metal
load_tensors: layer   8 assigned to device Metal
load_tensors: layer   9 assigned to device Metal
load_tensors: layer  10 assigned to device Metal
load_tensors: layer  11 assigned to device Metal
load_tensors: layer  12 assigned to device Metal
load_tensors: layer  13 assigned to device Metal
load_tensors: layer  14 assigned to device Metal
load_tensors: layer  15 assigned to device Metal
load_tensors: layer  16 assigned to device Metal
load_tensors: layer  17 assigned to device Metal
load_tensors: layer  18 assigned to device Metal
load_tensors: layer  19 assigned to device Metal
load_tensors: layer  20 assigned to device Metal
load_tensors: layer  21 assigned to device Metal
load_tensors: layer  22 assigned to device Metal
load_tensors: layer  23 assigned to device Metal
load_tensors: layer  24 assigned to device Metal
load_tensors: layer  25 assigned to device Metal
load_tensors: layer  26 assigned to device Metal
load_tensors: layer  27 assigned to device Metal
load_tensors: layer  28 assigned to device Metal
load_tensors: layer  29 assigned to device Metal
load_tensors: layer  30 assigned to device Metal
load_tensors: layer  31 assigned to device Metal
load_tensors: layer  32 assigned to device Metal
load_tensors: tensor 'token_embd.weight' (q8_0) (and 0 others) cannot be used with preferred buffer type CPU_AARCH64, using CPU instead
ggml_backend_metal_log_allocated_size: allocated buffer, size =  2510.53 MiB, ( 2510.61 /  5461.34)
load_tensors: offloading 32 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 33/33 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   622.76 MiB
load_tensors: Metal_Mapped model buffer size =  2510.53 MiB
Using 5 threads
llama_init_from_model: n_seq_max     = 1
llama_init_from_model: n_ctx         = 8192
llama_init_from_model: n_ctx_per_seq = 8192
llama_init_from_model: n_batch       = 8192
llama_init_from_model: n_ubatch      = 512
llama_init_from_model: flash_attn    = 1
llama_init_from_model: freq_base     = 10000.0
llama_init_from_model: freq_scale    = 1
llama_init_from_model: n_ctx_per_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
ggml_metal_init: allocating
ggml_metal_init: picking default device: Apple A18 Pro GPU
ggml_metal_init: using embedded metal library
Compiler failed with XPC_ERROR_CONNECTION_INTERRUPTED
Compiler failed with XPC_ERROR_CONNECTION_INTERRUPTED
Compiler failed with XPC_ERROR_CONNECTION_INTERRUPTED
MTLCompiler: Compilation failed with XPC_ERROR_CONNECTION_INTERRUPTED on 3 try
ggml_metal_init: error: Error Domain=MTLLibraryErrorDomain Code=3 "Compiler encountered an internal error" UserInfo={NSLocalizedDescription=Compiler encountered an internal error}
ggml_backend_metal_device_init: error: failed to allocate context
llama_init_from_model: failed to initialize Metal backend
Could not load context!
```
So this error is happening when the model is being compiled by by Metal. This is done in
`ggml/src/ggml-metal/ggml-metal.m`:
```c++
static struct ggml_backend_metal_context * ggml_metal_init(ggml_backend_dev_t dev) {
    ...
                metal_library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return NULL;
                }
```
When `newLibraryWithSource` is called it will take the kernel sources and send them to the
Metal compiler XPC service for compilation. 
```console
(lldb) p src
(__NSCFString *) 0x0000000148058000 @"#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n#if defined(GGML_METAL_EMBED_LIBRARY)\n#ifndef GGML_COMMON_DECL\n\n#if defined(GGML_COMMON_DECL_C)\n#include <stdint.h>\n\ntypedef uint16_t ggml_half;\ntypedef uint32_t ggml_half2;\n\n#define GGML_COMMON_AGGR_U\n#define GGML_COMMON_AGGR_S\n\n#define GGML_COMMON_DECL\n#elif defined(GGML_COMMON_DECL_CPP)\n#include <cstdint>\n\ntypedef uint16_t ggml_half;\ntypedef uint32_t ggml_half2;\n\n// std-c++ allow anonymous unions but some compiler warn on it\n#define GGML_COMMON_AGGR_U data\n// std-c++ do not allow it.\n#define GGML_COMMON_AGGR_S data\n\n#define GGML_COMMON_DECL\n#elif defined(GGML_COMMON_DECL_METAL)\n#include <metal_stdlib>\n\ntypedef half  ggml_half;\ntypedef half2 ggml_half2;\n\n#define GGML_COMMON_AGGR_U\n#define GGML_COMMON_AGGR_S\n\n#define GGML_COMMON_DECL\n#elif defined(GGML_COMMON_DECL_CUDA)\n#if defined(GGML_COMMON_DECL_MUSA)\n#include <musa_fp16.h>\n#else\n#include <cuda_fp16.h>\n#endif\n#include <cstdint>\n\ntypedef half  ggml_half;\ntypedef half2 ggml_half2;\n\n#define GGML_"
```
This is beginning of the content of the file `ggml/src/ggml-metal/ggml-metal.metal`.

So the `XPC_ERROR_CONNECTION_INTERRUPTED` error is happening perhaps because the XPC service
is experiencing some kind of interruptions while trying to compile the src (the shader code).
So the issue seems to be with the compilation of the metal shader code. In the output above
we see the following message:
```console
Memory pressure warning received
```
Perhaps the system is terminating the XPC service because of memory pressure.


Notice that the logged free GPU memory availabe is 5461 MiB.
```console
llama_model_load_from_file_impl: using device Metal (Apple A18 Pro GPU) - 5461 MiB free
```
The model size in this case is 2.64 GiB which is about 2703 MiB, so the it does not seem
that the model will not fit into the GPU memory, in fact is should be able to fit without
any issue.


And the rest of the error in the issue is:
```console
Memory pressure warning received
totalRAM 7.4770813, freeRAM 0.16456604, modelSize 2.64
currentPressue warning
llama_model_load_from_file_impl: using device Metal (Apple A18 Pro GPU) - 2950 MiB free
llama_model_loader: loaded meta data with 40 key-value pairs and 196 tensors from /var/mobile/Containers/Data/Application/46F3617C-4A90-4449-8323-6961420AB5FC/Documents/Phi-4-mini-instruct-Q4_K_L.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = phi3
llama_model_loader: - kv   1:              phi3.rope.scaling.attn_factor f32              = 1.190238
llama_model_loader: - kv   2:                               general.type str              = model
llama_model_loader: - kv   3:                               general.name str              = Phi 4 Mini Instruct
llama_model_loader: - kv   4:                           general.finetune str              = instruct
llama_model_loader: - kv   5:                           general.basename str              = Phi-4
llama_model_loader: - kv   6:                         general.size_label str              = mini
llama_model_loader: - kv   7:                            general.license str              = mit
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/microsoft/Phi-...
llama_model_loader: - kv   9:                               general.tags arr[str,3]       = ["nlp", "code", "text-generation"]
llama_model_loader: - kv  10:                          general.languages arr[str,1]       = ["multilingual"]
llama_model_loader: - kv  11:                        phi3.context_length u32              = 131072
llama_model_loader: - kv  12:  phi3.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  13:                      phi3.embedding_length u32              = 3072
llama_model_loader: - kv  14:                   phi3.feed_forward_length u32              = 8192
llama_model_loader: - kv  15:                           phi3.block_count u32              = 32
llama_model_loader: - kv  16:                  phi3.attention.head_count u32              = 24
llama_model_loader: - kv  17:               phi3.attention.head_count_kv u32              = 8
llama_model_loader: - kv  18:      phi3.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  19:                  phi3.rope.dimension_count u32              = 96
llama_model_loader: - kv  20:                        phi3.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  21:              phi3.attention.sliding_window u32              = 262144
llama_model_loader: - kv  22:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  23:                         tokenizer.ggml.pre str              = gpt-4o
llama_model_loader: - kv  24:                      tokenizer.ggml.tokens arr[str,200064]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  25:                  tokenizer.ggml.token_type arr[i32,200064]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  26:                      tokenizer.ggml.merges arr[str,199742]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "e r", ...
llama_model_loader: - kv  27:                tokenizer.ggml.bos_token_id u32              = 199999
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 199999
llama_model_loader: - kv  29:            tokenizer.ggml.unknown_token_id u32              = 199999
llama_model_loader: - kv  30:            tokenizer.ggml.padding_token_id u32              = 199999
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  33:                    tokenizer.chat_template str              = {% for message in messages %}{% if me...
llama_model_loader: - kv  34:               general.quantization_version u32              = 2
llama_model_loader: - kv  35:                          general.file_type u32              = 15
llama_model_loader: - kv  36:                      quantize.imatrix.file str              = /models_out/Phi-4-mini-instruct-GGUF/...
llama_model_loader: - kv  37:                   quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
llama_model_loader: - kv  38:             quantize.imatrix.entries_count i32              = 128
llama_model_loader: - kv  39:              quantize.imatrix.chunks_count i32              = 123
llama_model_loader: - type  f32:   67 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q4_K:   80 tensors
llama_model_loader: - type q5_K:   32 tensors
llama_model_loader: - type q6_K:   16 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 2.45 GiB (5.49 BPW) 
init_tokenizer: initializing tokenizer for type 2
load: control token: 200028 '<|tag|>' is not marked as EOG
load: control token: 200027 '<|tool_response|>' is not marked as EOG
load: control token: 200026 '<|/tool_call|>' is not marked as EOG
load: control token: 200025 '<|tool_call|>' is not marked as EOG
load: control token: 200022 '<|system|>' is not marked as EOG
load: control token: 200018 '<|endofprompt|>' is not marked as EOG
load: control token: 200021 '<|user|>' is not marked as EOG
load: control token: 200024 '<|/tool|>' is not marked as EOG
load: control token: 200023 '<|tool|>' is not marked as EOG
load: control token: 200019 '<|assistant|>' is not marked as EOG
load: special tokens cache size = 12
load: token to piece cache size = 1.3333 MB
print_info: arch             = phi3
print_info: vocab_only       = 0
print_info: n_ctx_train      = 131072
print_info: n_embd           = 3072
print_info: n_layer          = 32
print_info: n_head           = 24
print_info: n_head_kv        = 8
print_info: n_rot            = 96
print_info: n_swa            = 262144
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 3
print_info: n_embd_k_gqa     = 1024
print_info: n_embd_v_gqa     = 1024
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: n_ff             = 8192
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 4096
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 3B
print_info: model params     = 3.84 B
print_info: general.name     = Phi 4 Mini Instruct
print_info: vocab type       = BPE
print_info: n_vocab          = 200064
print_info: n_merges         = 199742
print_info: BOS token        = 199999 '<|endoftext|>'
print_info: EOS token        = 199999 '<|endoftext|>'
print_info: EOT token        = 200020 '<|end|>'
print_info: UNK token        = 199999 '<|endoftext|>'
print_info: PAD token        = 199999 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: EOG token        = 199999 '<|endoftext|>'
print_info: EOG token        = 200020 '<|end|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: layer   0 assigned to device Metal
load_tensors: layer   1 assigned to device Metal
load_tensors: layer   2 assigned to device Metal
load_tensors: layer   3 assigned to device Metal
load_tensors: layer   4 assigned to device Metal
load_tensors: layer   5 assigned to device Metal
load_tensors: layer   6 assigned to device Metal
load_tensors: layer   7 assigned to device Metal
load_tensors: layer   8 assigned to device Metal
load_tensors: layer   9 assigned to device Metal
load_tensors: layer  10 assigned to device Metal
load_tensors: layer  11 assigned to device Metal
load_tensors: layer  12 assigned to device Metal
load_tensors: layer  13 assigned to device Metal
load_tensors: layer  14 assigned to device Metal
load_tensors: layer  15 assigned to device Metal
load_tensors: layer  16 assigned to device Metal
load_tensors: layer  17 assigned to device Metal
load_tensors: layer  18 assigned to device Metal
load_tensors: layer  19 assigned to device Metal
load_tensors: layer  20 assigned to device Metal
load_tensors: layer  21 assigned to device Metal
load_tensors: layer  22 assigned to device Metal
load_tensors: layer  23 assigned to device Metal
load_tensors: layer  24 assigned to device Metal
load_tensors: layer  25 assigned to device Metal
load_tensors: layer  26 assigned to device Metal
load_tensors: layer  27 assigned to device Metal
load_tensors: layer  28 assigned to device Metal
load_tensors: layer  29 assigned to device Metal
load_tensors: layer  30 assigned to device Metal
load_tensors: layer  31 assigned to device Metal
load_tensors: layer  32 assigned to device Metal
load_tensors: tensor 'token_embd.weight' (q8_0) (and 0 others) cannot be used with preferred buffer type CPU_AARCH64, using CPU instead
ggml_backend_metal_log_allocated_size: allocated buffer, size =  2510.53 MiB, ( 5021.52 /  5461.34)
load_tensors: offloading 32 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 33/33 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   622.76 MiB
load_tensors: Metal_Mapped model buffer size =  2510.53 MiB
Using 5 threads
llama_init_from_model: n_seq_max     = 1
llama_init_from_model: n_ctx         = 8192
llama_init_from_model: n_ctx_per_seq = 8192
llama_init_from_model: n_batch       = 8192
llama_init_from_model: n_ubatch      = 512
llama_init_from_model: flash_attn    = 1
llama_init_from_model: freq_base     = 10000.0
llama_init_from_model: freq_scale    = 1
llama_init_from_model: n_ctx_per_seq (8192) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
ggml_metal_init: allocating
ggml_metal_init: picking default device: Apple A18 Pro GPU
ggml_metal_init: using embedded metal library
Compiler failed with XPC_ERROR_CONNECTION_INTERRUPTED
Compiler failed with XPC_ERROR_CONNECTION_INTERRUPTED
Compiler failed with XPC_ERROR_CONNECTION_INTERRUPTED
MTLCompiler: Compilation failed with XPC_ERROR_CONNECTION_INTERRUPTED on 3 try
ggml_metal_init: error: Error Domain=MTLLibraryErrorDomain Code=3 "Compiler encountered an internal error" UserInfo={NSLocalizedDescription=Compiler encountered an internal error}
ggml_backend_metal_device_init: error: failed to allocate context
llama_init_from_model: failed to initialize Metal backend
Could not load context!
```


### Adding the model to the llama.swiftui project
```console
$ git diff
diff --git a/examples/llama.swiftui/llama.swiftui/Models/LlamaState.swift b/examples/llama.swiftui/llama.swiftui/Models/LlamaState.swift
index b8f6a31d..95e43b91 100644
--- a/examples/llama.swiftui/llama.swiftui/Models/LlamaState.swift
+++ b/examples/llama.swiftui/llama.swiftui/Models/LlamaState.swift
@@ -98,6 +98,11 @@ class LlamaState: ObservableObject {
             name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
             url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
             filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
+        ),
+        Model(
+            name: "microsoft_Phi-4-mini-instruct-GGUF  (Q4_K_M, 2.53 GiB)",
+            url: "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_L.gguf?download=true",
+            filename: "microsoft_Phi-4-mini-instruct-Q4_K_L.gguf", status: "download"
         )
     ]
     func loadModel(modelUrl: URL?) throws {
```
