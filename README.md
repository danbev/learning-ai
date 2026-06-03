# Learning AI

This repository contains notes and code examples related to AI/ML, with a focus on
understanding the fundamentals of large language models, inference engines, and hardware acceleration.

## In-progress

* [Hailo](./npu/hailo) Hailo-10H AI accelerator (NPU), Raspberry Pi AI HAT+
* [Parakeet](./notes/whisper/parakeet.md) — Supporting Parakeet in whisper.cpp
* [Kimi-Linear](./notes/kimi-linear.md)
* [CUDA FA exploration](./notes/flash-attention.md)

---

## Table of Contents

### Notes
* [Model Architectures](#model-architectures)
* [Model Formats & Quantization](#model-formats--quantization)
* [Attention & Embeddings](#attention--embeddings)
* [Inference & Decoding](#inference--decoding)
* [Training & Fine-tuning](#training--fine-tuning)
* [Hardware & Acceleration](#hardware--acceleration)
* [Audio & Speech](#audio--speech)
* [Vision & Multimodal](#vision--multimodal)
* [Agents & Applications](#agents--applications)
* [Miscellaneous Topics](#miscellaneous-topics)

### Code
* [Fundamentals](#fundamentals)
* [GPU Code](#gpu-code)
* [NPU Code](#npu-code)
* [Vector Databases](#vector-databases)
* [Embeddings](#embeddings-1)
* [Audio Code](#audio-code)
* [Agents Code](#agents-code)

---

## Notes

### Model Architectures

* [Architectures Overview](./notes/architectures/README.md)
  * [Transformers](./notes/architectures/transformers.md)
  * [RNN](./notes/architectures/rnn.md)
  * [LSTM](./notes/architectures/lstm.md)
  * [CNN](./notes/architectures/cnn.md)
  * [State Space Models (SSM)](./notes/architectures/state-space-models.md)
  * [Mamba / Mamba-2](./notes/architectures/mamba.md)
  * [RWKV](./notes/architectures/rwkv.md)
  * [DeltaNet](./notes/architectures/delta-net.md)
  * [Linear Attention](./notes/architectures/linear-attention.md)
  * [MatFormer](./notes/architectures/matformer.md)
* [Mixture of Experts (MoE)](./notes/moe.md)
* [KAN](./notes/kan.md)
* [BitNet](./notes/bitnet.md)
* [Autoencoder](./notes/autoencoder.md)

### Model Formats & Quantization

* [GGML](./notes/ggml.md)
  * [GGML SIMD](./notes/ggml/ggml-simd.md)
  * [GGML Quantization](./notes/ggml/ggml-quants.md)
  * [GGML AArch64](./notes/ggml/ggml-aarch64.md)
  * [GGML Backend](./notes/ggml/cpu-backend.md)
  * [GGML CUDA](./notes/ggml/cuda.md)
  * [GGML WebGPU](./notes/ggml/webgpu.md)
  * [GGML OpenVINO Backend](./notes/ggml/openvino-backend.md)
  * [GGML Meta Backend](./notes/ggml/meta-backend.md)
  * [GGML Backend Scheduling](./notes/ggml/ggml-backend-sched.md)
  * [GGML Repack](./notes/ggml/repack.md)
  * [GGML Quantization Notes](./notes/ggml/quantization.md)
  * [GGML SSM Scan](./notes/ggml/ggml-ssm-scan.md)
  * [GGML Split Graph](./notes/ggml/ggml-split-graph.md)
  * [GGML Optimizers](./notes/ggml/ggml-optimizers.md)
  * [GGML Argsort](./notes/ggml/argsort.md)
  * [GGML ARM i8mm Issue](./notes/ggml/arm-i8mm-issue.md)
  * [GGML Version](./notes/ggml/version.md)
* [Llama.cpp](./notes/llama.md)
  * [llama.cpp Main](./notes/llama-main.md)
  * [llama.cpp Buildx](./notes/llama.cpp-buildx.md)
  * [llama.cpp Tasks](./notes/llama.cpp-tasks.md)
  * [llama.cpp Logging](./notes/llama-logging.md)
  * [llama.cpp Memory](./notes/llama-memory.md)
  * [llama.cpp Python Notes](./notes/llama-python-notes.md)
  * [llama.cpp CUDA](./notes/llama.cpp/cuda.md)
  * [llama.cpp KV Cache](./notes/llama.cpp/kv-cache.md)
  * [llama.cpp Quantization](./notes/llama.cpp/quantization.md)
  * [llama.cpp GPU Sampling](./notes/llama.cpp/gpu-sampling.md)
  * [llama.cpp Tensor Parallelism](./notes/llama.cpp/tensor-parallelism.md)
  * [llama.cpp Server](./notes/llama.cpp/llama-server.md)
  * [llama.cpp WebUI](./notes/llama.cpp/webui.md)
  * [llama.cpp MTMD](./notes/llama.cpp/mtmd.md)
  * [llama.cpp TTS](./notes/llama.cpp/tts.md)
  * [llama.cpp Embedding Gemma](./notes/llama.cpp/embedding-gemma-dense.md)
  * [llama.cpp LLaMA 3.2 Vision](./notes/llama.cpp/llama-3-2-vision.md)
  * [llama.cpp GPT-OSS](./notes/llama.cpp/gpt-oss.md)
  * [llama.cpp Convert](./notes/llama.cpp/convert.md)
  * [llama.cpp Convert Dequantize](./notes/llama.cpp/convert-dequantize.md)
  * [llama.cpp Debugging](./notes/llama.cpp/debugging.md)
  * [llama.cpp Packaging](./notes/llama.cpp/packaging.md)
  * [llama.cpp Tests](./notes/llama.cpp/tests.md)
  * [llama.cpp macOS](./notes/llama.cpp/macosx.md)
  * [llama.cpp HTTPS](./notes/llama.cpp/https.md)
  * [llama.cpp Backend Sampling](./notes/llama.cpp/backend-sampling-state.md)
  * [llama.cpp KV Cache Notes](./notes/llama-kv-cache.md)
* [Model Formats](./notes/model-formats.md)
* [Quantization](./notes/quantization.md)
* [GGUF / GPTQ / AWQ](./notes/gptq.md) / [AWQ](./notes/awq.md)
* [LoRA / QLoRA](./notes/lora.md) / [QLoRA](./notes/qlora.md)
* [iMatrix](./notes/imatrix.md)
* [Ollama](./notes/ollama.md)
* [Huggingface](./notes/huggingface.md)

### Attention & Embeddings

* [Attention](./notes/attention.md)
* [Attention Sink](./notes/attention-sink.md)
* [Flash Attention](./notes/flash-attention.md)
* [Sage Attention](./notes/sage-attention.md)
* [Ring Attention](./notes/ringattention.md)
* [MLA](./notes/mla.md)
* [Position Embeddings](./notes/position-embeddings/)
  * [RoPE](./notes/position-embeddings/rope.md)
  * [ALiBi](./notes/position-embeddings/alibi.md)
  * [Positional Encoding](./notes/position-embeddings/positional-encoding.md)
  * [YARN](./notes/position-embeddings/yarn.md)
  * [XPOS](./notes/position-embeddings/xpos.md)
  * [LongRoPE](./notes/position-embeddings/longrope.md)
  * [P-ROPE](./notes/position-embeddings/p-rope.md)
  * [MRL](./notes/position-embeddings/mrl.md)
  * [PLE](./notes/position-embeddings/ple.md)
  * [GGML RoPE](./notes/position-embeddings/ggml-rope.md)
  * [Embeddings](./notes/position-embeddings/embeddings.md)
* [Tokenization](./notes/tokenization/README.md)
  * [BPE](./notes/tokenization/bpe.md)
  * [WordPiece](./notes/tokenization/wordpiece.md)
  * [SentencePiece](./notes/tokenization/sentencepiece.md)
  * [Unigram](./notes/tokenization/unigram.md)
  * [RWKV](./notes/tokenization/rwkv.md)
  * [Tiktoken](./notes/tokenization/tiktoken.md)
* [Word Embeddings](./notes/word-embeddings.md)
* [Normalization](./notes/normalization.md)
* [Softmax / Logits](./notes/softmax.md) / [Logits](./notes/logits.md)
* [Residual Connections](./notes/residual-connections.md)
* [Activation Functions](./notes/activation-functions.md)
* [Loss Functions](./notes/loss-functions.md)
* [Exp](./notes/exp.md)
* [One-Hot Encoding](./notes/one-hot-encoding.md)
* [Control Vectors](./notes/control-vectors.md)
* [GRITLM](./notes/gritlm.md)

### Inference & Decoding

* [Sampling](./notes/sampling.md)
  * [LLaMA Sampling](./notes/llama-sampling.md)
* [Speculative Decoding](./notes/speculative-decoding/speculative-decoding.md)
  * [Medusa](./notes/speculative-decoding/medusa.md)
  * [Eagle](./notes/speculative-decoding/eagle.md)
  * [Self-Speculative](./notes/speculative-decoding/self-speculative.md)
  * [D-Flash](./notes/speculative-decoding/d-flash.md)
* [Continuous Batching](./notes/continous-batching.md)
* [Tensor Parallelism](./notes/tensor-parallelism.md)
* [Pipeline Parallelism](./notes/pipeline-parallelism.md)
* [LLaMA Self-Extend](./notes/llama-self-extend.md)
* [LLaMA Batch Embedding](./notes/llama-batch-embd.md)
* [Perplexity](./notes/perplexity.md)
* [Likelihood](./notes/likelihood.md)
* [Infill](./notes/infill.md)
* [Grammars](./notes/grammars.md)
* [llguidance](./notes/llguidance.md)
* [ChatML / Chat Templates](./notes/chatml.md) / [Chat Templates](./notes/chat-templates.md)
* [Prompt Engineering](./notes/prompt-engineering.md)

### Training & Fine-tuning

* [Fine-tuning](./notes/fine-tuning.md)
* [DPO](./notes/dpo.md)
* [Reinforcement Learning](./notes/reinforcement-learning.md)
* [Optimization Algorithms](./notes/optimization-algorithms.md)
* [LBFGS](./notes/lbfgs.md)
* [Linear Regression](./notes/linear-regression.md)
* [Markov Chains](./notes/markov-chains.md)
* [XOR Problem](./notes/xor-problem.md)
* [Flow Matching](./notes/flow-matching.md)
* [Generative Deep Learning](./notes/generative-deep-learning.md)

### Hardware & Acceleration

#### CPU
* [SIMD](./notes/simd.md)
* [SVE](./notes/sve.md)
* [NEON](./notes/neon.md)
* [AMX](./notes/amx.md)
* [VNNI](./notes/vnni.md)
* [ARM](./notes/arm.md)
* [Numa](./notes/numa.md)
* [BLAS](./notes/blas.md)
* [KleidiAI](./notes/kleidiai.md)

#### GPU
* [CUDA](./notes/cuda.md)
* [ROCm](./notes/rocm.md)
* [Metal](./notes/metal.md)
* [Vulkan](./notes/vulkan.md)
* [OpenCL](./notes/opencl.md)
* [WebGPU](./notes/webgpu.md)
* [WebNN](./notes/webnn.md)
* [MUSA](./notes/musa.md)
* [eGPU](./notes/egpu.md)
* [Mesa](./notes/mesa.md)
* [HIP](./notes/hip.md)
* [NCCL](./notes/nccl.md)
* [GPU Overview](./notes/gpu.md)

#### NPU / Other
* [Hailo](./notes/hailo.md)
* [OpenVINO](./notes/openvino.md)
* [SYCL](./notes/sycl.md)
* [CANN](./notes/cann.md)
* [CoreML](./notes/coreml.md)
* [WASM / WASI-NN](./notes/wasm.md) / [WASI-NN](./notes/wasi-nn.md)

### Audio & Speech

* [Whisper](./notes/whisper.md)
  * [Parakeet](./notes/whisper/parakeet.md)
  * [Parakeet Encoder](./notes/whisper/parakeet-encoder.md)
  * [Parakeet Decoder](./notes/whisper/parakeet-decoder.md)
  * [Parakeet Preprocessing](./notes/whisper/parakeet-preprocessing.md)
  * [Parakeet FFmpeg](./notes/whisper/parakeet-ffmpeg.md)
  * [Processing Notes](./notes/whisper/processing-notes.md)
  * [Token-level Timestamps](./notes/whisper/token-level-timestamps.md)
  * [VAD Segments](./notes/whisper/vad-segments-repeat-issue.md)
  * [Seek](./notes/whisper/seek.md)
  * [Android](./notes/whisper/android.md)
  * [WASM](./notes/whisper-wasm.md)
  * [Ruby](./notes/whisper-ruby.md)
* [VAD](./notes/vad.md)
* [Audio Notes](./notes/audio/)
  * [Conformer](./notes/audio/conformer.md)
  * [DTW](./notes/audio/dtw.md)
  * [Mel](./notes/audio/mel.md)
  * [LRC](./notes/audio/lrc.md)
  * [SRT](./notes/audio/srt.md)
  * [VTT](./notes/audio/vtt.md)
  * [SDL2](./notes/audio/sdl2.md)
  * [Whisper Stream](./notes/audio/whisper-stream.md)

### Vision & Multimodal

* [CLIP](./notes/vision/clip.md)
* [ViT](./notes/vision/vit.md)
* [LLaVA](./notes/vision/llava.md)
* [LLaMA Vision 3.2](./notes/vision/llama-vision-3-2-instruct.md)
* [Granite Vision](./notes/vision/granite.md)
* [JEPA](./notes/vision/jepa.md)
* [Image Preprocessing](./notes/vision/image-preprocessing.md)
* [CLIP Search](./notes/clip-search.md)
* [BLIP-2](./notes/blip2.md)
* [Mobile VLM](./notes/mobile-vlm.md)
* [LLaVA+](./notes/llava-plus.md)

### Agents & Applications

* [Agents Overview](./notes/agents/)
  * [Claude Code](./notes/agents/claude-code.md)
  * [Claude Code Router](./notes/agents/claude-code-router.md)
  * [Code Agents](./notes/agents/code-agents.md)
  * [Goose](./notes/agents/goose.md)
  * [OpenCode](./notes/agents/opencode.md)
* [RAG](./notes/rag.md)
* [LangChain](./notes/langchain.md)
* [LLM Chain](./notes/llm-chain.md)
* [MRKL](./notes/mrkl.md)
* [ReAct](./notes/react.md)
* [MCP](./notes/mcp.md)
* [Vector Databases](./notes/vector-databases.md)

### Miscellaneous Topics

* [LLM Overview](./notes/llm.md)
* [Diffusion / Stable Diffusion](./notes/diffusion.md) / [Stable Diffusion](./notes/stable-diffusion.md)
* [Apache Arrow](./notes/apache-arrow.md)
* [ONNX](./notes/onnx.md)
* [PyTorch](./notes/pytorch.md)
* [vLLM](./notes/vllm.md)
* [TRT-LLM](./notes/trt-llm.md)
* [Mistral](./notes/mistral.md)
* [Bloom](./notes/bloom.md)
* [Granite Model](./notes/granite-model.md)
* [Mod](./notes/mod.md)
* [Minja](./notes/minja.md)
* [Trie](./notes/trie.md)
* [Symbols](./notes/symbols.md)
* [Variables](./notes/variables.md)
* [Count-based](./notes/count-based.md)
* [Background](./notes/background.md)
* [Security](./notes/security.md)
* [Memory](./notes/memory.md)
* [Android](./notes/android.md)
* [Colab](./notes/colab.md)
* [Groq](./notes/groq.md)
* [ROC](./notes/roc.md)
* [zDNN](./notes/zDNN.md)
* [Spark](./notes/spark.md)
* [Copilot](./notes/copilot.md)

---

## Code

### Fundamentals

Exploration code for core AI/ML concepts, libraries, and frameworks.

| Project | Description |
|---------|-------------|
| [GGML](./fundamentals/ggml/README.md) | GGML C++ library exploration |
| [Llama.cpp](./fundamentals/llama.cpp/README.md) | Llama.cpp library exploration (inference, finetuning) |
| [Python](./fundamentals/python/README.md) | Python ML examples |
| [Rust](./fundamentals/rust/README.md) | Rust ML examples (llm-chains, tch-rs, etc.) |
| [vLLM](./fundamentals/vllm/README.md) | vLLM exploration |
| [OpenVINO](./fundamentals/openvino/README.md) | OpenVINO Python examples |
| [OpenVINO C++](./fundamentals/openvino-cpp/) | OpenVINO C++ examples |
| [PyTorch](./fundamentals/pytorch/) | PyTorch & pybind examples |
| [SIMD](./fundamentals/simd/README.md) | SIMD instruction exploration |
| [SIMD Assembly](./fundamentals/simd-assembly/README.md) | Low-level SIMD assembly |
| [SVE](./fundamentals/sve/README.md) | ARM SVE exploration |
| [NEON](./fundamentals/neon/) | ARM NEON examples |
| [AMX](./fundamentals/amx/README.md) | Intel AMX exploration |
| [VNNI](./fundamentals/vnni/README.md) | VNNI instruction exploration |
| [BLAS](./fundamentals/blas/openblas/README.md) | OpenBLAS exploration |
| [ROCm](./fundamentals/rocm/README.md) | AMD ROCm examples |
| [SYCL](./fundamentals/sycl/README.md) | SYCL examples |
| [KleidiAI](./fundamentals/kleidiai/) | KleidiAI examples |
| [Grammars](./fundamentals/grammars/llguidance/README.md) | LLaGuidance grammar exploration |
| [Tokenization](./fundamentals/tokenization/) | Tokenization examples |
| [Data Structures](./fundamentals/datastructures/README.md) | ML-relevant data structures |
| [Image Processing](./fundamentals/image-processing/) | Image processing examples |
| [JavaScript](./fundamentals/javascript/) | TensorFlow.js examples |
| [WASM](./fundamentals/wasm/wasi-nn-example/README.md) | WebAssembly NN examples |
| [Whisper](./fundamentals/whisper/) | Whisper.cpp exploration |
| [Templates](./fundamentals/templates/minja/) | Minja template engine |

### GPU Code

GPU compute exploration across multiple APIs.

| Project | Description |
|---------|-------------|
| [CUDA](./gpu/cuda/README.md) | CUDA examples in C++ |
| [OpenCL](./gpu/opencl/README.md) | OpenCL examples |
| [Vulkan](./gpu/vulkan/README.md) | Vulkan examples |
| [Kompute](./gpu/kompute/README.md) | Kompute (Vulkan compute) examples |
| [Metal](./gpu/metal/) | Metal examples |
| [ROCm](./gpu/rocm/README.md) | AMD ROCm/HIP examples |
| [WebGPU](./gpu/webgpu/README.md) | WebGPU examples |
| [XRT](./gpu/xrt/) | XRT examples |

### NPU Code

Neural Processing Unit exploration (Hailo).

| Project | Description |
|---------|-------------|
| [Hailo](./npu/hailo/README.md) | Hailo-10H AI accelerator, Raspberry Pi AI HAT+ |

### Vector Databases

Vector database examples and exploration.

| Project | Description |
|---------|-------------|
| [Qdrant](./vector-databases/qdrant/) | Qdrant examples (Python, Rust) |
| [LanceDB](./vector-databases/lancedb/) | LanceDB examples (Python, Rust) |

### Embeddings

Word and sentence embedding examples.

| Project | Description |
|---------|-------------|
| [Rust](./embeddings/rust/) | Embeddings examples in Rust |

### Audio Code

Audio processing and speech-to-text.

| Project | Description |
|---------|-------------|
| [Silero VAD](./audio/silero-vad/) | Silero Voice Activity Detection |
| [Whisper.cpp](./audio/whisper.cpp/) | Whisper.cpp submodule |

### Agents Code

AI agent frameworks and examples.

| Project | Description |
|---------|-------------|
| [llama-cpp-agent](./agents/llama-cpp-agent/README.md) | AI agent using llama.cpp |

---

## Huggingface API

| Language | Description |
|----------|-------------|
| [Python](./hugging-face/python/) | Huggingface API example |
| [Rust](./hugging-face/rust/) | Candle example |

---

## Notes Index

For a complete list of all notes, see the [notes](./notes/) directory.
