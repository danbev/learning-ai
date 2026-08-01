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
* [Llama.cpp](./notes/llama.md)
* [Model Formats](./notes/model-formats.md)
* [Quantization](./notes/quantization.md)
* [GGUF / GPTQ / AWQ](./notes/gptq.md) / [AWQ](./notes/awq.md)
* [LoRA / QLoRA](./notes/lora.md) / [QLoRA](./notes/qlora.md)

### Attention & Embeddings

* [Attention](./notes/attention.md)
* [Position Embeddings](./notes/position-embeddings/)
* [Tokenization](./notes/tokenization/README.md)
* [Word Embeddings](./notes/word-embeddings.md)
* [Normalization](./notes/normalization.md)
* [Softmax / Logits](./notes/softmax.md) / [Logits](./notes/logits.md)
* [Residual Connections](./notes/residual-connections.md)
* [Activation Functions](./notes/activation-functions.md)
* [Loss Functions](./notes/loss-functions.md)
* [One-Hot Encoding](./notes/one-hot-encoding.md)
* [Control Vectors](./notes/control-vectors.md)

### Inference & Decoding

* [Sampling](./notes/sampling.md)
* [Speculative Decoding](./notes/speculative-decoding/speculative-decoding.md)
* [Infill](./notes/infill.md)
* [Grammars](./notes/grammars.md)
* [llguidance](./notes/llguidance.md)
* [ChatML / Chat Templates](./notes/chatml.md) / [Chat Templates](./notes/chat-templates.md)

### Training & Fine-tuning

* [Fine-tuning](./notes/fine-tuning.md)
* [DPO](./notes/dpo.md)
* [Reinforcement Learning](./notes/reinforcement-learning.md)
* [Optimization Algorithms](./notes/optimization-algorithms.md)
* [LBFGS](./notes/lbfgs.md)
* [Linear Regression](./notes/linear-regression.md)
* [Markov Chains](./notes/markov-chains.md)
* [Flow Matching](./notes/flow-matching.md)

### Hardware & Acceleration

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
* [VAD](./notes/vad.md)

### Vision & Multimodal

* [CLIP](./notes/vision/clip.md)
* [ViT](./notes/vision/vit.md)
* [LLaVA](./notes/vision/llava.md)
* [Image Preprocessing](./notes/vision/image-preprocessing.md)

### Agents & Applications

* [Agents Overview](./notes/agents/)
* [RAG](./notes/rag.md)
* [LangChain](./notes/langchain.md)
* [LLM Chain](./notes/llm-chain.md)
* [MRKL](./notes/mrkl.md)
* [ReAct](./notes/react.md)
* [MCP](./notes/mcp.md)

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

### Audio Code

Audio processing and speech-to-text.

| Project | Description |
|---------|-------------|
| [Silero VAD](./audio/silero-vad/) | Silero Voice Activity Detection |
| [Whisper.cpp](./audio/whisper.cpp/) | Whisper.cpp submodule |

---
---

## Notes Index

For a complete list of all notes, see the [notes](./notes/) directory.
