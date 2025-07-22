## Claude Code
This is a terminal based coding agent

### Installation
```console
$ npm install -g @anthropic-ai/claude-code
```
This will install the command line tool named `claude`:
```console
$ claude --help
```

### Usage
We can simple start claude in a project directory and then start using it.

It is also possible to use `-p/--print` to have an non-interactive mode:
```console
$ echo "Give me an overview of this project" | claude -p
This is a learning repository focused on AI/ML fundamentals and implementation details. It contains comprehensive notes and exploration code covering:

**Key areas:**
- **Core Libraries**: GGML, Llama.cpp, whisper.cpp exploration
- **GPU Computing**: CUDA, Vulkan, Metal, OpenCL examples
- **ML Fundamentals**: Tokenization, architectures, embeddings, vector databases
- **Audio Processing**: Voice Activity Detection, Whisper integration
- **Vision**: Computer vision topics and implementations

The project emphasizes hands-on learning through C/C++, Python, and Rust code examples across different ML frameworks and hardware acceleration platforms.
```
The same can also be accomplished with the `claude` command:
```console
$ claude -p "Give me an overview of this project"
Based on the project structure, here's an overview:

## Learning AI Repository

This is a comprehensive AI/ML learning and exploration repository focused on low-level AI fundamentals and practical implementations. The project covers:

### Main Areas:

**1. Core AI Fundamentals**
- **GGML** - Deep exploration of the GGML tensor library (C code)
- **Llama.cpp** - Extensive work with the llama.cpp inference framework
- **Tokenization** - Various tokenization algorithms (BPE, SentencePiece, etc.)
- **Neural Network Architectures** - CNNs, RNNs, Transformers, Mamba, etc.

**2. GPU Computing**
- Examples across multiple GPU frameworks: CUDA, ROCm, OpenCL, Vulkan, Metal, WebGPU
- SIMD and low-level optimization code
- Cross-platform GPU development

**3. Audio Processing**
- **Whisper.cpp** integration and exploration
- **Voice Activity Detection (VAD)** using Silero VAD
- Audio sample processing and containers

**4. Practical AI Applications**
- **Agents** - LLM-based agent implementations using Rust/WASM
- **Embeddings** - Vector embeddings in Python and Rust
- **Vector Databases** - Qdrant and LanceDB examples
- **Hugging Face** integration

**5. Programming Languages**
- Multi-language approach: **C/C++**, **Rust**, **Python**, **JavaScript**
- Focus on systems programming for AI/ML

### Current Focus Areas:
- Model Context Protocol (MCP)
- Voice Activity Detection for whisper.cpp
- GPU acceleration techniques

The repository appears to be a personal learning journey through AI fundamentals, emphasizing hands-on implementation and low-level understanding rather than high-level frameworks.
```
Interesting that the output is different but that is also for LLMs.


### Init
The `init` command will generate a file named `CLAUDE.md` with information about
the project. This will be read by clause to get context about the project.

### Using images
For this example I copied notes/images/spectrogram.png and used the following
prompt:
```console
> [Image #1] Can you tell me what this images shows?

● This image shows a spectrogram of an audio signal labeled "Audio 1 #1". A spectrogram is a visual representation of audio that
  displays:

  - X-axis: Time progression (from left to right)
  - Y-axis: Frequency (from low frequencies at bottom to high frequencies at top, ranging from about 100Hz to 19000Hz)
  - Colors: Intensity/amplitude of different frequencies at different times (brighter colors like red/pink indicate stronger
  signal, darker colors indicate weaker signal)

  In this particular spectrogram, I can see:

  1. Two distinct speech segments - there are two main areas of activity with a gap of silence between them
  2. Typical speech patterns - the horizontal striped patterns in red/pink are characteristic of human speech, showing the
  harmonic structure of voiced sounds
  3. Frequency distribution - most of the energy is concentrated in the lower frequencies (below ~8kHz), which is typical for
  human speech
  4. Silence periods - the dark/black areas represent periods with little to no audio activity

  This type of visualization is commonly used in audio processing, speech recognition, and voice activity detection (VAD) systems
   - which aligns well with the audio processing work I saw in your learning-ai repository, particularly the Whisper.cpp and
  Silero VAD components.
```
This can be used for screenshot of error messages, images of code, or any other
image.

### Referencing files
```console
> Can you give me a summary of @notes/ggml/quantization.md

● Read(notes/ggml/quantization.md)
  ⎿  Error: File content (26740 tokens) exceeds maximum allowed tokens (25000). Please use offset and limit parameters to read
     specific portions of the file, or use the GrepTool to search for specific content.

● Read(notes/ggml/quantization.md)
  ⎿  Read 100 lines (ctrl+r to expand)

● Search(pattern: "^##|^###", path: "notes/ggml/quantization.md", output_mode: "content")
  ⎿  Found 27 lines (ctrl+r to expand)

● Read(notes/ggml/quantization.md)
  ⎿  Read 50 lines (ctrl+r to expand)

● Task(Summarize quantization concepts)
  ⎿  Done (6 tool uses · 41.4k tokens · 57.4s)

● Summary of GGML Quantization

  The notes/ggml/quantization.md file provides a comprehensive technical guide to GGML's quantization system. Here are the key
  concepts:

  Purpose & Benefits

  GGML quantization converts high-precision floats (32/16-bit) to lower-precision fixed-point numbers to save memory and
  accelerate computations while maintaining reasonable accuracy.

  Main Quantization Types

  Symmetric vs Asymmetric:
  - Symmetric: Maps zero in original data to zero in quantized form
  - Asymmetric: Allows offset, better for non-zero-centered data

  Block-based Quantization:
  - Divides tensors into blocks (32/64/128 elements)
  - Each block has its own scale factor to prevent outliers from affecting the entire tensor

  Key Data Structures

  Basic Blocks:
  - block_q4_0: 4-bit symmetric quantization (18 bytes per 32 values)
  - block_q4_1: 4-bit asymmetric with offset (20 bytes per 32 values)
  - block_q5_0/1: 5-bit quantization with higher precision
  - block_q8_0/1: 8-bit quantization for higher quality

  Super-blocks (K-variants):
  - block_q4_K: Stores 8 blocks efficiently, quantizes the scales/mins themselves (~10% space savings)
  - block_q2_K: Extreme 2-bit compression with 16 sub-blocks

  Mathematical Framework

  - Scale factor: (r_max - r_min) / (q_max - q_min) - maps float range to quantized range
  - Quantization: round((float_value - r_min) / scale)
  - Dequantization: scale * quantized_value + offset
  - Weighted least squares: Used in K-variants for optimal parameter fitting

  Advanced Features

  - Ternary Quantization (TQ1_0): Uses only 3 states (-1,0,1), achieving ~1.6 bits per weight
  - Sophisticated bit packing: Nibble packing, high-bit arrays, complex storage schemes
  - Type traits system: Abstraction layer for different quantization methods
  - Optimization techniques: Inverse deltas, rounding offsets, restrict pointers

  The document demonstrates how GGML achieves significant compression (e.g., 32-bit floats → 4-bit integers) while maintaining
  model performance through careful mathematical design and efficient implementation strategies.

```
This supports tab completion so just use tab to complete the file name.

### Copy error log
This can be done by simply pasting into the terminal (I use `Ctrl+Shift+V` in
in terminator on Linux but it would be `Ctrl+V` in most terminals):
```console
> Can you explain this error: [Pasted text #1 +19 lines]
```

### MCP server
```console
$ claude mcp --help
Usage: claude mcp [options] [command]

Configure and manage MCP servers

Options:
  -h, --help                                     Display help for command

Commands:
  serve [options]                                Start the Claude Code MCP server
  add [options] <name> <commandOrUrl> [args...]  Add a server
  remove [options] <name>                        Remove an MCP server
  list                                           List configured MCP servers
  get <name>                                     Get details about an MCP server
  add-json [options] <name> <json>               Add an MCP server (stdio or SSE) with a JSON string
  add-from-claude-desktop [options]              Import MCP servers from Claude Desktop (Mac and WSL only)
  reset-project-choices                          Reset all approved and rejected project-scoped (.mcp.json) servers within this
                                                 project
  help [command]                                 display help for command
```
So I have configured an MCP server for whisper.cpp and would like to try it
with claude code.
So lets add this (I actually added this using claude) but I want to have the
commands handy:
```console
$ claude mcp add-json whisper '{
    "command": "/home/danbev/work/ai/whisper-work/build/bin/whisper-mcp-server",
    "args": [
      "--model",
      "/home/danbev/work/ai/whisper-work/models/ggml-base.en.bin"
    ]
  }'
```
List the configured MCP servers:
```console
$ claude mcp list
whisper: /home/danbev/work/ai/whisper-work/build/bin/whisper-mcp-server --model /home/danbev/work/ai/whisper-work/models/for-tests-ggml-medium.bin
```
Get information about the whisper MCP server:
```console
$ claude mcp get whisper
whisper:
  Scope: Local (private to you in this project)

To remove this server, run: claude mcp remove "whisper" -s local
```
In an interactive claude session, which is how I added the mcp server, I needed
to restart the session to actually see the MCP server:
```console
 /mcp 
  ⎿  (no content)
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Manage MCP servers                                                                                                                 │
│                                                                                                                                    │
│ ❯ 1. whisper  ✔ connected · Enter to view details                                                                                  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
> /mcp
  ⎿  (no content)
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Whisper MCP Server                                                                                                                 │
│                                                                                                                                    │
│ Status: ✔ connected                                                                                                                │
│ Command: /home/danbev/work/ai/whisper-work/build/bin/whisper-mcp-server                                                            │
│ Args: --model /home/danbev/work/ai/whisper-work/models/ggml-base.en.bin                                                            │
│ Capabilities: tools                                                                                                                │
│ Tools: 2 tools                                                                                                                     │
│                                                                                                                                    │
│ ❯ 1. View tools                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
   Esc to go back

   > /mcp
  ⎿  (no content)
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Tools for whisper (2 tools)                                                                                                        │
│                                                                                                                                    │
│ ❯ 1. transcribe                                                                                                                    │
│   2. model_info                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
   Esc to go back

> /mcp 
  ⎿  (no content)
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ transcribe (whisper)                                                                                                               │
│                                                                                                                                    │
│ Tool name: transcribe                                                                                                              │
│ Full name: mcp__whisper__transcribe                                                                                                │
│                                                                                                                                    │
│ Description:                                                                                                                       │
│ Transcribe audio file using whisper.cpp                                                                                            │
│                                                                                                                                    │
│ Parameters:                                                                                                                        │
│   • file (required): string - Path to audio file                                                                                   │
│   • language: string - Language code (optional, auto-detect if not specified)                                                      │
│   • translate: boolean - Translate to English                                                                                      │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
   Esc to go back
```
So we should be able to transcribe a file now:
```console
> Can you transcribe @audio/samples/jfk.wav?

● I'll transcribe the audio file for you using the whisper transcription tool.

● whisper - transcribe (MCP)(file: "audio/samples/jfk.wav")

╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Tool use                                                                                                                           │
│                                                                                                                                    │
│   whisper - transcribe(file: "audio/samples/jfk.wav") (MCP)                                                                        │
│   Transcribe audio file using whisper.cpp                                                                                          │
│                                                                                                                                    │
│ Do you want to proceed?                                                                                                            │
│ ❯ 1. Yes                                                                                                                           │
│   2. Yes, and don't ask again for whisper - transcribe commands in /home/danbev/work/ai/learning-ai                                │
│   3. No, and tell Claude what to do differently (esc)                                                                              │
│                                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
Selecting `1. Yes` will produce the following output:
```console
> Can you transcribe @audio/samples/jfk.wav?

● whisper - transcribe (MCP)(file: "audio/samples/jfk.wav")
  ⎿   And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

● The transcription is: "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your
  country."
```
The mcp configuration is stored in `~/.claude.json`.


