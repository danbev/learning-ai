## Vitis AI
Is AMD's development pratform for their NPUs, the XDNA architecture found inside
Ryzen AI CPUs.

Similar to the Hailo work which is an NPU which uses per-compiled .hef models,
AMD's NPU also requires pre-compiled models. So the actual model is compiled
just like th Hailo models and are availble on HF. So we are not going to be
building up the model with ggml operations.

Xilinx was aquired by AMD in 2023, and the Vitis AI toolchain was rebranded as the
AMD AI toolchain. Xilinx was a FPGA company with products like Zynq and Versal.

### Xilinx runtime (xrt)
This is a low level hardware runtime for interacting with AMD/Ailinx AI accelerator
hardware. This handles device managment, memory allocations, and kernel scheduling
on the NPU/FPGA.

### FlexmlRT
FlexML is AMDs machine learning compiler which takes a neural network model and
quantizes and compiles it into a binary format that runs on the NPU. So for
whisper.cpp I think this would be used by the encoder to offload this to the
NPU and then the decoder can run on the GPU.

And the FlexmlRT is the runtime that loads compiled model. The models are
distributed in .rai (Runtime AI) files.


### Whisper.cpp PR review
PR: https://github.com/ggml-org/whisper.cpp/pull/3608

To build I used the following script (need to add this to CMakeUserPresets):
```console
#!/bin/bash

set -e

build_dir="build-vitisai"

source /opt/xilinx/xrt/setup.sh
source ../flexml/flexmlrt/setup.sh

WHISPER_HIPBLAS=1 cmake -B ${build_dir} \
    -DAMDGPU_TARGETS=gfx1151 \
    -DCMAKE_PREFIX_PATH=/opt/rocm-7.1.0 \
    -DCMAKE_HIP_COMPILER=/opt/rocm-7.1.0/lib/llvm/bin/clang++ \
    -DWHISPER_VITISAI=ON

cmake --build ${build_dir} --config Release -j$(nproc)
```
Running this (first setting LD_LIBRARY_PATH) I get:
```console
whisper_init_state: kv pad  size  =    3.15 MB
whisper_vitisai_init: Exception during Vitis AI runner creation: ERROR: Cannot open library libflexmlrt.so: libpython3.12.so.1.0: cannot open shared object file: No such file or directory.

whisper_init_state: failed to load Vitis AI model from 'models/ggml-base-encoder-vitisai.rai'
error: failed to initialize whisper context

```
I needed to install:
```console
ldd /home/danbev/work/ai/flexml/flexmlrt/lib/libflexmlrt.so
	linux-vdso.so.1 (0x00007dbf02806000)
	libpython3.12.so.1.0 => not found
	libxrt_coreutil.so.2 => /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 (0x00007dbf01600000)
	libxrt_core.so.2 => /opt/xilinx/xrt/lib/libxrt_core.so.2 (0x00007dbf0149a000)
	libm.so.6 => /usr/lib/x86_64-linux-gnu/libm.so.6 (0x00007dbf026de000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007dbf01200000)
	libgcc_s.so.1 => /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007dbf026af000)
	libc.so.6 => /usr/lib/x86_64-linux-gnu/libc.so.6 (0x00007dbf00e00000)
	/lib64/ld-linux-x86-64.so.2 (0x00007dbf02808000)
	libuuid.so.1 => /lib/x86_64-linux-gnu/libuuid.so.1 (0x00007dbf026a5000)
```
So this required libpython3.12 but I don't have that on my machine I currently
only have python 3.13.
Installing python 3.12:
```console
gmktec $ curl https://pyenv.run | bash
gmktec $ export PYENV_ROOT="$HOME/.pyenv"
gmktec $ export PATH="$PYENV_ROOT/bin:$PATH"

gmktec $ PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.12
```
And this should install python 3.12 into:
```console
gmktec $ ls ~/.pyenv/versions/3.12.13/lib/
libpython3.12.so  libpython3.12.so.1.0  libpython3.so  pkgconfig  python3.12
```
And adding that to the LD_LIBRARY_PATH enabled me to be able to run whisper-cli:
```console
#!/bin/bash

set -e

build_dir=build-vitisai
cmd=whisper-cli
audio=samples/jfk.wav

cmake --build ${build_dir} --target $cmd

model=models/ggml-base.bin
#ggml-base-encoder-vitisai.rai

source /opt/xilinx/xrt/setup.sh
source ../flexml/flexmlrt/setup.sh

# libflexmlrt.so depends on libpython3.12.so.1.0 which is not in Ubuntu 25.10 repos.
# Install via: PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.12
PYTHON_3_12_LIB="${HOME}/.pyenv/versions/3.12.13/lib"
export LD_LIBRARY_PATH="${PYTHON_3_12_LIB}:${LD_LIBRARY_PATH:-}"

#ldd /home/danbev/work/ai/flexml/flexmlrt/lib/libflexmlrt.so

${build_dir}/bin/whisper-cli \
    -m ${model} \
    -f ${audio} \
    --language auto
```

```console
gmktec $ ./run-cli.sh
...

whisper_init_state: Vitis AI encoder model loaded
whisper_init_state: compute buffer (conv)   =    5.62 MB
whisper_init_state: compute buffer (cross)  =    4.66 MB
whisper_init_state: compute buffer (decode) =   96.37 MB
read_audio_data: reading audio data from 'samples/jfk.wav' ...
read_audio_data: trying to decode with miniaudio

system_info: n_threads = 4 / 32 | WHISPER : VITISAI = 1 | COREML = 0 | OPENVINO = 0 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | OPENMP = 1 | REPACK = 1 | 

main: processing 'samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 processors, 5 beams + best of 5, lang = auto, task = transcribe, timestamps = 1 ...

whisper_full_with_state: auto-detected language: en (p = 0.961673)

[00:00:00.000 --> 00:00:08.000]   And so, my fellow Americans, ask not what your country can do for you,
[00:00:08.000 --> 00:00:11.000]   ask what you can do for your country.

whisper_print_timings:     load time =    61.58 ms
whisper_print_timings:     fallbacks =   0 p /   0 h
whisper_print_timings:      mel time =    11.10 ms
whisper_print_timings:   sample time =    38.96 ms /   148 runs (     0.26 ms per run)
whisper_print_timings:   encode time =   245.05 ms /     2 runs (   122.53 ms per run)
whisper_print_timings:   decode time =     2.93 ms /     1 runs (     2.93 ms per run)
whisper_print_timings:   batchd time =   180.86 ms /   146 runs (     1.24 ms per run)
whisper_print_timings:   prompt time =     0.00 ms /     1 runs (     0.00 ms per run)
whisper_print_timings:    total time =   757.37 ms
```
