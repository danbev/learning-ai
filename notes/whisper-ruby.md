## Whisper Ruby Bindings
This document contains note about the Ruby bindings in Whisper.

### Building
To bulid use the following command:
(I'm including all the ouput just to be able to stop any issues and perhaps find
missing files)
```console
$ rake build
cp ../../src/openvino/whisper-openvino-encoder.cpp ext/src/openvino/whisper-openvino-encoder.cpp
cp ../../src/whisper.cpp ext/src/whisper.cpp
cp ../../src/coreml/whisper-decoder-impl.h ext/src/coreml/whisper-decoder-impl.h
cp ../../src/coreml/whisper-encoder-impl.h ext/src/coreml/whisper-encoder-impl.h
cp ../../src/coreml/whisper-encoder.h ext/src/coreml/whisper-encoder.h
cp ../../src/openvino/whisper-openvino-encoder.h ext/src/openvino/whisper-openvino-encoder.h
cp ../../src/whisper-arch.h ext/src/whisper-arch.h
cp ../../src/coreml/whisper-decoder-impl.m ext/src/coreml/whisper-decoder-impl.m
cp ../../src/coreml/whisper-encoder-impl.m ext/src/coreml/whisper-encoder-impl.m
cp ../../include/whisper.h ext/include/whisper.h
cp ../../ggml/src/ggml-alloc.c ext/ggml/src/ggml-alloc.c
cp ../../ggml/src/ggml-cpu/ggml-cpu-quants.c ext/ggml/src/ggml-cpu/ggml-cpu-quants.c
cp ../../ggml/src/ggml-cpu/ggml-cpu.c ext/ggml/src/ggml-cpu/ggml-cpu.c
cp ../../ggml/src/ggml-quants.c ext/ggml/src/ggml-quants.c
cp ../../ggml/src/ggml.c ext/ggml/src/ggml.c
cp ../../ggml/src/ggml-amx/ggml-amx.cpp ext/ggml/src/ggml-amx/ggml-amx.cpp
cp ../../ggml/src/ggml-amx/mmq.cpp ext/ggml/src/ggml-amx/mmq.cpp
cp ../../ggml/src/ggml-backend-reg.cpp ext/ggml/src/ggml-backend-reg.cpp
cp ../../ggml/src/ggml-backend.cpp ext/ggml/src/ggml-backend.cpp
cp ../../ggml/src/ggml-blas/ggml-blas.cpp ext/ggml/src/ggml-blas/ggml-blas.cpp
cp ../../ggml/src/ggml-cann/acl_tensor.cpp ext/ggml/src/ggml-cann/acl_tensor.cpp
cp ../../ggml/src/ggml-cann/aclnn_ops.cpp ext/ggml/src/ggml-cann/aclnn_ops.cpp
cp ../../ggml/src/ggml-cann/ggml-cann.cpp ext/ggml/src/ggml-cann/ggml-cann.cpp
cp ../../ggml/src/ggml-cann/kernels/dup.cpp ext/ggml/src/ggml-cann/kernels/dup.cpp
cp ../../ggml/src/ggml-cann/kernels/get_row_f16.cpp ext/ggml/src/ggml-cann/kernels/get_row_f16.cpp
cp ../../ggml/src/ggml-cann/kernels/get_row_f32.cpp ext/ggml/src/ggml-cann/kernels/get_row_f32.cpp
cp ../../ggml/src/ggml-cann/kernels/get_row_q4_0.cpp ext/ggml/src/ggml-cann/kernels/get_row_q4_0.cpp
cp ../../ggml/src/ggml-cann/kernels/get_row_q8_0.cpp ext/ggml/src/ggml-cann/kernels/get_row_q8_0.cpp
cp ../../ggml/src/ggml-cann/kernels/quantize_f16_q8_0.cpp ext/ggml/src/ggml-cann/kernels/quantize_f16_q8_0.cpp
cp ../../ggml/src/ggml-cann/kernels/quantize_f32_q8_0.cpp ext/ggml/src/ggml-cann/kernels/quantize_f32_q8_0.cpp
cp ../../ggml/src/ggml-cann/kernels/quantize_float_to_q4_0.cpp ext/ggml/src/ggml-cann/kernels/quantize_float_to_q4_0.cpp
cp ../../ggml/src/ggml-cpu/amx/amx.cpp ext/ggml/src/ggml-cpu/amx/amx.cpp
cp ../../ggml/src/ggml-cpu/amx/mmq.cpp ext/ggml/src/ggml-cpu/amx/mmq.cpp
cp ../../ggml/src/ggml-cpu/cpu-feats-x86.cpp ext/ggml/src/ggml-cpu/cpu-feats-x86.cpp
cp ../../ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp ext/ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp
cp ../../ggml/src/ggml-cpu/ggml-cpu-hbm.cpp ext/ggml/src/ggml-cpu/ggml-cpu-hbm.cpp
cp ../../ggml/src/ggml-cpu/ggml-cpu-traits.cpp ext/ggml/src/ggml-cpu/ggml-cpu-traits.cpp
cp ../../ggml/src/ggml-cpu/ggml-cpu.cpp ext/ggml/src/ggml-cpu/ggml-cpu.cpp
cp ../../ggml/src/ggml-cpu/kleidiai/kernels.cpp ext/ggml/src/ggml-cpu/kleidiai/kernels.cpp
cp ../../ggml/src/ggml-cpu/kleidiai/kleidiai.cpp ext/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
cp ../../ggml/src/ggml-cpu/llamafile/sgemm.cpp ext/ggml/src/ggml-cpu/llamafile/sgemm.cpp
cp ../../ggml/src/ggml-kompute/ggml-kompute.cpp ext/ggml/src/ggml-kompute/ggml-kompute.cpp
cp ../../ggml/src/ggml-opencl/ggml-opencl.cpp ext/ggml/src/ggml-opencl/ggml-opencl.cpp
cp ../../ggml/src/ggml-opt.cpp ext/ggml/src/ggml-opt.cpp
cp ../../ggml/src/ggml-rpc/ggml-rpc.cpp ext/ggml/src/ggml-rpc/ggml-rpc.cpp
cp ../../ggml/src/ggml-sycl/common.cpp ext/ggml/src/ggml-sycl/common.cpp
cp ../../ggml/src/ggml-sycl/concat.cpp ext/ggml/src/ggml-sycl/concat.cpp
cp ../../ggml/src/ggml-sycl/conv.cpp ext/ggml/src/ggml-sycl/conv.cpp
cp ../../ggml/src/ggml-sycl/convert.cpp ext/ggml/src/ggml-sycl/convert.cpp
cp ../../ggml/src/ggml-sycl/cpy.cpp ext/ggml/src/ggml-sycl/cpy.cpp
cp ../../ggml/src/ggml-sycl/dmmv.cpp ext/ggml/src/ggml-sycl/dmmv.cpp
cp ../../ggml/src/ggml-sycl/element_wise.cpp ext/ggml/src/ggml-sycl/element_wise.cpp
cp ../../ggml/src/ggml-sycl/getrows.cpp ext/ggml/src/ggml-sycl/getrows.cpp
cp ../../ggml/src/ggml-sycl/ggml-sycl.cpp ext/ggml/src/ggml-sycl/ggml-sycl.cpp
cp ../../ggml/src/ggml-sycl/gla.cpp ext/ggml/src/ggml-sycl/gla.cpp
cp ../../ggml/src/ggml-sycl/im2col.cpp ext/ggml/src/ggml-sycl/im2col.cpp
cp ../../ggml/src/ggml-sycl/mmq.cpp ext/ggml/src/ggml-sycl/mmq.cpp
cp ../../ggml/src/ggml-sycl/mmvq.cpp ext/ggml/src/ggml-sycl/mmvq.cpp
cp ../../ggml/src/ggml-sycl/norm.cpp ext/ggml/src/ggml-sycl/norm.cpp
cp ../../ggml/src/ggml-sycl/outprod.cpp ext/ggml/src/ggml-sycl/outprod.cpp
cp ../../ggml/src/ggml-sycl/rope.cpp ext/ggml/src/ggml-sycl/rope.cpp
cp ../../ggml/src/ggml-sycl/softmax.cpp ext/ggml/src/ggml-sycl/softmax.cpp
cp ../../ggml/src/ggml-sycl/sycl_hw.cpp ext/ggml/src/ggml-sycl/sycl_hw.cpp
cp ../../ggml/src/ggml-sycl/tsembd.cpp ext/ggml/src/ggml-sycl/tsembd.cpp
cp ../../ggml/src/ggml-sycl/wkv6.cpp ext/ggml/src/ggml-sycl/wkv6.cpp
cp ../../ggml/src/ggml-threading.cpp ext/ggml/src/ggml-threading.cpp
cp ../../ggml/src/ggml-vulkan/ggml-vulkan.cpp ext/ggml/src/ggml-vulkan/ggml-vulkan.cpp
cp ../../ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp ext/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
cp ../../ggml/src/gguf.cpp ext/ggml/src/gguf.cpp
cp ../../ggml/include/ggml-alloc.h ext/ggml/include/ggml-alloc.h
cp ../../ggml/include/ggml-backend.h ext/ggml/include/ggml-backend.h
cp ../../ggml/include/ggml-blas.h ext/ggml/include/ggml-blas.h
cp ../../ggml/include/ggml-cann.h ext/ggml/include/ggml-cann.h
cp ../../ggml/include/ggml-cpp.h ext/ggml/include/ggml-cpp.h
cp ../../ggml/include/ggml-cpu.h ext/ggml/include/ggml-cpu.h
cp ../../ggml/include/ggml-cuda.h ext/ggml/include/ggml-cuda.h
cp ../../ggml/include/ggml-kompute.h ext/ggml/include/ggml-kompute.h
cp ../../ggml/include/ggml-metal.h ext/ggml/include/ggml-metal.h
cp ../../ggml/include/ggml-opencl.h ext/ggml/include/ggml-opencl.h
cp ../../ggml/include/ggml-opt.h ext/ggml/include/ggml-opt.h
cp ../../ggml/include/ggml-rpc.h ext/ggml/include/ggml-rpc.h
cp ../../ggml/include/ggml-sycl.h ext/ggml/include/ggml-sycl.h
cp ../../ggml/include/ggml-vulkan.h ext/ggml/include/ggml-vulkan.h
cp ../../ggml/include/ggml.h ext/ggml/include/ggml.h
cp ../../ggml/include/gguf.h ext/ggml/include/gguf.h
cp ../../ggml/src/ggml-amx/common.h ext/ggml/src/ggml-amx/common.h
cp ../../ggml/src/ggml-amx/mmq.h ext/ggml/src/ggml-amx/mmq.h
cp ../../ggml/src/ggml-backend-impl.h ext/ggml/src/ggml-backend-impl.h
cp ../../ggml/src/ggml-cann/acl_tensor.h ext/ggml/src/ggml-cann/acl_tensor.h
cp ../../ggml/src/ggml-cann/aclnn_ops.h ext/ggml/src/ggml-cann/aclnn_ops.h
cp ../../ggml/src/ggml-cann/common.h ext/ggml/src/ggml-cann/common.h
cp ../../ggml/src/ggml-cann/kernels/ascendc_kernels.h ext/ggml/src/ggml-cann/kernels/ascendc_kernels.h
cp ../../ggml/src/ggml-common.h ext/ggml/src/ggml-common.h
cp ../../ggml/src/ggml-cpu/amx/amx.h ext/ggml/src/ggml-cpu/amx/amx.h
cp ../../ggml/src/ggml-cpu/amx/common.h ext/ggml/src/ggml-cpu/amx/common.h
cp ../../ggml/src/ggml-cpu/amx/mmq.h ext/ggml/src/ggml-cpu/amx/mmq.h
cp ../../ggml/src/ggml-cpu/ggml-cpu-aarch64.h ext/ggml/src/ggml-cpu/ggml-cpu-aarch64.h
cp ../../ggml/src/ggml-cpu/ggml-cpu-hbm.h ext/ggml/src/ggml-cpu/ggml-cpu-hbm.h
cp ../../ggml/src/ggml-cpu/ggml-cpu-impl.h ext/ggml/src/ggml-cpu/ggml-cpu-impl.h
cp ../../ggml/src/ggml-cpu/ggml-cpu-quants.h ext/ggml/src/ggml-cpu/ggml-cpu-quants.h
cp ../../ggml/src/ggml-cpu/ggml-cpu-traits.h ext/ggml/src/ggml-cpu/ggml-cpu-traits.h
cp ../../ggml/src/ggml-cpu/kleidiai/kernels.h ext/ggml/src/ggml-cpu/kleidiai/kernels.h
cp ../../ggml/src/ggml-cpu/kleidiai/kleidiai.h ext/ggml/src/ggml-cpu/kleidiai/kleidiai.h
cp ../../ggml/src/ggml-cpu/llamafile/sgemm.h ext/ggml/src/ggml-cpu/llamafile/sgemm.h
cp ../../ggml/src/ggml-cuda/vendors/cuda.h ext/ggml/src/ggml-cuda/vendors/cuda.h
cp ../../ggml/src/ggml-cuda/vendors/hip.h ext/ggml/src/ggml-cuda/vendors/hip.h
cp ../../ggml/src/ggml-cuda/vendors/musa.h ext/ggml/src/ggml-cuda/vendors/musa.h
cp ../../ggml/src/ggml-impl.h ext/ggml/src/ggml-impl.h
cp ../../ggml/src/ggml-metal/ggml-metal-impl.h ext/ggml/src/ggml-metal/ggml-metal-impl.h
cp ../../ggml/src/ggml-quants.h ext/ggml/src/ggml-quants.h
cp ../../ggml/src/ggml-threading.h ext/ggml/src/ggml-threading.h
cp ../../ggml/src/ggml-metal/ggml-metal.m ext/ggml/src/ggml-metal/ggml-metal.m
cp ../../ggml/src/ggml-metal/ggml-metal.metal ext/ggml/src/ggml-metal/ggml-metal.metal
cp ../../scripts/get-flags.mk ext/scripts/get-flags.mk
cp ../../examples/common.h ext/examples/common.h
cp ../../examples/common.cpp ext/examples/common.cpp
cp ../../examples/common-whisper.h ext/examples/common-whisper.h
cp ../../examples/common-whisper.cpp ext/examples/common-whisper.cpp
cp ../../examples/stb_vorbis.c ext/examples/stb_vorbis.c
cp ../../examples/miniaudio.h ext/examples/miniaudio.h
cp ../../LICENSE LICENSE
whispercpp 1.3.1 built to pkg/whispercpp-1.3.1.gem.
```

## Testing
```console
$ cd bindings/ruby
$ rake test
Loaded suite /usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/rake_test_loader
Started
.failed to process audio
.failed to process audio
...................  Successfully built RubyGem
  Name: whispercpp
  Version: 1.3.1
  File: 20250327-10956-on0nwn
.whispercpp 1.3.1 built to pkg/whispercpp-1.3.1.gem.
Building native extensions. This could take a while...
Successfully installed whispercpp-1.3.1
1 gem installed
...................................................................................[BUG] Segmentation fault at 0x0000000000000030
ruby 3.2.3 (2024-01-18 revision 52bb2ac0a6) [x86_64-linux-gnu]

-- Machine register context ------------------------------------------------
 RIP: 0x0000783864a5fc84 RBP: 0x00007838364be110 RSP: 0x00007838364be070
 RAX: 0x00000f070c86f54f RBX: 0x000078383783dcd0 RCX: 0x00007838364be120
 RDX: 0x00000003c2b3ceb1 RDI: 0x0000783864ba6d38 RSI: 0x000078386437d538
  R8: 0x0000000000000277  R9: 0x00005608c34797c0 R10: 0x0000000000000004
 R11: 0x0000783864abf1d0 R12: 0x0000000000000003 R13: 0x00007838364be120
 R14: 0x0000000000000d61 R15: 0x0000000000000000 EFL: 0x0000000000010216

-- C level backtrace information -------------------------------------------
SEGV received in SEGV handler
Aborted (core dumped)
rake aborted!
Command failed with status (134)

Tasks: TOP => test
(See full trace by running task with --trace)

```
With `--trace`:
```console
Loaded suite /usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/rake_test_loader
Started
.failed to process audio
.failed to process audio
...................  Successfully built RubyGem
  Name: whispercpp
  Version: 1.3.1
  File: 20250327-14061-o0lsl9
.whispercpp 1.3.1 built to pkg/whispercpp-1.3.1.gem.
Building native extensions. This could take a while...
Successfully installed whispercpp-1.3.1
1 gem installed
...................................................................................[BUG] Segmentation fault at 0x0000000000000030
ruby 3.2.3 (2024-01-18 revision 52bb2ac0a6) [x86_64-linux-gnu]

-- Machine register context ------------------------------------------------
 RIP: 0x000079bd5ac5fc84 RBP: 0x000079bd2c68b110 RSP: 0x000079bd2c68b070
 RAX: 0x00000f37ab5bd567 RBX: 0x000079bd5401dca0 RCX: 0x000079bd2c68b120
 RDX: 0x00000003cede8e99 RDI: 0x000079bd5ada6d38 RSI: 0x000079bd5aded528
  R8: 0x000000000000025d  R9: 0x000056d31a9acf00 R10: 0x0000000000000004
 R11: 0x000079bd5acbf1d0 R12: 0x0000000000000003 R13: 0x000079bd2c68b120
 R14: 0x0000000000000d61 R15: 0x0000000000000000 EFL: 0x0000000000010212

-- C level backtrace information -------------------------------------------
SEGV received in SEGV handler
Aborted (core dumped)
rake aborted!
Command failed with status (134): [ruby -w -I"lib" /usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/rake_test_loader.rb "tests/test_callback.rb" "tests/test_error.rb" "tests/test_model.rb" "tests/test_package.rb" "tests/test_params.rb" "tests/test_segment.rb" "tests/test_whisper.rb" ]
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/testtask.rb:130:in `block (3 levels) in define'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/file_utils.rb:57:in `sh'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/file_utils.rb:104:in `ruby'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/testtask.rb:117:in `block (2 levels) in define'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/file_utils_ext.rb:58:in `verbose'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/testtask.rb:111:in `block in define'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/task.rb:281:in `block in execute'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/task.rb:281:in `each'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/task.rb:281:in `execute'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/task.rb:219:in `block in invoke_with_call_chain'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/task.rb:199:in `synchronize'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/task.rb:199:in `invoke_with_call_chain'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/task.rb:188:in `invoke'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:160:in `invoke_task'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:116:in `block (2 levels) in top_level'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:116:in `each'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:116:in `block in top_level'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:125:in `run_with_threads'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:110:in `top_level'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:83:in `block in run'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:186:in `standard_exception_handling'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/application.rb:80:in `run'
/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/exe/rake:27:in `<top (required)>'
/usr/bin/rake:25:in `load'
/usr/bin/rake:25:in `<main>'
Tasks: TOP => test
```

Run the test in gdb:
```console
$ gdb --args ruby -I lib tests/test_whisper.rb TESTOPTS="-v"

Thread 654 "ruby" received signal SIGSEGV, Segmentation fault.
[Switching to Thread 0x7fffc96826c0 (LWP 49817)]
0x00007ffff7c5fc84 in rb_funcallv () from /lib/x86_64-linux-gnu/libruby-3.2.so.3.2

(gdb) up
#1  0x00007ffff250d73a in ruby_whisper_log_callback (user_data=<optimized out>,
    buffer=0x7fffc96801a0 "whisper_full_with_state: input is too short - 680 ms < 1000 ms. consider padding the input audio with silence\n", level=GGML_LOG_LEVEL_WARN) at ruby_whisper.c:95
warning: Source file is more recent than executable.
95	  VALUE udata = rb_iv_get(mWhisper, "user_data");

(gdb) up
#2  ruby_whisper_log_callback (level=GGML_LOG_LEVEL_WARN,
    buffer=0x7fffc96801a0 "whisper_full_with_state: input is too short - 680 ms < 1000 ms. consider padding the input audio with silence\n", user_data=<optimized out>) at ruby_whisper.c:89
89	static void

(gdb) up
#3  0x00007ffff25142b3 in whisper_log_internal (level=level@entry=GGML_LOG_LEVEL_WARN,
    format=format@entry=0x7ffff254db58 "%s: input is too short - %d ms < 1000 ms. consider padding the input audio with silence\n")
    at src/whisper.cpp:7531
7531	        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
```
So this originates from:
```c++
    if (seek_end < seek_start + 100) {
        WHISPER_LOG_WARN("%s: input is too short - %d ms < 1000 ms. consider padding the input audio with silence\n", __func__, (seek_end - seek_start)*10);
        return 0;
    }
```
Which is in `whisper_full_with_state` in `src/whisper.cpp`. And the problem seems
to be with the userdata.

Run a single test case in gdb:
```console
$ gdb --args ruby -I"lib" -r/usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/rake_test_loader.rb tests/test_whisper.rb -- --name=test_full_parallel
```

Run a single test:
```console
$ rake test TEST=tests/test_model.rb
Finished in 5.35648552 seconds.
--------------------------------------------------------------------------------------------------------------------------------------
7 tests, 63 assertions, 0 failures, 0 errors, 0 pendings, 0 omissions, 0 notifications
100% passed
--------------------------------------------------------------------------------------------------------------------------------------
1.31 tests/s, 11.76 assertions/s

```console
$ rake test TEST=tests/test_whisper.rb TESTOPTS="-v --name=test_full_parallel"
```

Find the command that rake uses to run a single test:
```console
$ rake test TEST=tests/test_whisper.rb TESTOPTS="-v --name=test_full_parallel" --trace
```
Then use that command in gdb:
```
$ gdb --args ruby -w -I"lib" /usr/lib/ruby/gems/3.2.0/gems/rake-13.0.6/lib/rake/rake_test_loader.rb "tests/test_whisper.rb" -v --name=test_full_parallel
```

In `whisper_full` we have:
```c++
int whisper_full_parallel(
        struct whisper_context * ctx,
        struct whisper_full_params params,
        const float * samples,
        int n_samples,
        int n_processors) {
    ...
    // combine results into result_state->result_all from all other states
    for (int i = 0; i < n_processors - 1; ++i) {
        auto& results_i = states[i]->result_all;

        for (auto& result : results_i) {
            // correct the segment timestamp taking into account the offset
            result.t0 += 100 * ((i + 1) * n_samples_per_processor) / WHISPER_SAMPLE_RATE + offset_t;
            result.t1 += 100 * ((i + 1) * n_samples_per_processor) / WHISPER_SAMPLE_RATE + offset_t;

            // make sure that segments are not overlapping
            if (!ctx->state->result_all.empty()) {
                result.t0 = std::max(result.t0, ctx->state->result_all.back().t1);
            }

            ctx->state->result_all.push_back(std::move(result));

            // call the new_segment_callback for each segment
            if (params.new_segment_callback) {
                params.new_segment_callback(ctx, ctx->state, 1, params.new_segment_callback_user_data);
            }
        }
        ...
    }
```
So this will get the results from each processor and combine them. But, when
running the Ruby bindings test:
```ruby
    def test_full_parallel
      @whisper.full_parallel(@params, @samples, @samples.length, Etc.nprocessors)

      assert_equal Etc.nprocessors, @whisper.full_n_segments
      text = @whisper.each_segment.collect(&:text).join
      assert_match(/ask what you can do/i, text)
      assert_match(/for your country/i, text)
    end
```
The result_all are 0
```console
(gdb) p states[0]->result_all
$8 = std::vector of length 0, capacity 0
(gdb) p states[1]->result_all
$9 = std::vector of length 0, capacity 0
(gdb) p states[2]->result_all
$10 = std::vector of length 0, capacity 0
(gdb) p states[3]->result_all
$11 = std::vector of length 0, capacity 0
(gdb) p states[4]->result_all
$12 = std::vector of length 0, capacity 0
(gdb) p states[5]->result_all
$13 = std::vector of length 0, capacity 0
```
This means that the `new_segment_callback` will not get called and means that
they asserts in the test will fail:
```console
whisper_full_with_state: input is too short - 680 ms < 1000 ms. consider padding the input audio with silence

whisper_full_parallel: the audio has been split into 16 chunks at the following times:
whisper_full_parallel: split 1 - 00:00:00.680
whisper_full_parallel: split 2 - 00:00:01.370
whisper_full_parallel: split 3 - 00:00:02.060
whisper_full_parallel: split 4 - 00:00:02.750
whisper_full_parallel: split 5 - 00:00:03.430
whisper_full_parallel: split 6 - 00:00:04.120
whisper_full_parallel: split 7 - 00:00:04.810
whisper_full_parallel: split 8 - 00:00:05.500
whisper_full_parallel: split 9 - 00:00:06.180
whisper_full_parallel: split 10 - 00:00:06.870
whisper_full_parallel: split 11 - 00:00:07.560
whisper_full_parallel: split 12 - 00:00:08.250
whisper_full_parallel: split 13 - 00:00:08.930
whisper_full_parallel: split 14 - 00:00:09.620
whisper_full_parallel: split 15 - 00:00:10.310
whisper_full_parallel: the transcription quality may be degraded near these boundaries
F
======================================================================================================================================
Failure: test_full_parallel(TestWhisper::full)
/home/danbev/work/ai/whisper-work/bindings/ruby/tests/test_whisper.rb:180:in `test_full_parallel'
     177:     def test_full_parallel
     178:       @whisper.full_parallel(@params, @samples, @samples.length, Etc.nprocessors)
     179: 
  => 180:       assert_equal Etc.nprocessors, @whisper.full_n_segments
     181:       text = @whisper.each_segment.collect(&:text).join
     182:       assert_match(/ask what you can do/i, text)
     183:       assert_match(/for your country/i, text)
<16> expected but was
<0>
```
Notice the message:
```
whisper_full_with_state: input is too short - 680 ms < 1000 ms. consider padding the input audio with silence
```
This is logged here:
```c++
int whisper_full_with_state(
        struct whisper_context * ctx,
          struct whisper_state * state,
    struct whisper_full_params   params,
                   const float * samples,
                           int   n_samples) {
   ...
    if (seek_end < seek_start + 100) {
        WHISPER_LOG_WARN("%s: input is too short - %d ms < 1000 ms. consider padding the input audio with silence\n", __func__, (seek_end - seek_start)*10);
        return 0;
    }
```
So this will return early and there will be no segments to process.
