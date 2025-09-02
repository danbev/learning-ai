## llama.cpp tests

### Building
```console
$ cmake --fresh -S . -B build \
    -DCMAKE_BUILD_TYPE=Debug\
    -DLLAMA_BUILD_TESTS=ON
$ cmake --build build -j8
```

### Listing all tests
```console
$ ctest --test-dir build -N
Internal ctest changing into directory: /home/danbev/work/ai/llama.cpp/build
Test project /home/danbev/work/ai/llama.cpp/build
  Test  #1: test-tokenizer-0-bert-bge
  Test  #2: test-tokenizer-0-command-r
  Test  #3: test-tokenizer-0-deepseek-coder
  Test  #4: test-tokenizer-0-deepseek-llm
  Test  #5: test-tokenizer-0-falcon
  Test  #6: test-tokenizer-0-gpt-2
  Test  #7: test-tokenizer-0-llama-bpe
  Test  #8: test-tokenizer-0-llama-spm
  Test  #9: test-tokenizer-0-mpt
  Test #10: test-tokenizer-0-phi-3
  Test #11: test-tokenizer-0-qwen2
  Test #12: test-tokenizer-0-refact
  Test #13: test-tokenizer-0-starcoder
  Test #14: test-tokenizers-ggml-vocabs
  Test #15: test-sampling
  Test #16: test-grammar-parser
  Test #17: test-grammar-integration
  Test #18: test-llama-grammar
  Test #19: test-chat
  Test #20: test-json-schema-to-grammar
  Test #21: test-tokenizer-1-llama-spm
  Test #22: test-chat-parser
  Test #23: test-chat-template
  Test #24: test-json-partial
  Test #25: test-log
  Test #26: test-regex-partial
  Test #27: test-thread-safety
  Test #28: test-arg-parser
  Test #29: test-opt
  Test #30: test-gguf
  Test #31: test-backend-ops
  Test #32: test-model-load-cancel
  Test #33: test-autorelease
  Test #34: test-barrier
  Test #35: test-quantize-fns
  Test #36: test-quantize-perf
  Test #37: test-rope
  Test #38: test-mtmd-c-api
  Test #39: test-eval-callback

Total Tests: 39
```

### test-backend-ops
This tests uses multiple backends to compare different backend operations with
the CPU backend as reference.

If you don't have multiple backends available/built, this test will be skipped:
```console
$ ctest --test-dir build -R 'test-backend-ops'
test 31
    Start 31: test-backend-ops

31: Test command: /home/danbev/work/ai/llama.cpp/build/bin/test-backend-ops
31: Working Directory: .
31: Test timeout computed to be: 1500
31: register_backend: registered backend CPU (1 devices)
31: register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
31: load_backend: failed to find ggml_backend_init in /home/danbev/work/ai/llama.cpp/build/bin/libggml-cpu.so
31: Testing 1 devices
31:
31: Backend 1/1: CPU
31:   Skipping CPU backend
31: 1/1 backends passed
31: OK
1/1 Test #31: test-backend-ops .................   Passed    0.01 sec

The following tests passed:
	test-backend-ops

100% tests passed, 0 tests failed out of 1

Label Time Summary:
main    =   0.01 sec*proc (1 test)

Total Test time (real) =   0.01 sec
```
But if we build with CUDA we should see the following:
```console
$ ctest --test-dir build -R 'test-backend-ops'
Internal ctest changing into directory: /home/danbev/work/ai/llama.cpp/build
Test project /home/danbev/work/ai/llama.cpp/build
    Start 31: test-backend-ops
1/1 Test #31: test-backend-ops .................   Passed  108.74 sec

100% tests passed, 0 tests failed out of 1

Label Time Summary:
main    = 108.74 sec*proc (1 test)

Total Test time (real) = 108.75 sec
```
By default ctests has `--output-on-failure` enables so output will only be shown
in case of a failure. To see all output, use `-V`:
```console
31: ggml_cuda_init: found 1 CUDA devices:                                       
31:   Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes            
31: Testing 2 devices                                                           
31:                                                                             
31: Backend 1/2: CUDA0                                                          
31:   Device description: NVIDIA GeForce RTX 4070                               
31:   Device memory: 11903 MB (11734 MB free)                                   
31:                                                                             
31:   ABS(type=f16,ne_a=[128,2,2,2],v=0): ^[[1;32mOK^[[0m
```
I piped the output to a file so be able to inspect it and it includes
ANSI escape codes which is the equivalent of:
```console
ABS(type=f16,ne_a=[128,2,2,2],v=0): OK
```
We can run the test directly to see the options it supports:
```console
$ ./build/bin/test-backend-ops --help
Usage: ./build/bin/test-backend-ops [mode] [-o <op,..>] [-b <backend>] [-p <params regex>] [--output <console|sql|csv>]
    valid modes:
      - test (default, compare with CPU backend for correctness)
      - grad (compare gradients from backpropagation with method of finite differences)
      - perf (performance evaluation)
      - support (probe backend operation support)
    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc),
        optionally including the full test case string (e.g. "ADD(type=f16,ne=[1,1,8,1],nr=[1,1,1,1],nf=1)")
    --output specifies output format (default: console, options: console, sql, csv)
```

#### Testing a specific operation:
```console
$ ./build/bin/test-backend-ops -o "ABS(type=f16,ne_a=[128,2,2,2],v=0)"
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
Testing 2 devices

Backend 1/2: CUDA0
  Device description: NVIDIA GeForce RTX 4070
  Device memory: 11903 MB (11734 MB free)

  ABS(type=f16,ne_a=[128,2,2,2],v=0): OK
  11837/11837 tests passed
  Backend CUDA0: OK
Backend 2/2: CPU
  Skipping CPU backend
2/2 backends passed
OK
```

### Walkthrough
```console
$ gdb --args ./build/bin/test-backend-ops -o "ABS(type=f16,ne_a=[128,2,2,2],v=0)"
```

```c++
   ...
        bool ok = test_backend(backend, mode, op_names_filter, params_filter, output_printer.get());
```
```console
(gdb) p mode
$1 = MODE_TEST
(gdb) p op_names_filter
$2 = 0x7fffffffdbd6 "ABS(type=f16,ne_a=[128,2,2,2],v=0)"
```
```c++
static bool test_backend(ggml_backend_t backend, test_mode mode, const char * op_names_filter, const char * params_filter,
                         printer * output_printer) {
    auto filter_test_cases = [](std::vector<std::unique_ptr<test_case>> & test_cases, const char * params_filter) {
        if (params_filter == nullptr) {
            return;
        }

        std::regex params_filter_regex(params_filter);

        for (auto it = test_cases.begin(); it != test_cases.end();) {
            if (!std::regex_search((*it)->vars(), params_filter_regex)) {
                it = test_cases.erase(it);
                continue;
            }

            it++;
        }
    };

    if (mode == MODE_TEST) {
        auto test_cases = make_test_cases_eval();
```
```c++
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    // unary ops
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int v : {0, 1}) {
            for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
                test_cases.emplace_back(new test_unary((ggml_unary_op) op, type, { 128, 2, 2, 2 }, v));
                test_cases.emplace_back(new test_unary((ggml_unary_op) op, type, { 5, 7, 11, 13 }, v));
            }
        }
    }
```
Notice that this is using a brace-initialized list (initializer list) to iterate
over the two types `GGML_TYPE_F16` and `GGML_TYPE_F32`.
I think `v` is for view and if it is 1 then a view will be created instead of
a normal tensor.
Next all the unary operations are iterated over:
```c
    enum ggml_unary_op {
        GGML_UNARY_OP_ABS,
        GGML_UNARY_OP_SGN,
        GGML_UNARY_OP_NEG,
        GGML_UNARY_OP_STEP,
        GGML_UNARY_OP_TANH,
        GGML_UNARY_OP_ELU,
        GGML_UNARY_OP_RELU,
        GGML_UNARY_OP_SIGMOID,
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_GELU_QUICK,
        GGML_UNARY_OP_SILU,
        GGML_UNARY_OP_HARDSWISH,
        GGML_UNARY_OP_HARDSIGMOID,
        GGML_UNARY_OP_EXP,
        GGML_UNARY_OP_GELU_ERF,

        GGML_UNARY_OP_COUNT,
    };
```
```console
(gdb) p (ggml_unary_op) op
$10 = GGML_UNARY_OP_ABS
```
