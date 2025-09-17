The following in an issue that can be reproduced using the test-thread-safety.

### Building
This was run on macOS M3 and llama.cpp was built using:
```console
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_SANITIZE_THREAD=ON \
    -DGGML_OPENMP=OFF\
    -DGGML_BLAS=OFF \
    -DGGML_METAL_EMBED_LIBRARY=OFF \
    -DGGML_METAL=OFF

cmake --build build -j8
```
In the code snippets below I've removed OpenMP related code and also for non macos platforms to
simplify things a bit.

### Reproducing the issue
When running the following command:
```console
./build/bin/test-thread-safety -hf ggml-org/gemma-3-270m-qat-GGUF "-ngl" "99" "-p" "The meaning of life is" "-n" "128" "-c" "256" "-ub" "32" "-np" "2" -t 2
```

### Error
When running the following code with ThreadSanitizer enabled:
```console
==================
WARNING: ThreadSanitizer: data race (pid=44523)
  Write of size 4 at 0x000109288158 by thread T8:
    #0 ggml_threadpool_new_impl <null> (libggml-cpu.dylib:arm64+0x65a8)
    #1 ggml_graph_compute <null> (libggml-cpu.dylib:arm64+0x6bf0)
    #2 ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*) <null> (libggml-cpu.dylib:arm64+0xa400)
    #3 ggml_backend_sched_graph_compute_async <null> (libggml-base.dylib:arm64+0x2dda4)
    #4 llama_context::graph_compute(ggml_cgraph*, bool) <null> (libllama.dylib:arm64+0x2bd28)
    #5 llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) <null> (libllama.dylib:arm64+0x2b840)
    #6 llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) <null> (libllama.dylib:arm64+0x2b554)
    #7 llama_decode <null> (libllama.dylib:arm64+0x33848)
    #8 void* std::__1::__thread_proxy[abi:ne200100]<std::__1::tuple<std::__1::unique_ptr<std::__1::__thread_struct, std::__1::default_delete<std::__1::__thread_struct>>, main::$_0>>(void*) <null> (test-thread-safety:arm64+0x100006e18)

  Previous read of size 4 at 0x000109288158 by thread T10:
    #0 ggml_threadpool_free <null> (libggml-cpu.dylib:arm64+0x58bc)
    #1 ggml_graph_compute <null> (libggml-cpu.dylib:arm64+0x6cd8)
    #2 ggml_backend_cpu_graph_compute(ggml_backend*, ggml_cgraph*) <null> (libggml-cpu.dylib:arm64+0xa400)
    #3 ggml_backend_sched_graph_compute_async <null> (libggml-base.dylib:arm64+0x2dda4)
    #4 llama_context::graph_compute(ggml_cgraph*, bool) <null> (libllama.dylib:arm64+0x2bd28)
    #5 llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) <null> (libllama.dylib:arm64+0x2b840)
    #6 llama_context::process_ubatch(llama_ubatch const&, llm_graph_type, llama_memory_context_i*, ggml_status&) <null> (libllama.dylib:arm64+0x2b554)
    #7 llama_decode <null> (libllama.dylib:arm64+0x33848)
    #8 void* std::__1::__thread_proxy[abi:ne200100]<std::__1::tuple<std::__1::unique_ptr<std::__1::__thread_struct, std::__1::default_delete<std::__1::__thread_struct>>, main::$_0>>(void*) <null> (test-thread-safety:arm64+0x100006e18)

  Thread T8 (tid=90952676, running) created by main thread at:
    #0 pthread_create <null> (libclang_rt.tsan_osx_dynamic.dylib:arm64e+0x2f708)
    #1 main <null> (test-thread-safety:arm64+0x100002fd8)

  Thread T10 (tid=90952678, running) created by main thread at:
    #0 pthread_create <null> (libclang_rt.tsan_osx_dynamic.dylib:arm64e+0x2f708)
    #1 main <null> (test-thread-safety:arm64+0x100002fd8)

SUMMARY: ThreadSanitizer: data race (libggml-cpu.dylib:arm64+0x65a8) in ggml_threadpool_new_impl+0xf8
```

### Troubleshooting
So we have two threads, T18 and T10, one is trying to create a threadpool 
```c++
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    if (n_threads > threadpool->n_threads_max) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads_max);
        n_threads = threadpool->n_threads_max;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}
```
In this case the cplan threadpool is null so a new threadpool is created by T10, which
executes the function and eventually calls `ggml_threadpool_free`.

Now, Thread 8 will execute the same function and its cplan threadpool is also null, so it will
call `ggml_threadpool_new_impl`. Depending on the timing, sometimes T10 is calling
`ggml_threadpool_free` at the same time as T8 is calling `ggml_threadpool_new_impl`.

```c++
static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads_max    = tpp->n_threads;
        threadpool->n_threads_cur    = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

    return threadpool;
}
```
So this will first allocate the threadpool structure by calling `ggml_aligned_malloc`.
```c++
void * ggml_aligned_malloc(size_t size) {
    const int alignment = 64;

    if (size == 0) {
        GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for ggml_aligned_malloc!\n");
        return NULL;
    }
    void * aligned_memory = NULL;
    GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    int result = EFAULT;
    switch (alloc_status) {
        case KERN_SUCCESS:
            result = 0;
            break;
        case KERN_INVALID_ADDRESS:
            result = EINVAL;
            break;
        case KERN_NO_SPACE:
            result = ENOMEM;
            break;
        default:
            result = EFAULT;
            break;
    }
    if (result != 0) {
        // Handle allocation failure
        const char *error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        return NULL;
    }
    return aligned_memory;
}
```
Now, if we look at `ggml_threadpool_free`:
```c++
void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads_max;

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}
```
So we can see that the thread pool is being used to first read `n_threads_max` and then
the workers are freed and then the threadpool itself is freed.

I've tried using a Debug build but this issue does not happen for it, and adding printf statments
seems to make it go away as well, so it seems like a timing issue.
Lets try using the debugger anyway to see how the compiler executing the instructions.
```console
$ lldb ./build/bin/test-thread-safety -- -hf ggml-org/gemma-3-270m-qat-GGUF "-ngl" "99" "-p" "The meaning of life is" "-n" "128" "-c" "256" "-ub" "32" "-np" "2" -t 2
(lldb) target create "./build/bin/test-thread-safety"
Current executable set to '/Users/danbev/work/ai/llama.cpp/build/bin/test-thread-safety' (arm64).
(lldb) settings set -- target.run-args  "-hf" "ggml-org/gemma-3-270m-qat-GGUF" "-ngl" "99" "-p" "The meaning of life is" "-n" "128" "-c" "256" "-ub" "32" "-np" "2" "-t" "2"
(lldb) br set -n ggml_threadpool_free
Breakpoint 1: where = libggml-cpu.dylib`ggml_threadpool_free, address = 0x0000000000005894
(lldb) r
(lldb) (lldb) disassemble
libggml-cpu.dylib`ggml_threadpool_free:
->  0x100891894 <+0>:   sub    sp, sp, #0x40
    0x100891898 <+4>:   stp    x22, x21, [sp, #0x10]
    0x10089189c <+8>:   stp    x20, x19, [sp, #0x20]
    0x1008918a0 <+12>:  stp    x29, x30, [sp, #0x30]
    0x1008918a4 <+16>:  add    x29, sp, #0x30
    0x1008918a8 <+20>:  mov    x19, x0
    0x1008918ac <+24>:  mov    x0, x30
    0x1008918b0 <+28>:  bl     0x10094bdac    ; symbol stub for: __tsan_func_entry
    0x1008918b4 <+32>:  cbz    x19, 0x10089197c ; <+232>
    0x1008918b8 <+36>:  add    x0, x19, #0x158
    0x1008918bc <+40>:  bl     0x10094be0c    ; symbol stub for: __tsan_read4
    0x1008918c0 <+44>:  ldr    w21, [x19, #0x158] ; <----------------  read n_threads_max (notice read4)
    0x1008918c4 <+48>:  add    x0, x19, #0x150
    0x1008918c8 <+52>:  bl     0x10094be18    ; symbol stub for: __tsan_read8
    0x1008918cc <+56>:  ldr    x20, [x19, #0x150] ; <----------------  read workers (notice read8)
    0x1008918d0 <+60>:  mov    x0, x19
    0x1008918d4 <+64>:  bl     0x10094c1c0    ; symbol stub for: pthread_mutex_lock
    0x1008918d8 <+68>:  add    x0, x19, #0x144
    0x1008918dc <+72>:  mov    w1, #0x1 ; =1
    0x1008918e0 <+76>:  mov    w2, #0x5 ; =5
    0x1008918e4 <+80>:  bl     0x10094bda0    ; symbol stub for: __tsan_atomic8_store
    0x1008918e8 <+84>:  add    x0, x19, #0x145
    0x1008918ec <+88>:  mov    w1, #0x0 ; =0
    0x1008918f0 <+92>:  mov    w2, #0x5 ; =5
    0x1008918f4 <+96>:  bl     0x10094bda0    ; symbol stub for: __tsan_atomic8_store
    0x1008918f8 <+100>: add    x0, x19, #0x40
    0x1008918fc <+104>: bl     0x10094c160    ; symbol stub for: pthread_cond_broadcast
    0x100891900 <+108>: mov    x0, x19
    0x100891904 <+112>: bl     0x10094c1cc    ; symbol stub for: pthread_mutex_unlock
    0x100891908 <+116>: cmp    w21, #0x2
    0x10089190c <+120>: b.lt   0x100891940    ; <+172>
    0x100891910 <+124>: add    x20, x20, #0x220
    0x100891914 <+128>: sub    x22, x21, #0x1
    0x100891918 <+132>: mov    x0, x20
    0x10089191c <+136>: bl     0x10094be18    ; symbol stub for: __tsan_read8
    0x100891920 <+140>: ldr    x0, [x20]
    0x100891924 <+144>: mov    x1, #0x0 ; =0
    0x100891928 <+148>: bl     0x10094c19c    ; symbol stub for: pthread_join
    0x10089192c <+152>: cmp    w0, #0x2
    0x100891930 <+156>: b.hs   0x100891994    ; <+256>
    0x100891934 <+160>: add    x20, x20, #0x220
    0x100891938 <+164>: subs   x22, x22, #0x1
    0x10089193c <+168>: b.ne   0x100891918    ; <+132>
    0x100891940 <+172>: mov    x0, x19
    0x100891944 <+176>: bl     0x10094c1a8    ; symbol stub for: pthread_mutex_destroy
    0x100891948 <+180>: add    x0, x19, #0x40
    0x10089194c <+184>: bl     0x10094c16c    ; symbol stub for: pthread_cond_destroy
    0x100891950 <+188>: sxtw   x8, w21
    0x100891954 <+192>: add    x8, x8, w21, sxtw #4
    0x100891958 <+196>: lsl    x20, x8, #5
    0x10089195c <+200>: add    x0, x19, #0x150
    0x100891960 <+204>: bl     0x10094be18    ; symbol stub for: __tsan_read8
    0x100891964 <+208>: ldr    x0, [x19, #0x150]
    0x100891968 <+212>: mov    x1, x20
    0x10089196c <+216>: bl     0x10094bf2c    ; symbol stub for: ggml_aligned_free
    0x100891970 <+220>: mov    x0, x19
    0x100891974 <+224>: mov    w1, #0x180 ; =384
    0x100891978 <+228>: bl     0x10094bf2c    ; symbol stub for: ggml_aligned_free
    0x10089197c <+232>: bl     0x10094bdb8    ; symbol stub for: __tsan_func_exit
    0x100891980 <+236>: ldp    x29, x30, [sp, #0x30]
    0x100891984 <+240>: ldp    x20, x19, [sp, #0x20]
    0x100891988 <+244>: ldp    x22, x21, [sp, #0x10]
    0x10089198c <+248>: add    sp, sp, #0x40
    0x100891990 <+252>: ret
    0x100891994 <+256>: adrp   x8, 205
    0x100891998 <+260>: add    x8, x8, #0x9cb ; "rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED"
    0x10089199c <+264>: adrp   x0, 205
    0x1008919a0 <+268>: add    x0, x0, #0x7c0 ; "/Users/danbev/work/ai/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c"
    0x1008919a4 <+272>: str    x8, [sp]
    0x1008919a8 <+276>: adrp   x2, 205
    0x1008919ac <+280>: add    x2, x2, #0x7fd ; "GGML_ASSERT(%s) failed"
    0x1008919b0 <+284>: mov    w1, #0xa36 ; =2614
    0x1008919b4 <+288>: bl     0x10094bf20    ; symbol stub for: ggml_abort
```

Debugging the value of the w21 register after the read instruction shows:
```console
(lldb) register read w21
     w21 = 0x00000002
```
This corresponds with the number of threads we are using.

And this was reported by the ThreadSanitizer as the read:
```
  Previous read of size 4 at 0x000109288158 by thread T10:
    #0 ggml_threadpool_free <null> (libggml-cpu.dylib:arm64+0x58bc)
```
And there is only one read4 instruction in ggml_threadpool_free, so this must be it.

And the write is happening in:
```console
  Write of size 4 at 0x000109288158 by thread T8:
    #0 ggml_threadpool_new_impl <null> (libggml-cpu.dylib:arm64+0x65a8)
```
Lets inspect `ggml_threadpool_new_impl`:
```console
(lldb) br set -n ggml_threadpool_new_impl
Breakpoint 2: where = libggml-cpu.dylib`ggml_threadpool_new_impl, address = 0x00000001008924b0
(lldb) c
Process 75170 resuming
Process 75170 stopped
* thread #7, stop reason = breakpoint 2.1
    frame #0: 0x00000001008924b0 libggml-cpu.dylib`ggml_threadpool_new_impl
libggml-cpu.dylib`ggml_threadpool_new_impl:
->  0x1008924b0 <+0>:  sub    sp, sp, #0xc0
    0x1008924b4 <+4>:  stp    d9, d8, [sp, #0x50]
    0x1008924b8 <+8>:  stp    x28, x27, [sp, #0x60]
    0x1008924bc <+12>: stp    x26, x25, [sp, #0x70]
Target 0: (test-thread-safety) stopped.
(lldb) disassemble
libggml-cpu.dylib`ggml_threadpool_new_impl:
->  0x1008924b0 <+0>:    sub    sp, sp, #0xc0
    0x1008924b4 <+4>:    stp    d9, d8, [sp, #0x50]
    0x1008924b8 <+8>:    stp    x28, x27, [sp, #0x60]
    0x1008924bc <+12>:   stp    x26, x25, [sp, #0x70]
    0x1008924c0 <+16>:   stp    x24, x23, [sp, #0x80]
    0x1008924c4 <+20>:   stp    x22, x21, [sp, #0x90]
    0x1008924c8 <+24>:   stp    x20, x19, [sp, #0xa0]
    0x1008924cc <+28>:   stp    x29, x30, [sp, #0xb0]
    0x1008924d0 <+32>:   add    x29, sp, #0xb0
    0x1008924d4 <+36>:   mov    x21, x2
    0x1008924d8 <+40>:   mov    x22, x1
    0x1008924dc <+44>:   mov    x20, x0
    0x1008924e0 <+48>:   mov    x0, x30
    0x1008924e4 <+52>:   bl     0x10094bdac    ; symbol stub for: __tsan_func_entry
    0x1008924e8 <+56>:   mov    w0, #0x180 ; =384
    0x1008924ec <+60>:   bl     0x10094bf38    ; symbol stub for: ggml_aligned_malloc
    0x1008924f0 <+64>:   mov    x19, x0
    0x1008924f4 <+68>:   add    x0, x0, #0x70
    0x1008924f8 <+72>:   bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x1008924fc <+76>:   str    x22, [x19, #0x70]
    0x100892500 <+80>:   add    x0, x19, #0x78
    0x100892504 <+84>:   bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x100892508 <+88>:   str    x21, [x19, #0x78]
    0x10089250c <+92>:   add    x0, x19, #0x80
    0x100892510 <+96>:   mov    w1, #0x0 ; =0
    0x100892514 <+100>:  mov    w2, #0x5 ; =5
    0x100892518 <+104>:  bl     0x10094bd88    ; symbol stub for: __tsan_atomic32_store
    0x10089251c <+108>:  add    x0, x19, #0xc0
    0x100892520 <+112>:  mov    w1, #0x0 ; =0
    0x100892524 <+116>:  mov    w2, #0x5 ; =5
    0x100892528 <+120>:  bl     0x10094bd88    ; symbol stub for: __tsan_atomic32_store
    0x10089252c <+124>:  add    x0, x19, #0x100
    0x100892530 <+128>:  mov    w1, #0x0 ; =0
    0x100892534 <+132>:  mov    w2, #0x5 ; =5
    0x100892538 <+136>:  bl     0x10094bd88    ; symbol stub for: __tsan_atomic32_store
    0x10089253c <+140>:  add    x0, x19, #0x140
    0x100892540 <+144>:  mov    w1, #0x0 ; =0
    0x100892544 <+148>:  mov    w2, #0x5 ; =5
    0x100892548 <+152>:  bl     0x10094bd88    ; symbol stub for: __tsan_atomic32_store
    0x10089254c <+156>:  add    x0, x19, #0x144
    0x100892550 <+160>:  mov    w1, #0x0 ; =0
    0x100892554 <+164>:  mov    w2, #0x5 ; =5
    0x100892558 <+168>:  bl     0x10094bda0    ; symbol stub for: __tsan_atomic8_store
    0x10089255c <+172>:  add    x0, x20, #0x20d
    0x100892560 <+176>:  bl     0x10094bde8    ; symbol stub for: __tsan_read1
    0x100892564 <+180>:  add    x0, x19, #0x145
    0x100892568 <+184>:  ldrb   w1, [x20, #0x20d]
    0x10089256c <+188>:  str    x0, [sp, #0x30]
    0x100892570 <+192>:  mov    w2, #0x5 ; =5
    0x100892574 <+196>:  bl     0x10094bda0    ; symbol stub for: __tsan_atomic8_store
    0x100892578 <+200>:  add    x0, x19, #0x148
    0x10089257c <+204>:  mov    w1, #-0x1 ; =-1
    0x100892580 <+208>:  mov    w2, #0x5 ; =5
    0x100892584 <+212>:  bl     0x10094bd88    ; symbol stub for: __tsan_atomic32_store
    0x100892588 <+216>:  add    x0, x19, #0x150
    0x10089258c <+220>:  bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x100892590 <+224>:  str    xzr, [x19, #0x150]
    0x100892594 <+228>:  add    x23, x20, #0x200
    0x100892598 <+232>:  mov    x0, x23
    0x10089259c <+236>:  bl     0x10094be0c    ; symbol stub for: __tsan_read4
    0x1008925a0 <+240>:  ldrsw  x24, [x20, #0x200]
    0x1008925a4 <+244>:  add    x0, x19, #0x158
    0x1008925a8 <+248>:  bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x1008925ac <+252>:  str    w24, [x19, #0x158] ; <---------------- write n_threads_max to offset 0x158
    0x1008925b0 <+256>:  add    x0, x19, #0x15c
    0x1008925b4 <+260>:  mov    x1, x24
    0x1008925b8 <+264>:  mov    w2, #0x5 ; =5
    0x1008925bc <+268>:  bl     0x10094bd88    ; symbol stub for: __tsan_atomic32_store
    0x1008925c0 <+272>:  add    x25, x20, #0x204
    0x1008925c4 <+276>:  add    x21, x19, #0x160
    0x1008925c8 <+280>:  mov    x0, x25
    0x1008925cc <+284>:  bl     0x10094be48    ; symbol stub for: __tsan_unaligned_read8
    0x1008925d0 <+288>:  ldr    d8, [x25]
    0x1008925d4 <+292>:  str    x21, [sp, #0x28]
    0x1008925d8 <+296>:  mov    x0, x21
    0x1008925dc <+300>:  bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x1008925e0 <+304>:  str    d8, [x19, #0x160]
    0x1008925e4 <+308>:  add    x0, x19, #0x168
    0x1008925e8 <+312>:  bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x1008925ec <+316>:  str    wzr, [x19, #0x168]
    0x1008925f0 <+320>:  add    x8, x24, x24, lsl #4
    0x1008925f4 <+324>:  lsl    x25, x8, #5
    0x1008925f8 <+328>:  mov    x0, x25
    0x1008925fc <+332>:  bl     0x10094bf38    ; symbol stub for: ggml_aligned_malloc
    0x100892600 <+336>:  mov    x24, x0
    0x100892604 <+340>:  mov    w1, #0x0 ; =0
    0x100892608 <+344>:  mov    x2, x25
    0x10089260c <+348>:  bl     0x10094bddc    ; symbol stub for: __tsan_memset
    0x100892610 <+352>:  mov    x0, x23
    0x100892614 <+356>:  bl     0x10094be0c    ; symbol stub for: __tsan_read4
    0x100892618 <+360>:  ldr    w27, [x20, #0x200]
    0x10089261c <+364>:  cmp    w27, #0x1
    0x100892620 <+368>:  b.lt   0x100892724    ; <+628>
    0x100892624 <+372>:  cmp    w27, #0x3
    0x100892628 <+376>:  b.hi   0x100892634    ; <+388>
    0x10089262c <+380>:  mov    x21, #0x0 ; =0
    0x100892630 <+384>:  b      0x1008926f0    ; <+576>
    0x100892634 <+388>:  mov    w22, #0x0 ; =0
    0x100892638 <+392>:  mov    x25, #0x0 ; =0
    0x10089263c <+396>:  and    x21, x27, #0x7ffffffc
    0x100892640 <+400>:  str    x27, [sp, #0x20]
    0x100892644 <+404>:  ubfx   x8, x27, #2, #29
    0x100892648 <+408>:  add    x8, x8, x8, lsl #4
    0x10089264c <+412>:  lsl    x8, x8, #7
    0x100892650 <+416>:  stp    x23, x8, [sp, #0x38]
    0x100892654 <+420>:  mov    x26, x24
    0x100892658 <+424>:  add    w24, w22, #0x1
    0x10089265c <+428>:  add    w28, w22, #0x2
    0x100892660 <+432>:  add    w23, w22, #0x3
    0x100892664 <+436>:  add    x27, x26, x25
    0x100892668 <+440>:  add    x0, x27, #0x210
    0x10089266c <+444>:  bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x100892670 <+448>:  str    x19, [x27, #0x210]
    0x100892674 <+452>:  add    x0, x27, #0x430
    0x100892678 <+456>:  bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x10089267c <+460>:  str    x19, [x27, #0x430]
    0x100892680 <+464>:  add    x0, x27, #0x650
    0x100892684 <+468>:  bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x100892688 <+472>:  str    x19, [x27, #0x650]
    0x10089268c <+476>:  add    x0, x27, #0x870
    0x100892690 <+480>:  bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x100892694 <+484>:  str    x19, [x27, #0x870]
    0x100892698 <+488>:  add    x0, x27, #0x218
    0x10089269c <+492>:  bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x1008926a0 <+496>:  str    w22, [x27, #0x218]
    0x1008926a4 <+500>:  add    x0, x27, #0x438
    0x1008926a8 <+504>:  bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x1008926ac <+508>:  str    w24, [x27, #0x438]
    0x1008926b0 <+512>:  add    x0, x27, #0x658
    0x1008926b4 <+516>:  bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x1008926b8 <+520>:  str    w28, [x27, #0x658]
    0x1008926bc <+524>:  add    x0, x27, #0x878
    0x1008926c0 <+528>:  bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x1008926c4 <+532>:  str    w23, [x27, #0x878]
    0x1008926c8 <+536>:  add    x25, x25, #0x880
    0x1008926cc <+540>:  add    w22, w22, #0x4
    0x1008926d0 <+544>:  ldr    x8, [sp, #0x40]
    0x1008926d4 <+548>:  cmp    x8, x25
    0x1008926d8 <+552>:  b.ne   0x100892658    ; <+424>
    0x1008926dc <+556>:  ldr    x27, [sp, #0x20]
    0x1008926e0 <+560>:  cmp    x21, x27
    0x1008926e4 <+564>:  ldr    x23, [sp, #0x38]
    0x1008926e8 <+568>:  mov    x24, x26
    0x1008926ec <+572>:  b.eq   0x100892724    ; <+628>
    0x1008926f0 <+576>:  mov    w8, #0x220 ; =544
    0x1008926f4 <+580>:  umaddl x8, w21, w8, x24
    0x1008926f8 <+584>:  add    x25, x8, #0x218
    0x1008926fc <+588>:  sub    x0, x25, #0x8
    0x100892700 <+592>:  bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x100892704 <+596>:  stur   x19, [x25, #-0x8]
    0x100892708 <+600>:  mov    x0, x25
    0x10089270c <+604>:  bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x100892710 <+608>:  str    w21, [x25]
    0x100892714 <+612>:  add    x21, x21, #0x1
    0x100892718 <+616>:  add    x25, x25, #0x220
    0x10089271c <+620>:  cmp    x27, x21
    0x100892720 <+624>:  b.ne   0x1008926fc    ; <+588>
    0x100892724 <+628>:  add    x0, x19, #0x150
    0x100892728 <+632>:  bl     0x10094becc    ; symbol stub for: __tsan_write8
    0x10089272c <+636>:  str    x24, [x19, #0x150]
    0x100892730 <+640>:  mov    x0, x19
    0x100892734 <+644>:  mov    x1, #0x0 ; =0
    0x100892738 <+648>:  bl     0x10094c1b4    ; symbol stub for: pthread_mutex_init
    0x10089273c <+652>:  add    x0, x19, #0x40
    0x100892740 <+656>:  mov    x1, #0x0 ; =0
    0x100892744 <+660>:  bl     0x10094c178    ; symbol stub for: pthread_cond_init
    0x100892748 <+664>:  mov    x0, x23
    0x10089274c <+668>:  bl     0x10094be0c    ; symbol stub for: __tsan_read4
    0x100892750 <+672>:  ldr    w8, [x23]
    0x100892754 <+676>:  cmp    w8, #0x1
    0x100892758 <+680>:  b.le   0x1008928b0    ; <+1024>
    0x10089275c <+684>:  mov    w25, #0x0 ; =0
    0x100892760 <+688>:  mov    w21, #0x220 ; =544
    0x100892764 <+692>:  mov    w22, #0x1 ; =1
    0x100892768 <+696>:  adrp   x28, 3
    0x10089276c <+700>:  add    x28, x28, #0x228 ; ggml_graph_compute_secondary_thread
    0x100892770 <+704>:  str    x23, [sp, #0x38]
    0x100892774 <+708>:  madd   x26, x22, x21, x24
    0x100892778 <+712>:  add    x27, x26, #0x8
    0x10089277c <+716>:  add    x0, x20, #0x20c
    0x100892780 <+720>:  bl     0x10094bde8    ; symbol stub for: __tsan_read1
    0x100892784 <+724>:  ldrb   w8, [x20, #0x20c]
    0x100892788 <+728>:  tbz    w8, #0x0, 0x1008927e0 ; <+816>
    0x10089278c <+732>:  mov    x23, x19
    0x100892790 <+736>:  mov    x19, x24
    0x100892794 <+740>:  mov    w21, #0x200 ; =512
    0x100892798 <+744>:  mov    x0, x27
    0x10089279c <+748>:  mov    w1, #0x0 ; =0
    0x1008927a0 <+752>:  mov    w2, #0x200 ; =512
    0x1008927a4 <+756>:  bl     0x10094bddc    ; symbol stub for: __tsan_memset
    0x1008927a8 <+760>:  mov    x28, x25
    0x1008927ac <+764>:  sub    w8, w28, #0x200
    0x1008927b0 <+768>:  cmp    w28, #0x1ff
    0x1008927b4 <+772>:  csel   w8, w8, w28, gt
    0x1008927b8 <+776>:  sxtw   x24, w8
    0x1008927bc <+780>:  add    x0, x20, x24
    0x1008927c0 <+784>:  bl     0x10094bde8    ; symbol stub for: __tsan_read1
    0x1008927c4 <+788>:  ldrb   w8, [x20, x24]
    0x1008927c8 <+792>:  cmp    w8, #0x1
    0x1008927cc <+796>:  b.eq   0x1008927f4    ; <+836>
    0x1008927d0 <+800>:  add    w28, w28, #0x1
    0x1008927d4 <+804>:  subs   w21, w21, #0x1
    0x1008927d8 <+808>:  b.ne   0x1008927ac    ; <+764>
    0x1008927dc <+812>:  b      0x100892808    ; <+856>
    0x1008927e0 <+816>:  mov    x0, x27
    0x1008927e4 <+820>:  mov    x1, x20
    0x1008927e8 <+824>:  mov    w2, #0x200 ; =512
    0x1008927ec <+828>:  bl     0x10094bdd0    ; symbol stub for: __tsan_memcpy
    0x1008927f0 <+832>:  b      0x100892820    ; <+880>
    0x1008927f4 <+836>:  add    x0, x27, x24
    0x1008927f8 <+840>:  bl     0x10094be9c    ; symbol stub for: __tsan_write1
    0x1008927fc <+844>:  mov    w8, #0x1 ; =1
    0x100892800 <+848>:  strb   w8, [x27, x24]
    0x100892804 <+852>:  add    w25, w24, #0x1
    0x100892808 <+856>:  mov    x24, x19
    0x10089280c <+860>:  mov    x19, x23
    0x100892810 <+864>:  ldr    x23, [sp, #0x38]
    0x100892814 <+868>:  mov    w21, #0x220 ; =544
    0x100892818 <+872>:  adrp   x28, 3
    0x10089281c <+876>:  add    x28, x28, #0x228 ; ggml_graph_compute_secondary_thread
    0x100892820 <+880>:  mov    x0, x26
    0x100892824 <+884>:  mov    x1, #0x0 ; =0
    0x100892828 <+888>:  mov    x2, x28
    0x10089282c <+892>:  mov    x3, x26
    0x100892830 <+896>:  bl     0x10094c190    ; symbol stub for: pthread_create
    0x100892834 <+900>:  cbnz   w0, 0x100892a2c ; <+1404>
    0x100892838 <+904>:  add    x22, x22, #0x1
    0x10089283c <+908>:  mov    x0, x23
    0x100892840 <+912>:  bl     0x10094be0c    ; symbol stub for: __tsan_read4
    0x100892844 <+916>:  ldrsw  x8, [x23]
    0x100892848 <+920>:  cmp    x22, x8
    0x10089284c <+924>:  b.lt   0x100892774    ; <+708>
    0x100892850 <+928>:  add    x23, x24, #0x8
    0x100892854 <+932>:  add    x0, x20, #0x20c
    0x100892858 <+936>:  bl     0x10094bde8    ; symbol stub for: __tsan_read1
    0x10089285c <+940>:  ldrb   w8, [x20, #0x20c]
    0x100892860 <+944>:  tbz    w8, #0x0, 0x1008928c8 ; <+1048>
    0x100892864 <+948>:  mov    w21, #0x200 ; =512
    0x100892868 <+952>:  mov    x0, x23
    0x10089286c <+956>:  mov    w1, #0x0 ; =0
    0x100892870 <+960>:  mov    w2, #0x200 ; =512
    0x100892874 <+964>:  bl     0x10094bddc    ; symbol stub for: __tsan_memset
    0x100892878 <+968>:  ldr    x24, [sp, #0x30]
    0x10089287c <+972>:  sub    w8, w25, #0x200
    0x100892880 <+976>:  cmp    w25, #0x1ff
    0x100892884 <+980>:  csel   w8, w8, w25, gt
    0x100892888 <+984>:  sxtw   x22, w8
    0x10089288c <+988>:  add    x0, x20, x22
    0x100892890 <+992>:  bl     0x10094bde8    ; symbol stub for: __tsan_read1
    0x100892894 <+996>:  ldrb   w8, [x20, x22]
    0x100892898 <+1000>: cmp    w8, #0x1
    0x10089289c <+1004>: b.eq   0x1008928f4    ; <+1092>
    0x1008928a0 <+1008>: add    w25, w25, #0x1
    0x1008928a4 <+1012>: subs   w21, w21, #0x1
    0x1008928a8 <+1016>: b.ne   0x10089287c    ; <+972>
    0x1008928ac <+1020>: b      0x1008928dc    ; <+1068>
    0x1008928b0 <+1024>: mov    w25, #0x0 ; =0
    0x1008928b4 <+1028>: add    x23, x24, #0x8
    0x1008928b8 <+1032>: add    x0, x20, #0x20c
    0x1008928bc <+1036>: bl     0x10094bde8    ; symbol stub for: __tsan_read1
    0x1008928c0 <+1040>: ldrb   w8, [x20, #0x20c]
    0x1008928c4 <+1044>: tbnz   w8, #0x0, 0x100892864 ; <+948>
    0x1008928c8 <+1048>: mov    x0, x23
    0x1008928cc <+1052>: mov    x1, x20
    0x1008928d0 <+1056>: mov    w2, #0x200 ; =512
    0x1008928d4 <+1060>: bl     0x10094bdd0    ; symbol stub for: __tsan_memcpy
    0x1008928d8 <+1064>: ldr    x24, [sp, #0x30]
    0x1008928dc <+1068>: mov    x0, x24
    0x1008928e0 <+1072>: mov    w1, #0x5 ; =5
    0x1008928e4 <+1076>: bl     0x10094bd94    ; symbol stub for: __tsan_atomic8_load
    0x1008928e8 <+1080>: ldarb  wzr, [x24]
    0x1008928ec <+1084>: tbz    w0, #0x0, 0x100892918 ; <+1128>
    0x1008928f0 <+1088>: b      0x1008929dc    ; <+1324>
    0x1008928f4 <+1092>: add    x0, x23, x22
    0x1008928f8 <+1096>: bl     0x10094be9c    ; symbol stub for: __tsan_write1
    0x1008928fc <+1100>: mov    w8, #0x1 ; =1
    0x100892900 <+1104>: strb   w8, [x23, x22]
    0x100892904 <+1108>: mov    x0, x24
    0x100892908 <+1112>: mov    w1, #0x5 ; =5
    0x10089290c <+1116>: bl     0x10094bd94    ; symbol stub for: __tsan_atomic8_load
    0x100892910 <+1120>: ldarb  wzr, [x24]
    0x100892914 <+1124>: tbnz   w0, #0x0, 0x1008929dc ; <+1324>
    0x100892918 <+1128>: ldr    x20, [sp, #0x28]
    0x10089291c <+1132>: mov    x0, x20
    0x100892920 <+1136>: bl     0x10094be0c    ; symbol stub for: __tsan_read4
    0x100892924 <+1140>: ldr    w22, [x20]
    0x100892928 <+1144>: mov    w20, #0x1 ; =1
    0x10089292c <+1148>: cmp    w22, #0x0
    0x100892930 <+1152>: b.le   0x10089295c    ; <+1196>
    0x100892934 <+1156>: cmp    w22, #0x1
    0x100892938 <+1160>: b.eq   0x10089296c    ; <+1212>
    0x10089293c <+1164>: cmp    w22, #0x2
    0x100892940 <+1168>: b.eq   0x10089297c    ; <+1228>
    0x100892944 <+1172>: cmp    w22, #0x3
    0x100892948 <+1176>: b.ne   0x100892990    ; <+1248>
    0x10089294c <+1180>: add    x0, sp, #0x48
    0x100892950 <+1184>: bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x100892954 <+1188>: mov    w8, #0x5a ; =90
    0x100892958 <+1192>: b      0x100892988    ; <+1240>
    0x10089295c <+1196>: cmn    w22, #0x1
    0x100892960 <+1200>: b.eq   0x100892a08    ; <+1368>
    0x100892964 <+1204>: cbz    w22, 0x1008929dc ; <+1324>
    0x100892968 <+1208>: b      0x100892990    ; <+1248>
    0x10089296c <+1212>: add    x0, sp, #0x48
    0x100892970 <+1216>: bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x100892974 <+1220>: mov    w8, #0x28 ; =40
    0x100892978 <+1224>: b      0x100892988    ; <+1240>
    0x10089297c <+1228>: add    x0, sp, #0x48
    0x100892980 <+1232>: bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x100892984 <+1236>: mov    w8, #0x50 ; =80
    0x100892988 <+1240>: str    w8, [sp, #0x48]
    0x10089298c <+1244>: mov    w20, #0x4 ; =4
    0x100892990 <+1248>: bl     0x10094c1d8    ; symbol stub for: pthread_self
    0x100892994 <+1252>: add    x2, sp, #0x48
    0x100892998 <+1256>: mov    x1, x20
    0x10089299c <+1260>: bl     0x10094c1e4    ; symbol stub for: pthread_setschedparam
    0x1008929a0 <+1264>: cbz    w0, 0x1008929dc ; <+1324>
    0x1008929a4 <+1268>: mov    x20, x0
    0x1008929a8 <+1272>: adrp   x21, 222
    0x1008929ac <+1276>: ldr    x21, [x21, #0x3c0]
    0x1008929b0 <+1280>: mov    x0, x21
    0x1008929b4 <+1284>: bl     0x10094be18    ; symbol stub for: __tsan_read8
    0x1008929b8 <+1288>: ldr    x21, [x21]
    0x1008929bc <+1292>: mov    x0, x20
    0x1008929c0 <+1296>: bl     0x10094c2d4    ; symbol stub for: strerror
    0x1008929c4 <+1300>: stp    x0, x20, [sp, #0x8]
    0x1008929c8 <+1304>: str    x22, [sp]
    0x1008929cc <+1308>: adrp   x1, 204
    0x1008929d0 <+1312>: add    x1, x1, #0xb07 ; "warn: failed to set thread priority %d : %s (%d)\n"
    0x1008929d4 <+1316>: mov    x0, x21
    0x1008929d8 <+1320>: bl     0x10094bf14    ; symbol stub for: fprintf
    0x1008929dc <+1324>: bl     0x10094bdb8    ; symbol stub for: __tsan_func_exit
    0x1008929e0 <+1328>: mov    x0, x19
    0x1008929e4 <+1332>: ldp    x29, x30, [sp, #0xb0]
    0x1008929e8 <+1336>: ldp    x20, x19, [sp, #0xa0]
    0x1008929ec <+1340>: ldp    x22, x21, [sp, #0x90]
    0x1008929f0 <+1344>: ldp    x24, x23, [sp, #0x80]
    0x1008929f4 <+1348>: ldp    x26, x25, [sp, #0x70]
    0x1008929f8 <+1352>: ldp    x28, x27, [sp, #0x60]
    0x1008929fc <+1356>: ldp    d9, d8, [sp, #0x50]
    0x100892a00 <+1360>: add    sp, sp, #0xc0
    0x100892a04 <+1364>: ret
    0x100892a08 <+1368>: add    x0, sp, #0x48
    0x100892a0c <+1372>: bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x100892a10 <+1376>: str    wzr, [sp, #0x48]
    0x100892a14 <+1380>: bl     0x10094c1d8    ; symbol stub for: pthread_self
    0x100892a18 <+1384>: add    x2, sp, #0x48
    0x100892a1c <+1388>: mov    x1, x20
    0x100892a20 <+1392>: bl     0x10094c1e4    ; symbol stub for: pthread_setschedparam
    0x100892a24 <+1396>: cbnz   w0, 0x1008929a4 ; <+1268>
    0x100892a28 <+1400>: b      0x1008929dc    ; <+1324>
    0x100892a2c <+1404>: adrp   x8, 204
    0x100892a30 <+1408>: add    x8, x8, #0xaff ; "rc == 0"
    0x100892a34 <+1412>: adrp   x0, 204
    0x100892a38 <+1416>: add    x0, x0, #0x7c0 ; "/Users/danbev/work/ai/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c"
    0x100892a3c <+1420>: str    x8, [sp]
    0x100892a40 <+1424>: adrp   x2, 204
    0x100892a44 <+1428>: add    x2, x2, #0x7fd ; "GGML_ASSERT(%s) failed"
    0x100892a48 <+1432>: mov    w1, #0xc1a ; =3098
    0x100892a4c <+1436>: bl     0x10094bf20    ; symbol stub for: ggml_abort
```
And we know that we should be looking for a read4.

Read:
```
    0x1008918bc <+40>:  bl     0x10094be0c    ; symbol stub for: __tsan_read4
    0x1008918c0 <+44>:  ldr    w21, [x19, #0x158] ; <----------------  read n_threads_max from offset 0x158
```
Write:
```
    0x1008925a8 <+248>:  bl     0x10094bec0    ; symbol stub for: __tsan_write4
    0x1008925ac <+252>:  str    w24, [x19, #0x158] ; <---------------- write n_threads_max to offset 0x158
```

To debug this futher we need set a watchpoint using an offset as otherwise Address Space Layout Randomization (ASLR)
means the actual memory addresses will keep changing.
```console
WARNING: ThreadSanitizer: data race (pid=76815)
  Write of size 4 at 0x00010bdfc158 by thread T11:
    #0 ggml_threadpool_new_impl <null> (libggml-cpu.dylib:arm64+0x65a8)
```
Get the base address of the libary/image:
```console
(lldb) image list libggml-cpu.dylib
[  0] D559DE63-E171-3E44-AE2C-DE88B07AA6F5 0x000000010088c000 /Users/danbev/work/ai/llama.cpp/build/bin/libggml-cpu.dylib
```
Then use it to set the breakpoints:
```console
(lldb) br set -a 0x000000010088c000+0x65a8
(lldb) br set -a 0x000000010088c000+0x58bc
```
```console
(lldb) br set -a 0x000000010088c000+0x658c
(lldb) br set -a 0x000000010088c000+0x5960
```

I've tried to reproduce this on Linux with the same setting but without success and this only
seems to happen on macOS so it might be a compiler bug/issue.

### ThreadSanitizer on linux
I used the same build configuration as before but when I ran this I got the following error:
```console
```
TODO: add error and workaround when this has been commited. I'm currently on my mac.
