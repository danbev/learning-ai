### llm-chain-llama sender issue
I'm seeing the following issue with 

```consoleVectorStoreTool
$ cd fundamentals/rust/llm-chains-chat-demo
$ env RUST_BACKTRACE=full cargo r --bin llama
```

```console
VectorStoreTool description called!
MultiTool Checking VectorStoreTool against VectorStoreTool
VectorStoreTool description called!
VectorStoreTool description called!
MultiTool Invoking VectorStoreTool
VectorStoreTool invoked_typed: query: "RHSA-2020:5566", limit: 1
Joined text: tracking":{"current_release_date":"2020-12-16T12:43:00Z","generator":{"date":"2023-07-01T04:28:00Z","engine":{"name":"Red Hat SDEngine","version":"3.18.0"}},"id":"RHSA-2020:5566","initial_release_date":"2020-12-16T12:43:00Z","revision_history":[{"date":"2020-12-16T12:43:00Z","number":"1","summary":"Current version"}],"status":"final","version":"1"}},"product_tree":{"branches":[{"branches":[{"branches":[{"category":"product_name","name":"Red Hat Enterprise Linux Client (v. 7)","product":{"name":"Red Hat Enterprise Linux Client (v. 7)","product_id":"7Client-7.9.Z","product_identification_helper":{"cpe":"cpe:/o:redhat:enterprise_linux:7::client"}}},{"category":"product_name","name":"Red Hat Enterprise Linux Client Optional (v. 7)","product":{"name":"Red Hat Enterprise Linux Client Optional (v. 7)","product_id":"7Client-optional-7.9.
Result: <OutputStream>
executor sampling: n_remaining: 2606
executor: str_output1: UnboundedSender { chan: Tx { inner: Chan { tx: Tx { block_tail: 0x5559c6cbab50, tail_position: 0 }, semaphore: Semaphore(1), rx_waker: AtomicWaker, tx_count: 1, rx_fields: "..." } } }
thread 'tokio-runtime-worker' panicked at 'unable to send message', /home/danielbevenius/work/ai/llm-chain/crates/llm-chain-llama/src/executor.rs:161:21
stack backtrace:
   0:     0x5559c5e6bdc1 - std::backtrace_rs::backtrace::libunwind::trace::h6aeaf83abc038fe6
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/../../backtrace/src/backtrace/libunwind.rs:93:5
   1:     0x5559c5e6bdc1 - std::backtrace_rs::backtrace::trace_unsynchronized::h4f9875212db0ad97
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/../../backtrace/src/backtrace/mod.rs:66:5
   2:     0x5559c5e6bdc1 - std::sys_common::backtrace::_print_fmt::h3f820027e9c39d3b
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/sys_common/backtrace.rs:65:5
   3:     0x5559c5e6bdc1 - <std::sys_common::backtrace::_print::DisplayBacktrace as core::fmt::Display>::fmt::hded4932df41373b3
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/sys_common/backtrace.rs:44:22
   4:     0x5559c5e96e3f - core::fmt::rt::Argument::fmt::hc8ead7746b2406d6
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/core/src/fmt/rt.rs:138:9
   5:     0x5559c5e96e3f - core::fmt::write::hb1cb56105a082ad9
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/core/src/fmt/mod.rs:1094:21
   6:     0x5559c5e68dd1 - std::io::Write::write_fmt::h797fda7085c97e57
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/io/mod.rs:1713:15
   7:     0x5559c5e6bbd5 - std::sys_common::backtrace::_print::h492d3c92d7400346
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/sys_common/backtrace.rs:47:5
   8:     0x5559c5e6bbd5 - std::sys_common::backtrace::print::hf74aa2eef05af215
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/sys_common/backtrace.rs:34:9
   9:     0x5559c5e6d077 - std::panicking::default_hook::{{closure}}::h8cad394227ea3de8
  10:     0x5559c5e6ce64 - std::panicking::default_hook::h249cc184fec99a8a
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs:288:9
  11:     0x5559c5e6d52c - std::panicking::rust_panic_with_hook::h82ebcd5d5ed2fad4
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs:705:13
  12:     0x5559c5e6d3e1 - std::panicking::begin_panic_handler::{{closure}}::h810bed8ecbe66f1a
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs:595:13
  13:     0x5559c5e6c1f6 - std::sys_common::backtrace::__rust_end_short_backtrace::h1410008071796261
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/sys_common/backtrace.rs:151:18
  14:     0x5559c5e6d172 - rust_begin_unwind
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs:593:5
  15:     0x5559c4e801e3 - core::panicking::panic_fmt::ha0a42a25e0cf258d
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/core/src/panicking.rs:67:14
  16:     0x5559c4f44573 - llm_chain_llama::executor::Executor::run_model::{{closure}}::{{closure}}::h552e1ee31bd83332
                               at /home/danielbevenius/work/ai/llm-chain/crates/llm-chain-llama/src/executor.rs:161:21
  17:     0x5559c4f522a8 - <tokio::runtime::blocking::task::BlockingTask<T> as core::future::future::Future>::poll::hfc572848f916b43b
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/blocking/task.rs:42:21
  18:     0x5559c4f4db43 - tokio::runtime::task::core::Core<T,S>::poll::{{closure}}::ha18027d9ed950a78
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/core.rs:328:17
  19:     0x5559c4f4d82f - tokio::loom::std::unsafe_cell::UnsafeCell<T>::with_mut::h1817e9ecda95c9d0
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/loom/std/unsafe_cell.rs:16:9
  20:     0x5559c4f4d82f - tokio::runtime::task::core::Core<T,S>::poll::hd59450038f8c4782
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/core.rs:317:13
  21:     0x5559c4f49335 - tokio::runtime::task::harness::poll_future::{{closure}}::h0248bde59f52643f
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/harness.rs:485:19
  22:     0x5559c4f560b4 - <core::panic::unwind_safe::AssertUnwindSafe<F> as core::ops::function::FnOnce<()>>::call_once::hf3d57256580584bb
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/core/src/panic/unwind_safe.rs:271:9
  23:     0x5559c4f5587c - std::panicking::try::do_call::h63af09fb9a6ab113
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs:500:40
  24:     0x5559c4f55f3b - __rust_try
  25:     0x5559c4f55577 - std::panicking::try::hbc683139bcca7479
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs:464:19
  26:     0x5559c4f53a7b - std::panic::catch_unwind::h5e20edfbdaf6ff79
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panic.rs:142:14
  27:     0x5559c4f4910f - tokio::runtime::task::harness::poll_future::hb4173137cab73620
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/harness.rs:473:18
  28:     0x5559c4f495c9 - tokio::runtime::task::harness::Harness<T,S>::poll_inner::h0b903119ec99396d
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/harness.rs:208:27
  29:     0x5559c4f49c97 - tokio::runtime::task::harness::Harness<T,S>::poll::h9fd75d6a71125a06
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/harness.rs:153:15
  30:     0x5559c4f52a0d - tokio::runtime::task::raw::poll::h592c47a2582208b4
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/raw.rs:276:5
  31:     0x5559c5d83b67 - tokio::runtime::task::raw::RawTask::poll::h20760f8611641e51
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/raw.rs:200:18
  32:     0x5559c5d578b7 - tokio::runtime::task::UnownedTask<S>::run::hcc05dddb01fb8c1c
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/task/mod.rs:445:9
  33:     0x5559c5d57c57 - tokio::runtime::blocking::pool::Task::run::h61aa98cbb3afd7eb
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/blocking/pool.rs:159:9
  34:     0x5559c5d5aac4 - tokio::runtime::blocking::pool::Inner::run::h1f771eea105e2b44
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/blocking/pool.rs:513:17
  35:     0x5559c5d59fe4 - tokio::runtime::blocking::pool::Spawner::spawn_thread::{{closure}}::he73c09d4fe4e6bcf
                               at /home/danielbevenius/.cargo/registry/src/index.crates.io-6f17d22bba15001f/tokio-1.33.0/src/runtime/blocking/pool.rs:471:13
  36:     0x5559c5d42049 - std::sys_common::backtrace::__rust_begin_short_backtrace::ha110e777d7674b34
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/sys_common/backtrace.rs:135:18
  37:     0x5559c5d801c2 - std::thread::Builder::spawn_unchecked_::{{closure}}::{{closure}}::h1fe19b51b81a1ed6
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/thread/mod.rs:529:17
  38:     0x5559c5d52672 - <core::panic::unwind_safe::AssertUnwindSafe<F> as core::ops::function::FnOnce<()>>::call_once::h6492171f8ed3c02c
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/core/src/panic/unwind_safe.rs:271:9
  39:     0x5559c5d8eb0b - std::panicking::try::do_call::h6a1239a8c3a61756
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs:500:40
  40:     0x5559c5d8f70b - __rust_try
  41:     0x5559c5d8e771 - std::panicking::try::he7db911d0bc4aa6b
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panicking.rs:464:19
  42:     0x5559c5d7ffd2 - std::panic::catch_unwind::h24e8f9039d9d7c2f
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/panic.rs:142:14
  43:     0x5559c5d7ffd2 - std::thread::Builder::spawn_unchecked_::{{closure}}::h738f0e2708d25bc4
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/thread/mod.rs:528:30
  44:     0x5559c5d6581f - core::ops::function::FnOnce::call_once{{vtable.shim}}::h0c3e5e95bd16a3f8
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/core/src/ops/function.rs:250:5
  45:     0x5559c5e72045 - <alloc::boxed::Box<F,A> as core::ops::function::FnOnce<Args>>::call_once::h9adfc2ae43657457
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/alloc/src/boxed.rs:1985:9
  46:     0x5559c5e72045 - <alloc::boxed::Box<F,A> as core::ops::function::FnOnce<Args>>::call_once::h14fefbfa7b574396
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/alloc/src/boxed.rs:1985:9
  47:     0x5559c5e72045 - std::sys::unix::thread::Thread::new::thread_start::ha211bb47f6f5cedc
                               at /rustc/8ede3aae28fe6e4d52b38157d7bfe0d3bceef225/library/std/src/sys/unix/thread.rs:108:17
  48:     0x7f79e30ae19d - start_thread
  49:     0x7f79e312fc40 - clone3
  50:                0x0 - <unknown>
```

