## baby-llama issue

I ran into this when investigating an issue related to the baby-llama example in
llama.cpp. When stepping through the code to understand the original issue I
notice the following in `init_mode`:
```console
$ gdb --args ./llama-baby-llama
Reading symbols from ./llama-baby-llama...
(gdb) br init_model
Breakpoint 1 at 0x251767: file examples/baby-llama/baby-llama.cpp, line 190.
(gdb) r
...
(gdb)
204	    model->layers.resize(n_layer);
```
Now, if we inspect the size of `model->layers` we see that it is 0:
```console
(gdb) p model->layers.size()
$1 = 0
```
And also `n_layer` is 1:
```console
(gdb) p n_layer
$3 = 1
```
And we can inspect the type of `model->layers`:
```console
(gdb) ptype model->layers
type = std::vector<llama_layer>
```

Now if we step over the resize function we will see something interesting:
```console
(gdb) p model->layers.size()
$2 = 12
```
I also added two print statements to show the size of `n_layer` and
`model->layers.size()`:
```console
n_layer: 1
layers.size(): 2049638230412172414
```
After some digging into this a little and first thinking that this was something
to do with the `resize` function in combination with the `llama_layer` I later
realized that this was actually a type conflict. There is a `llama_layer` in
llama.cpp and this object file is compiled into the `llama-baby-llama` binary:
```console
/usr/bin/ccache c++ -std=c++11 -fPIC -O0 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp  -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -D_GLIBCXX_ASSERTIONS -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE  ggml/src/llamafile/sgemm.o ggml/src/ggml.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-quants.o ggml/src/ggml-aarch64.o src/llama.o src/llama-vocab.o src/llama-grammar.o src/llama-sampling.o src/unicode.o src/unicode-data.o common/common.o common/arg.o common/log.o common/console.o common/ngram-cache.o common/sampling.o common/train.o common/build-info.o common/json-schema-to-grammar.o examples/baby-llama/baby-llama.o -o llama-baby-llama -g
```
This could be worked around by renaming the `llama_layer` baby-llama.cpp to
use `baby_llama_layer` instead, use a namespace in baby-llama.cpp, or remove
the object files that are not needed for the baby-llama example. I think the
intent of the baby-llama example is to be a standalone example, not actually
using llama.cpp, so the latter option might be the best. Actually it looks like
the baby-llama example uses `train.h` and might need llama.cpp indirectly, so
perhaps just renaming the `llama_layer` in baby-llama.cpp is the best option.

Renaming the `llama_layer` in baby-llama.cpp to `baby_llama_layer`:
```console
(gdb) p model->layers.size()
$1 = 0
(gdb) ptype model->layers
type = std::vector<baby_llama_layer>
(gdb) p model->layers.size()
$2 = 1
```

### Reproducing the issue

The following can be produced on the current master branch:
```console
The issue here can be reproduce using the `make` build with the following steps:
```console
$ git co master
$ make clean
$ make llama-baby-llama -j8 LLAMA_DEBUG=1
```
Running the `llama-baby-llama` executable results in the following error:
```console
$ ./llama-baby-llama
Segmentation fault (core dumped)
```
Running this in the debugger shows the following:
```console
$ gdb --args ./llama-baby-llama
Reading symbols from ./llama-baby-llama...
(gdb) r
Starting program: /home/danbev/work/ai/llama.cpp/llama-baby-llama
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

Program received signal SIGSEGV, Segmentation fault.
__memset_avx2_unaligned_erms () at ../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S:328
warning: 328	../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S: No such file or directory
(gdb) bt
#0  __memset_avx2_unaligned_erms () at ../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S:328
#1  0x00005555556fa3ae in llama_model::llama_model (this=0x7fffffffd910) at src/llama.cpp:2842
#2  0x00005555558eaba8 in main (argc=1, argv=0x7fffffffdb28) at examples/baby-llama/baby-llama.cpp:1447
(gdb) up
#1  0x00005555556fa3ae in llama_model::llama_model (this=0x7fffffffd910) at src/llama.cpp:2842
2842	struct llama_model {
```

Trying this using the `cmake` build produces:
```console
$ rm -rf build
$ cmake -S . -B build -DLLAMA_CURL=ON
$ cmake --build build
```
And running `llama-baby-llama`:
```console
$ ./build/bin/llama-baby-llama
init model
init_kv_cache
/home/danbev/work/ai/llama.cpp/ggml/src/ggml.c:6815: GGML_ASSERT(false && "backwards pass not implemented") failed
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```
This is a different issue and there is no segfault. Lets try a debug build using
cmake and see if that produces a different result:
```console
$ ./build/bin/llama-baby-llama
init model
init_kv_cache
/home/danbev/work/ai/llama.cpp/ggml/src/ggml.c:6815: GGML_ASSERT(false && "backwards pass not implemented") failed
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```
This produces the same result which is different from the cmake results where
we are seeing the segfault and also the issue with the `llama_layer` struct.

So what is different in the case of `make` and `cmake`?  

One this is that the binary build by cmake is a is dynamically linked:
```console
$ ldd build/bin/llama-baby-llama
	linux-vdso.so.1 (0x00007ffcb89eb000)
	libllama.so => /home/danbev/work/ai/llama.cpp/build/src/libllama.so (0x00007525a7e00000)
	libggml.so => /home/danbev/work/ai/llama.cpp/build/ggml/src/libggml.so (0x00007525a7c72000)
	libcurl.so.4 => /lib/x86_64-linux-gnu/libcurl.so.4 (0x00007525a82e6000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007525a7800000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007525a7b89000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007525a7b5c000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007525a7400000)
	/lib64/ld-linux-x86-64.so.2 (0x00007525a84f1000)
	libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007525a7b05000)
	libnghttp2.so.14 => /lib/x86_64-linux-gnu/libnghttp2.so.14 (0x00007525a7ada000)
	libidn2.so.0 => /lib/x86_64-linux-gnu/libidn2.so.0 (0x00007525a7ab8000)
	librtmp.so.1 => /lib/x86_64-linux-gnu/librtmp.so.1 (0x00007525a7a9a000)
	libssh.so.4 => /lib/x86_64-linux-gnu/libssh.so.4 (0x00007525a778f000)
	libpsl.so.5 => /lib/x86_64-linux-gnu/libpsl.so.5 (0x00007525a7a86000)
	libssl.so.3 => /lib/x86_64-linux-gnu/libssl.so.3 (0x00007525a76e5000)
	libcrypto.so.3 => /lib/x86_64-linux-gnu/libcrypto.so.3 (0x00007525a6e00000)
	libgssapi_krb5.so.2 => /lib/x86_64-linux-gnu/libgssapi_krb5.so.2 (0x00007525a7691000)
	libldap.so.2 => /lib/x86_64-linux-gnu/libldap.so.2 (0x00007525a7634000)
	liblber.so.2 => /lib/x86_64-linux-gnu/liblber.so.2 (0x00007525a7624000)
	libzstd.so.1 => /lib/x86_64-linux-gnu/libzstd.so.1 (0x00007525a7346000)
	libbrotlidec.so.1 => /lib/x86_64-linux-gnu/libbrotlidec.so.1 (0x00007525a7616000)
	libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007525a732a000)
	libunistring.so.5 => /lib/x86_64-linux-gnu/libunistring.so.5 (0x00007525a6c53000)
	libgnutls.so.30 => /lib/x86_64-linux-gnu/libgnutls.so.30 (0x00007525a6a59000)
	libhogweed.so.6 => /lib/x86_64-linux-gnu/libhogweed.so.6 (0x00007525a6a11000)
	libnettle.so.8 => /lib/x86_64-linux-gnu/libnettle.so.8 (0x00007525a69bc000)
	libgmp.so.10 => /lib/x86_64-linux-gnu/libgmp.so.10 (0x00007525a6938000)
	libkrb5.so.3 => /lib/x86_64-linux-gnu/libkrb5.so.3 (0x00007525a686f000)
	libk5crypto.so.3 => /lib/x86_64-linux-gnu/libk5crypto.so.3 (0x00007525a6843000)
	libcom_err.so.2 => /lib/x86_64-linux-gnu/libcom_err.so.2 (0x00007525a7a7e000)
	libkrb5support.so.0 => /lib/x86_64-linux-gnu/libkrb5support.so.0 (0x00007525a731d000)
	libsasl2.so.2 => /lib/x86_64-linux-gnu/libsasl2.so.2 (0x00007525a6829000)
	libbrotlicommon.so.1 => /lib/x86_64-linux-gnu/libbrotlicommon.so.1 (0x00007525a6806000)
	libp11-kit.so.0 => /lib/x86_64-linux-gnu/libp11-kit.so.0 (0x00007525a6662000)
	libtasn1.so.6 => /lib/x86_64-linux-gnu/libtasn1.so.6 (0x00007525a664c000)
	libkeyutils.so.1 => /lib/x86_64-linux-gnu/libkeyutils.so.1 (0x00007525a7316000)
	libresolv.so.2 => /lib/x86_64-linux-gnu/libresolv.so.2 (0x00007525a6639000)
	libffi.so.8 => /lib/x86_64-linux-gnu/libffi.so.8 (0x00007525a662d000)
```
Lets check `libllama.so` for the `llama_layer` struct:
```console
$ nm -D build/src/libllama.so | c++filt | grep llama_layer
000000000036914a W llama_layer::llama_layer()
000000000036914a W llama_layer::llama_layer()
000000000035879c W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_get_Tp_allocator() const
000000000034abac W std::vector<llama_layer, std::allocator<llama_layer> >::_M_check_len(unsigned long, char const*) const
000000000033a3da W std::vector<llama_layer, std::allocator<llama_layer> >::size() const
000000000034ab58 W std::vector<llama_layer, std::allocator<llama_layer> >::max_size() const
0000000000329ab6 W std::vector<llama_layer, std::allocator<llama_layer> >::operator[](unsigned long) const
0000000000351eed W void std::_Destroy_aux<true>::__destroy<llama_layer*>(llama_layer*, llama_layer*)
000000000034acb0 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_allocate(unsigned long)
000000000032aa72 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::_Vector_impl()
000000000032aa72 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::_Vector_impl()
000000000031ad16 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::~_Vector_impl()
000000000031ad16 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::~_Vector_impl()
000000000033d20c W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_deallocate(llama_layer*, unsigned long)
000000000033d1ca W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl_data::_Vector_impl_data()
000000000033d1ca W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl_data::_Vector_impl_data()
000000000032ffde W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_get_Tp_allocator()
000000000031ad3e W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_base()
000000000031ad3e W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_base()
000000000032aaa4 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::~_Vector_base()
000000000032aaa4 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::~_Vector_base()
0000000000359b62 W std::__new_allocator<llama_layer>::deallocate(llama_layer*, unsigned long)
0000000000362a8a W std::__new_allocator<llama_layer>::allocate(unsigned long, void const*)
000000000033d1fc W std::__new_allocator<llama_layer>::~__new_allocator()
000000000033d1fc W std::__new_allocator<llama_layer>::~__new_allocator()
0000000000362a3d W llama_layer* std::__uninitialized_default_n_1<false>::__uninit_default_n<llama_layer*, unsigned long>(llama_layer*, unsigned long)
000000000035871b W std::vector<llama_layer, std::allocator<llama_layer> >::_S_max_size(std::allocator<llama_layer> const&)
000000000034acfd W std::vector<llama_layer, std::allocator<llama_layer> >::_S_relocate(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
000000000033a63a W std::vector<llama_layer, std::allocator<llama_layer> >::_M_erase_at_end(llama_layer*)
000000000033a40c W std::vector<llama_layer, std::allocator<llama_layer> >::_M_default_append(unsigned long)
0000000000328bf4 W std::vector<llama_layer, std::allocator<llama_layer> >::resize(unsigned long)
000000000031ad5e W std::vector<llama_layer, std::allocator<llama_layer> >::vector()
000000000031ad5e W std::vector<llama_layer, std::allocator<llama_layer> >::vector()
0000000000321bae W std::vector<llama_layer, std::allocator<llama_layer> >::~vector()
0000000000321bae W std::vector<llama_layer, std::allocator<llama_layer> >::~vector()
0000000000328da2 W std::vector<llama_layer, std::allocator<llama_layer> >::operator[](unsigned long)
00000000003691b3 W void std::_Construct<llama_layer>(llama_layer*)
0000000000369137 W llama_layer* std::__addressof<llama_layer>(llama_layer&)
0000000000362af8 W llama_layer* std::__niter_base<llama_layer*>(llama_layer*)
00000000003587db W llama_layer* std::__relocate_a<llama_layer*, llama_layer*, std::allocator<llama_layer> >(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
0000000000362b0a W llama_layer* std::__relocate_a_1<llama_layer*, llama_layer*, std::allocator<llama_layer> >(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
00000000003691f8 W void std::__relocate_object_a<llama_layer, llama_layer, std::allocator<llama_layer> >(llama_layer*, llama_layer*, std::allocator<llama_layer>&)
00000000003587ae W llama_layer* std::__uninitialized_default_n<llama_layer*, unsigned long>(llama_layer*, unsigned long)
000000000034ab7e W llama_layer* std::__uninitialized_default_n_a<llama_layer*, unsigned long, llama_layer>(llama_layer*, unsigned long, std::allocator<llama_layer>&)
000000000036cd76 W std::remove_reference<llama_layer&>::type&& std::move<llama_layer&>(llama_layer&)
000000000036ed59 W llama_layer&& std::forward<llama_layer>(std::remove_reference<llama_layer>::type&)
00000000003420fe W void std::_Destroy<llama_layer*>(llama_layer*, llama_layer*)
```
Notice that these symbols (like constructors for `llama_layers`) are `weak`
symbols (W) which allows the multiple definitions accross different compilation
units. There will not be an error raise in this case.

Lets also inspect the `llama-baby-llama` binary:
```console
$ nm -D build/bin/llama-baby-llama | c++filt | grep llama_layer 
000000000005e598 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_get_Tp_allocator() const
000000000005dfa4 W std::vector<llama_layer, std::allocator<llama_layer> >::_M_check_len(unsigned long, char const*) const
000000000005d602 W std::vector<llama_layer, std::allocator<llama_layer> >::size() const
000000000005df50 W std::vector<llama_layer, std::allocator<llama_layer> >::max_size() const
000000000005e7ce W void std::_Destroy_aux<true>::__destroy<llama_layer*>(llama_layer*, llama_layer*)
000000000005e0a8 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_allocate(unsigned long)
000000000005d17c W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::_Vector_impl()
000000000005d17c W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::_Vector_impl()
000000000005cd96 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::~_Vector_impl()
000000000005cd96 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::~_Vector_impl()
000000000005dc6a W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_deallocate(llama_layer*, unsigned long)
000000000005dc28 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl_data::_Vector_impl_data()
000000000005dc28 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl_data::_Vector_impl_data()
000000000005dcbc W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_get_Tp_allocator()
000000000005cdbe W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_base()
000000000005cdbe W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_base()
000000000005d1ae W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::~_Vector_base()
000000000005d1ae W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::~_Vector_base()
000000000005e790 W std::__new_allocator<llama_layer>::deallocate(llama_layer*, unsigned long)
000000000005e9ca W std::__new_allocator<llama_layer>::allocate(unsigned long, void const*)
000000000005dc5a W std::__new_allocator<llama_layer>::~__new_allocator()
000000000005dc5a W std::__new_allocator<llama_layer>::~__new_allocator()
000000000005e517 W std::vector<llama_layer, std::allocator<llama_layer> >::_S_max_size(std::allocator<llama_layer> const&)
000000000005e0f5 W std::vector<llama_layer, std::allocator<llama_layer> >::_S_relocate(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
000000000005d884 W std::vector<llama_layer, std::allocator<llama_layer> >::_M_erase_at_end(llama_layer*)
000000000005d634 W std::vector<llama_layer, std::allocator<llama_layer> >::_M_default_append(unsigned long)
000000000005cf90 W std::vector<llama_layer, std::allocator<llama_layer> >::resize(unsigned long)
000000000005cdde W std::vector<llama_layer, std::allocator<llama_layer> >::vector()
000000000005cdde W std::vector<llama_layer, std::allocator<llama_layer> >::vector()
000000000005d20a W std::vector<llama_layer, std::allocator<llama_layer> >::~vector()
000000000005d20a W std::vector<llama_layer, std::allocator<llama_layer> >::~vector()
000000000005d026 W std::vector<llama_layer, std::allocator<llama_layer> >::operator[](unsigned long)
000000000005ed3f W void std::_Construct<llama_layer>(llama_layer*)
000000000005ed2d W llama_layer* std::__addressof<llama_layer>(llama_layer&)
000000000005ea3f W llama_layer* std::__niter_base<llama_layer*>(llama_layer*)
000000000005e5d7 W llama_layer* std::__relocate_a<llama_layer*, llama_layer*, std::allocator<llama_layer> >(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
000000000005e5aa W llama_layer* std::__uninitialized_default_n<llama_layer*, unsigned long>(llama_layer*, unsigned long)
000000000005df76 W llama_layer* std::__uninitialized_default_n_a<llama_layer*, unsigned long, llama_layer>(llama_layer*, unsigned long, std::allocator<llama_layer>&)
000000000005e36c W void std::_Destroy<llama_layer*>(llama_layer*, llama_layer*)
```
Now, my understanding of how the dynamic linker (ld.so) will handle this is that
it will look for symbols in the llama-baby-llama executable first, where it will
find the symbol and not look any further. There will be no warning or error like
we mentioned above.

While the `make` build is statically linked:
```console
$ ldd ./llama-baby-llama
	linux-vdso.so.1 (0x00007ffcf0755000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x0000742771000000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x0000742771a22000)
	libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007427719cb000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x000074277199e000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x0000742770c00000)
	/lib64/ld-linux-x86-64.so.2 (0x0000742771b1e000)
```
And the `llama_layer` symbols are:
```console
$ nm ./llama-baby-llama | c++filt | grep llama_layer
00000000001409d0 t llm_build_rwkv6_time_mix(llama_context&, ggml_context*, llama_layer const*, ggml_tensor*, ggml_tensor*, ggml_tensor**)
0000000000141541 t llm_build_rwkv6_channel_mix(llama_context&, ggml_context*, llama_layer const*, ggml_tensor*, ggml_tensor*)
00000000001ff63a W llama_layer::llama_layer()
00000000001ff63a W llama_layer::llama_layer()
00000000001eda86 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_get_Tp_allocator() const
0000000000397f8c W std::_Vector_base<llama_layer_lora, std::allocator<llama_layer_lora> >::_M_get_Tp_allocator() const
000000000020b178 W std::move_iterator<llama_layer*>::base() const
0000000000207c88 W std::move_iterator<llama_layer*>::operator*() const
00000000003989d2 W std::move_iterator<llama_layer_lora*>::base() const
00000000001df26c W std::vector<llama_layer, std::allocator<llama_layer> >::_M_check_len(unsigned long, char const*) const
00000000001cbf4a W std::vector<llama_layer, std::allocator<llama_layer> >::size() const
00000000001df218 W std::vector<llama_layer, std::allocator<llama_layer> >::max_size() const
00000000001b6396 W std::vector<llama_layer, std::allocator<llama_layer> >::operator[](unsigned long) const
0000000000397cda W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::_M_check_len(unsigned long, char const*) const
0000000000397732 W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::size() const
0000000000397c74 W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::max_size() const
0000000000398c04 W void std::__copy_move<true, false, std::random_access_iterator_tag>::__assign_one<llama_layer, llama_layer>(llama_layer*, llama_layer*)
0000000000398c47 W void std::__copy_move<true, false, std::random_access_iterator_tag>::__assign_one<llama_layer_lora, llama_layer_lora>(llama_layer_lora*, llama_layer_lora*)
0000000000398aa2 W llama_layer* std::__copy_move<true, true, std::random_access_iterator_tag>::__copy_m<llama_layer, llama_layer>(llama_layer*, llama_layer*, llama_layer*)
0000000000398b4d W llama_layer_lora* std::__copy_move<true, true, std::random_access_iterator_tag>::__copy_m<llama_layer_lora, llama_layer_lora>(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*)
00000000001e730f W void std::_Destroy_aux<true>::__destroy<llama_layer*>(llama_layer*, llama_layer*)
00000000003982dc W void std::_Destroy_aux<true>::__destroy<llama_layer_lora*>(llama_layer_lora*, llama_layer_lora*)
00000000001df370 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_allocate(unsigned long)
00000000001b74be W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::_Vector_impl()
00000000001b74be W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::_Vector_impl()
00000000001a5fb8 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::~_Vector_impl()
00000000001a5fb8 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl::~_Vector_impl()
00000000001cf932 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_deallocate(llama_layer*, unsigned long)
00000000001cf8f0 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl_data::_Vector_impl_data()
00000000001cf8f0 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_impl_data::_Vector_impl_data()
00000000001be4e4 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_M_get_Tp_allocator()
00000000001a5fe0 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_base()
00000000001a5fe0 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::_Vector_base()
00000000001b74f0 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::~_Vector_base()
00000000001b74f0 W std::_Vector_base<llama_layer, std::allocator<llama_layer> >::~_Vector_base()
0000000000397dde W std::_Vector_base<llama_layer_lora, std::allocator<llama_layer_lora> >::_M_allocate(unsigned long)
0000000000397e2c W std::_Vector_base<llama_layer_lora, std::allocator<llama_layer_lora> >::_M_deallocate(llama_layer_lora*, unsigned long)
0000000000397c9a W std::_Vector_base<llama_layer_lora, std::allocator<llama_layer_lora> >::_M_get_Tp_allocator()
00000000001f845a W std::move_iterator<llama_layer*>::move_iterator(llama_layer*)
00000000001f845a W std::move_iterator<llama_layer*>::move_iterator(llama_layer*)
0000000000207c60 W std::move_iterator<llama_layer*>::operator++()
0000000000398276 W std::move_iterator<llama_layer_lora*>::move_iterator(llama_layer_lora*)
0000000000398276 W std::move_iterator<llama_layer_lora*>::move_iterator(llama_layer_lora*)
00000000001eefe0 W std::__new_allocator<llama_layer>::deallocate(llama_layer*, unsigned long)
00000000001f8386 W std::__new_allocator<llama_layer>::allocate(unsigned long, void const*)
00000000001cf922 W std::__new_allocator<llama_layer>::~__new_allocator()
00000000001cf922 W std::__new_allocator<llama_layer>::~__new_allocator()
00000000003981e8 W std::__new_allocator<llama_layer_lora>::deallocate(llama_layer_lora*, unsigned long)
000000000039816e W std::__new_allocator<llama_layer_lora>::allocate(unsigned long, void const*)
00000000001ff78b W llama_layer* std::__uninitialized_copy<false>::__uninit_copy<std::move_iterator<llama_layer*>, llama_layer*>(std::move_iterator<llama_layer*>, std::move_iterator<llama_layer*>, llama_layer*)
00000000003983b6 W llama_layer* std::__uninitialized_copy<true>::__uninit_copy<std::move_iterator<llama_layer*>, llama_layer*>(std::move_iterator<llama_layer*>, std::move_iterator<llama_layer*>, llama_layer*)
000000000039857b W llama_layer_lora* std::__uninitialized_copy<true>::__uninit_copy<std::move_iterator<llama_layer_lora*>, llama_layer_lora*>(std::move_iterator<llama_layer_lora*>, std::move_iterator<llama_layer_lora*>, llama_layer_lora*)
00000000001f8339 W llama_layer* std::__uninitialized_default_n_1<false>::__uninit_default_n<llama_layer*, unsigned long>(llama_layer*, unsigned long)
00000000003980ac W llama_layer* std::__uninitialized_default_n_1<true>::__uninit_default_n<llama_layer*, unsigned long>(llama_layer*, unsigned long)
000000000039810d W llama_layer_lora* std::__uninitialized_default_n_1<true>::__uninit_default_n<llama_layer_lora*, unsigned long>(llama_layer_lora*, unsigned long)
00000000001eda05 W std::vector<llama_layer, std::allocator<llama_layer> >::_S_max_size(std::allocator<llama_layer> const&)
00000000001df3bd W std::vector<llama_layer, std::allocator<llama_layer> >::_S_relocate(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
00000000001edac5 W std::vector<llama_layer, std::allocator<llama_layer> >::_S_do_relocate(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&, std::integral_constant<bool, true>)
00000000001cc39a W std::vector<llama_layer, std::allocator<llama_layer> >::_M_erase_at_end(llama_layer*)
00000000001cbf7c W std::vector<llama_layer, std::allocator<llama_layer> >::_S_use_relocate()
00000000001cbfc2 W std::vector<llama_layer, std::allocator<llama_layer> >::_M_default_append(unsigned long)
00000000001cbfb2 W std::vector<llama_layer, std::allocator<llama_layer> >::_S_nothrow_relocate(std::integral_constant<bool, true>)
00000000001b5244 W std::vector<llama_layer, std::allocator<llama_layer> >::resize(unsigned long)
00000000001a6000 W std::vector<llama_layer, std::allocator<llama_layer> >::vector()
00000000001a6000 W std::vector<llama_layer, std::allocator<llama_layer> >::vector()
00000000001ad762 W std::vector<llama_layer, std::allocator<llama_layer> >::~vector()
00000000001ad762 W std::vector<llama_layer, std::allocator<llama_layer> >::~vector()
00000000001b53f2 W std::vector<llama_layer, std::allocator<llama_layer> >::operator[](unsigned long)
0000000000397f0b W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::_S_max_size(std::allocator<llama_layer_lora> const&)
0000000000397e7e W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::_S_relocate(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*, std::allocator<llama_layer_lora>&)
0000000000397fcb W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::_S_do_relocate(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*, std::allocator<llama_layer_lora>&, std::integral_constant<bool, true>)
0000000000397bec W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::_M_erase_at_end(llama_layer_lora*)
0000000000397764 W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::_S_use_relocate()
00000000003977aa W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::_M_default_append(unsigned long)
000000000039779a W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::_S_nothrow_relocate(std::integral_constant<bool, true>)
00000000003975fe W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::resize(unsigned long)
000000000039769a W std::vector<llama_layer_lora, std::allocator<llama_layer_lora> >::operator[](unsigned long)
00000000001ff6a3 W void std::_Construct<llama_layer>(llama_layer*)
0000000000207c9d W void std::_Construct<llama_layer, llama_layer>(llama_layer*, llama_layer&&)
00000000003983f9 W void std::_Construct<llama_layer_lora>(llama_layer_lora*)
00000000003985ac W llama_layer* std::__fill_n_a<llama_layer*, unsigned long, llama_layer>(llama_layer*, unsigned long, llama_layer const&, std::random_access_iterator_tag)
000000000039866e W llama_layer_lora* std::__fill_n_a<llama_layer_lora*, unsigned long, llama_layer_lora>(llama_layer_lora*, unsigned long, llama_layer_lora const&, std::random_access_iterator_tag)
00000000001ff627 W llama_layer* std::__addressof<llama_layer>(llama_layer&)
00000000003983e7 W llama_layer_lora* std::__addressof<llama_layer_lora>(llama_layer_lora&)
000000000039876e W decltype (__miter_base(({parm#1}.base)())) std::__miter_base<llama_layer*>(std::move_iterator<llama_layer*>)
000000000039891b W llama_layer* std::__miter_base<llama_layer*>(llama_layer*)
0000000000398834 W decltype (__miter_base(({parm#1}.base)())) std::__miter_base<llama_layer_lora*>(std::move_iterator<llama_layer_lora*>)
00000000003989e7 W llama_layer_lora* std::__miter_base<llama_layer_lora*>(llama_layer_lora*)
00000000001ff6ec W llama_layer* std::__niter_base<llama_layer*>(llama_layer*)
00000000003984c1 W llama_layer_lora* std::__niter_base<llama_layer_lora*>(llama_layer_lora*)
000000000039895e W llama_layer* std::__niter_wrap<llama_layer*>(llama_layer* const&, llama_layer*)
0000000000398a2a W llama_layer_lora* std::__niter_wrap<llama_layer_lora*>(llama_layer_lora* const&, llama_layer_lora*)
00000000001f83f3 W llama_layer* std::__relocate_a<llama_layer*, llama_layer*, std::allocator<llama_layer> >(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
000000000039820f W llama_layer_lora* std::__relocate_a<llama_layer_lora*, llama_layer_lora*, std::allocator<llama_layer_lora> >(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*, std::allocator<llama_layer_lora>&)
0000000000398794 W llama_layer* std::__copy_move_a<true, llama_layer*, llama_layer*>(llama_layer*, llama_layer*, llama_layer*)
000000000039885a W llama_layer_lora* std::__copy_move_a<true, llama_layer_lora*, llama_layer_lora*>(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*)
000000000039892d W llama_layer* std::__copy_move_a1<true, llama_layer*, llama_layer*>(llama_layer*, llama_layer*, llama_layer*)
00000000003989f9 W llama_layer_lora* std::__copy_move_a1<true, llama_layer_lora*, llama_layer_lora*>(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*)
0000000000398a40 W llama_layer* std::__copy_move_a2<true, llama_layer*, llama_layer*>(llama_layer*, llama_layer*, llama_layer*)
0000000000398a71 W llama_layer_lora* std::__copy_move_a2<true, llama_layer_lora*, llama_layer_lora*>(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*)
000000000039832c W std::enable_if<std::__is_bitwise_relocatable<llama_layer, void>::value, llama_layer*>::type std::__relocate_a_1<llama_layer, llama_layer>(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
00000000003984d3 W std::enable_if<std::__is_bitwise_relocatable<llama_layer_lora, void>::value, llama_layer_lora*>::type std::__relocate_a_1<llama_layer_lora, llama_layer_lora>(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*, std::allocator<llama_layer_lora>&)
00000000001ff6fe W llama_layer* std::__relocate_a_1<llama_layer*, llama_layer*, std::allocator<llama_layer> >(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
0000000000204773 W llama_layer* std::__do_uninit_copy<std::move_iterator<llama_layer*>, llama_layer*>(std::move_iterator<llama_layer*>, std::move_iterator<llama_layer*>, llama_layer*)
00000000001f8487 W llama_layer* std::uninitialized_copy<std::move_iterator<llama_layer*>, llama_layer*>(std::move_iterator<llama_layer*>, std::move_iterator<llama_layer*>, llama_layer*)
00000000003982a3 W llama_layer_lora* std::uninitialized_copy<std::move_iterator<llama_layer_lora*>, llama_layer_lora*>(std::move_iterator<llama_layer_lora*>, std::move_iterator<llama_layer_lora*>, llama_layer_lora*)
00000000002046b4 W void std::__relocate_object_a<llama_layer, llama_layer, std::allocator<llama_layer> >(llama_layer*, llama_layer*, std::allocator<llama_layer>&)
00000000001edb47 W llama_layer* std::__uninitialized_copy_a<std::move_iterator<llama_layer*>, llama_layer*, llama_layer>(std::move_iterator<llama_layer*>, std::move_iterator<llama_layer*>, llama_layer*, std::allocator<llama_layer>&)
000000000039804d W llama_layer_lora* std::__uninitialized_copy_a<std::move_iterator<llama_layer_lora*>, llama_layer_lora*, llama_layer_lora>(std::move_iterator<llama_layer_lora*>, std::move_iterator<llama_layer_lora*>, llama_layer_lora*, std::allocator<llama_layer_lora>&)
00000000001eda98 W llama_layer* std::__uninitialized_default_n<llama_layer*, unsigned long>(llama_layer*, unsigned long)
0000000000397f9e W llama_layer_lora* std::__uninitialized_default_n<llama_layer_lora*, unsigned long>(llama_layer_lora*, unsigned long)
00000000001df23e W llama_layer* std::__uninitialized_default_n_a<llama_layer*, unsigned long, llama_layer>(llama_layer*, unsigned long, std::allocator<llama_layer>&)
0000000000397cac W llama_layer_lora* std::__uninitialized_default_n_a<llama_layer_lora*, unsigned long, llama_layer_lora>(llama_layer_lora*, unsigned long, std::allocator<llama_layer_lora>&)
00000000001edafb W std::move_iterator<llama_layer*> std::__make_move_if_noexcept_iterator<llama_layer, std::move_iterator<llama_layer*> >(llama_layer*)
0000000000398001 W std::move_iterator<llama_layer_lora*> std::__make_move_if_noexcept_iterator<llama_layer_lora, std::move_iterator<llama_layer_lora*> >(llama_layer_lora*)
00000000001df3f3 W llama_layer* std::__uninitialized_move_if_noexcept_a<llama_layer*, llama_layer*, std::allocator<llama_layer> >(llama_layer*, llama_layer*, llama_layer*, std::allocator<llama_layer>&)
0000000000397eb4 W llama_layer_lora* std::__uninitialized_move_if_noexcept_a<llama_layer_lora*, llama_layer_lora*, std::allocator<llama_layer_lora> >(llama_layer_lora*, llama_layer_lora*, llama_layer_lora*, std::allocator<llama_layer_lora>&)
000000000039861f W llama_layer* std::copy<std::move_iterator<llama_layer*>, llama_layer*>(std::move_iterator<llama_layer*>, std::move_iterator<llama_layer*>, llama_layer*)
00000000003986ed W llama_layer_lora* std::copy<std::move_iterator<llama_layer_lora*>, llama_layer_lora*>(std::move_iterator<llama_layer_lora*>, std::move_iterator<llama_layer_lora*>, llama_layer_lora*)
0000000000207c21 W std::remove_reference<llama_layer&>::type&& std::move<llama_layer&>(llama_layer&)
0000000000398c94 W std::remove_reference<llama_layer_lora&>::type&& std::move<llama_layer_lora&>(llama_layer_lora&)
00000000001ff779 W std::remove_reference<llama_layer*&>::type&& std::move<llama_layer*&>(llama_layer*&)
0000000000398569 W std::remove_reference<llama_layer_lora*&>::type&& std::move<llama_layer_lora*&>(llama_layer_lora*&)
00000000003982ef W llama_layer* std::fill_n<llama_layer*, unsigned long, llama_layer>(llama_layer*, unsigned long, llama_layer const&)
0000000000398484 W llama_layer_lora* std::fill_n<llama_layer_lora*, unsigned long, llama_layer_lora>(llama_layer_lora*, unsigned long, llama_layer_lora const&)
0000000000209e43 W llama_layer&& std::forward<llama_layer>(std::remove_reference<llama_layer>::type&)
00000000001d5fc4 W void std::_Destroy<llama_layer*>(llama_layer*, llama_layer*)
0000000000398082 W void std::_Destroy<llama_layer_lora*>(llama_layer_lora*, llama_layer_lora*)
000000000039873c W void std::__fill_a<llama_layer*, llama_layer>(llama_layer*, llama_layer*, llama_layer const&)
0000000000398802 W void std::__fill_a<llama_layer_lora*, llama_layer_lora>(llama_layer_lora*, llama_layer_lora*, llama_layer_lora const&)
00000000003988c8 W __gnu_cxx::__enable_if<!std::__is_scalar<llama_layer>::__value, void>::__type std::__fill_a1<llama_layer*, llama_layer>(llama_layer*, llama_layer*, llama_layer const&)
0000000000398974 W __gnu_cxx::__enable_if<!std::__is_scalar<llama_layer_lora>::__value, void>::__type std::__fill_a1<llama_layer_lora*, llama_layer_lora>(llama_layer_lora*, llama_layer_lora*, llama_layer_lora const&)
0000000000209e55 W bool std::operator==<llama_layer*>(std::move_iterator<llama_layer*> const&, std::move_iterator<llama_layer*> const&)
0000000000207c33 W bool std::operator!=<llama_layer*>(std::move_iterator<llama_layer*> const&, std::move_iterator<llama_layer*> const&)
```
And in this case there is a potential for a symbol collision where the dynamic
linker could pick the "wrong" symbol.

_wip_
