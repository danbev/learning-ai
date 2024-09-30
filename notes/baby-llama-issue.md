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

#### Baba-llama notes
One thing to note about this example is that it is almost "self-contained" from
llama.cpp, so structs like `llama_model`, `llama_layer`,  `llama_kv_cache` can
be found in the exampl. There are part that use train.h though.

In the example we have weights that are randomized but also the training data
is generated by `get_example_targets`. Target in this context would be the
expected output, the prediction that is done by the model.



#### "backwards pass not implemented" error
This following error is raised when running the `llama-baby-llama` executable
(after the above mentioned changes):
```console
$ gdb --args ./llama-baby-llama
(gdb) r
Starting program: /home/danbev/work/ai/llama.cpp/llama-baby-llama
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
init model
init_kv_cache
ggml/src/ggml.c:6808: GGML_ASSERT(false && "backwards pass not implemented") failed
```
We can inspect the backtrace to see where the error is coming from:
```console
(gdb) bt
#0  ggml_rope_back (ctx=0x555555aacf68 <g_state+104>, a=0x7fff776e0eb0, b=0x7fff774a41f0, c=0x0, n_dims=4, mode=0, n_ctx_orig=0,
    freq_base=10000, freq_scale=1, ext_factor=0, attn_factor=1, beta_fast=0, beta_slow=0) at ggml/src/ggml.c:6808
#1  0x00005555555cba7d in ggml_compute_backward (ctx=0x555555aacf68 <g_state+104>, tensor=0x7fff774bb4c0, zero_table=0x7fffffffd4f0)
    at ggml/src/ggml.c:18586
#2  0x00005555555cd072 in ggml_build_backward_expand (ctx=0x555555aacf68 <g_state+104>, gf=0x7fff77576890, gb=0x7fff7758ab20,
    keep=true) at ggml/src/ggml.c:19022
#3  0x00005555555d5404 in ggml_opt_resume (ctx=0x555555aacf68 <g_state+104>, opt=0x7fffffffd560, f=0x7fff77575860)
    at ggml/src/ggml.c:21854
#4  0x00005555555d5354 in ggml_opt (ctx=0x555555aacf68 <g_state+104>, params=..., f=0x7fff77575860) at ggml/src/ggml.c:21835
#5  0x00005555558e99fe in (anonymous namespace)::baby_llama_main (argc=1, argv=0x7fffffffdb28)
    at examples/baby-llama/baby-llama.cpp:1555
#6  0x00005555558e9fd8 in main (argc=1, argv=0x7fffffffdb28) at examples/baby-llama/baby-llama.cpp:1646
```
If we go up the stack frames to `ggml_rope_back` we can see where the error is
coming from:
```c
struct ggml_tensor * ggml_rope_back(
    ...
    if (a->grad) {
        GGML_ASSERT(false && "backwards pass not implemented");
        is_node = false;
    }
```
And if we go up one more frame:
```c
static void ggml_compute_backward(struct ggml_context * ctx, struct ggml_tensor * tensor, struct ggml_hash_set * zero_table, struct ggml_hash_set * acc_table) {
    ...
        case GGML_OP_ROPE:
            {
                ...
                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_rope_back(ctx,
                                tensor->grad,
                                src1,
                                src2,
                                ...
```

If we go back up the stack frame one more time we see that
`ggml_compute_backward` is called from `ggml_build_backward_expand`:
```c
void ggml_build_backward_expand(struct ggml_context * ctx, struct ggml_cgraph * gf, struct ggml_cgraph * gb, bool keep) {
    ...

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct ggml_tensor * node = gf->nodes[i];

        // inplace operations to add gradients are not created by ggml_compute_backward
        // use allocator to automatically make inplace operations
        if (node->grad) {
            ggml_compute_backward(ctx, node, &zero_table);
        }
    }
```
Lets inspect the node:
```console
(gdb) p *node
$1 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {4, 8, 8, 8}, nb = {4, 16, 128, 1024},
  op = GGML_OP_ROPE, op_params = {0, 4, 0, 0, 0, 1176256512, 1065353216, 0, 1065353216, 0, 0, 0, 0, 0, 0, 0}, flags = 0,
  grad = 0x7fff776c6120, src = {0x7fff774b91e0, 0x7fff774a41f0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0,
  view_offs = 0, data = 0x7fff774bb610, name = "Qcur", '\000' <repeats 59 times>, extra = 0x0}
```
I've given some of the tensors names to make it easier to understand what is
going on. So `Qcur` was previously `node_34`.

We can see that `node_34/Qcur` in the computation graph:
```console
(gdb) p ggml_graph_print(gf)
=== GRAPH ===
n_nodes = 62
 -   0: [    32,     8,     1]             NONE x   (output)
 -   1: [    32,     1,     1]             NONE x   (norm)
 -   2: [    32,    64,     1]           REPEAT g   (?)
 -   3: [    86,    32,     1]             NONE x   (w2)
 -   4: [    32,    86,     1]             NONE x   (w1)
 -   5: [    32,     1,     1]             NONE x   (ffn_norm)
 -   6: [    32,    64,     1]           REPEAT g   (?)
 -   7: [    32,    32,     1]             NONE x   (wo)
 -   8: [    32,    32,     1]             NONE x   (wv)
 -   9: [    32,     1,     1]             NONE x   (attention_norm)
 -  10: [    32,    64,     1]           REPEAT g   (?)
 -  11: [    32,     8,     1]             NONE x   (tok_embeddings)
 -  12: [    32,    64,     1]         GET_ROWS g   (inpL)
 -  13: [    32,    64,     1]         RMS_NORM g   (rms_norm)
 -  14: [    32,    64,     1]              MUL g
 -  15: [    32,    64,     1]          MUL_MAT g
 -  16: [    32,     8,     8]          RESHAPE g
 -  17: [     8,    32,     8]          PERMUTE g
 -  18: [     8,    32,     8]             CONT g
 -  19: [   256,     8,     1]          RESHAPE g
 -  20: [  2048,     1,     1]              SET g
 -  21: [     8,     4,     8]             VIEW g
 -  22: [    32,    32,     1]             NONE x   (wk)
 -  23: [    32,    64,     1]          MUL_MAT g
 -  24: [     4,     8,     8]          RESHAPE g
 -  25: [     4,     8,     8]             ROPE g   (Kcur)
 -  26: [   256,     8,     1]          RESHAPE g   (Kcur (reshaped))
 -  27: [  2048,     1,     1]              SET g
 -  28: [    32,     8,     8]             VIEW g
 -  29: [     4,     8,     8]          RESHAPE g
 -  30: [     4,     8,     8]          PERMUTE g
 -  31: [    32,    32,     1]             NONE x   (wq)
 -  32: [    32,    64,     1]          MUL_MAT g
 -  33: [     4,     8,     8]          RESHAPE g
 -  34: [     4,     8,     8]             ROPE g   (Qcur)  <--- this is the node
 -  35: [     4,     8,     8]          PERMUTE g   (Qcur (permuted))
 -  36: [     8,     8,     8]          MUL_MAT g
 -  37: [     8,     8,     8]            SCALE g
 -  38: [     8,     8,     8]    DIAG_MASK_INF g
 -  39: [     8,     8,     8]         SOFT_MAX g
 -  40: [     4,     8,     8]          MUL_MAT g
 -  41: [     4,     8,     8]          PERMUTE g
 -  42: [     4,     8,     8]             CONT g
 -  43: [    32,    64,     1]          RESHAPE g
 -  44: [    32,    64,     1]          MUL_MAT g
 -  45: [    32,    64,     1]              ADD g
 -  46: [    32,    64,     1]         RMS_NORM g
 -  47: [    32,    64,     1]              MUL g
 -  48: [    86,    64,     1]          MUL_MAT g
 -  49: [    86,    64,     1]            UNARY g
 -  50: [    32,    86,     1]             NONE x   (w3)
 -  51: [    86,    64,     1]          MUL_MAT g
 -  52: [    86,    64,     1]              MUL g
 -  53: [    32,    64,     1]          MUL_MAT g
 -  54: [    32,    64,     1]              ADD g
 -  55: [    32,    64,     1]         RMS_NORM g
 -  56: [    32,    64,     1]              MUL g
 -  57: [     8,    64,     1]          MUL_MAT g
 -  58: [     8,     8,     8]          RESHAPE g
 -  59: [     8,     8,     8]              SUB g
 -  60: [     8,     8,     8]              SQR g
 -  61: [     1,     1,     1]              SUM g   (error)
n_leafs = 5
 -   0: [     8,     8]     NONE          targets
 -   1: [  2048,     1]     NONE           leaf_0
 -   2: [    64,     1]     NONE           tokens
 -   3: [  2048,     1]     NONE           leaf_2
 -   4: [     8,     1]     NONE           KQ_pos
========================================
```
So that was how we got to the error, now lets inspect that the arguments that
are being passed to `ggml_rope_back`:
```c
static void ggml_compute_backward(struct ggml_context * ctx, struct ggml_tensor * tensor, struct ggml_hash_set * zero_table, struct ggml_hash_set * acc_table) {
    ...
        case GGML_OP_ROPE:
            {
                ...
                    src0->grad = ggml_add_or_set(ctx,
                            src0->grad,
                            ggml_rope_back(ctx,
                                tensor->grad,
                                src1,
                                src2,
                                ...
```
Notice that we are passing `tensor-grad` as the first argument to
`ggml_rope_back` (which becomes `a` in the function). So lets inspect
`tensor-grad`
```console
(gdb) p *tensor->grad
$24 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {4, 8, 8, 8}, nb = {4, 128, 16, 1024},
op = GGML_OP_PERMUTE, op_params = {0, 2, 1, 3, 0 <repeats 12 times>}, flags = 0,
grad = 0x7fff776c6290, src = {0x7fff776c1e40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
view_src = 0x7fff776c1e40, view_offs = 0, data = 0x7fff776c1f90,
name = " (permuted)", '\000' <repeats 52 times>, extra = 0x0}
```
One things to also note where is that this tensor's grad also has a grad field
(second order derivative?). This is what is causing the error in
`ggml_rope_back` which is checking this field.
```c
    if (a->grad) {
        GGML_ASSERT(false && "backwards pass not implemented");
        is_node = false;
    }
```
So why is this happening?  
I'm currently suspecting that Qcur grad tensor should not also have a grad
tensor but this is only a hunch at the moment. What I mean is that this tensor
is the actual gradient tensor operation for src0 and should not have a gradient
itself.

So lets try to figure out where this gradient is coming from:
```console
(gdb) br ggml.c:19263 if node->grad->grad != 0x0
(gdb) r

Breakpoint 5, ggml_build_backward_expand (ctx=0x555555aaaf88 <g_state+104>, gf=0x7fff7756d440, gb=0x7fff775816d0, accumulate=false,
    keep=true) at ggml/src/ggml.c:19263
19263	            ggml_compute_backward(ctx, node, &zero_table, &acc_table);
(gdb) p i
$101 = 55
(gdb) p node->name
$102 = "norm*inpL", '\000' <repeats 54 times>

(gdb) p  *node->grad
$112 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {32, 64, 1, 1}, nb = {4, 128, 8192, 8192},
op = GGML_OP_OUT_PROD, op_params = {0 <repeats 16 times>}, flags = 0,
grad = 0x7fff776182d0, src = {0x7fffb7600790, 0x7fff77615ff0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff776162b0, name = '\000' <repeats 63 times>, extra = 0x0}
(gdb) p node->grad
$113 = (struct ggml_tensor *) 0x7fff77616160
(gdb) watch *0x7fff77616160
(gdb) r

Hardware watchpoint 6: *0x7fff77616160

Old value = <unreadable>
New value = 0
0x0000555555593d93 in ggml_new_tensor_impl (ctx=0x555555aaaf88 <g_state+104>, type=GGML_TYPE_F32, n_dims=4, ne=0x7fffffffd270,
    view_src=0x0, view_offs=0) at ggml/src/ggml.c:3989
3989	    *result = (struct ggml_tensor) {
(gdb) bt
#0  0x0000555555593d93 in ggml_new_tensor_impl (ctx=0x555555aaaf88 <g_state+104>, type=GGML_TYPE_F32, n_dims=4, ne=0x7fffffffd270,
    view_src=0x0, view_offs=0) at ggml/src/ggml.c:3989
#1  0x0000555555593fa9 in ggml_new_tensor (ctx=0x555555aaaf88 <g_state+104>, type=GGML_TYPE_F32, n_dims=4, ne=0x7fffffffd270)
    at ggml/src/ggml.c:4035
#2  0x0000555555598fac in ggml_out_prod (ctx=0x555555aaaf88 <g_state+104>, a=0x7fffb7600790, b=0x7fff77615ff0)
    at ggml/src/ggml.c:5811
#3  0x00005555555cbc0e in ggml_compute_backward (ctx=0x555555aaaf88 <g_state+104>, tensor=0x7fff77569200, zero_table=0x7fffffffd530,
    acc_table=0x7fffffffd550) at ggml/src/ggml.c:18550
#4  0x00005555555ce391 in ggml_build_backward_expand (ctx=0x555555aaaf88 <g_state+104>, gf=0x7fff7756d440, gb=0x7fff775816d0,
    accumulate=false, keep=true) at ggml/src/ggml.c:19263
#5  0x00005555555d69f1 in ggml_opt_resume (ctx=0x555555aaaf88 <g_state+104>, opt=0x7fffffffd5c0, f=0x7fff7756c410)
    at ggml/src/ggml.c:22139
#6  0x00005555555d693b in ggml_opt (ctx=0x555555aaaf88 <g_state+104>, params=..., f=0x7fff7756c410) at ggml/src/ggml.c:22120
#7  0x00005555558ea857 in (anonymous namespace)::baby_llama_main (argc=1, argv=0x7fffffffdb28)
    at examples/baby-llama/baby-llama.cpp:815
#8  0x00005555558eaa89 in main (argc=1, argv=0x7fffffffdb28) at examples/baby-llama/baby-llama.cpp:852
(gdb) up
#1  0x0000555555593fa9 in ggml_new_tensor (ctx=0x555555aaaf88 <g_state+104>, type=GGML_TYPE_F32, n_dims=4, ne=0x7fffffffd270)
    at ggml/src/ggml.c:4035
4035	    return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
(gdb) up
#2  0x0000555555598fac in ggml_out_prod (ctx=0x555555aaaf88 <g_state+104>, a=0x7fffb7600790, b=0x7fff77615ff0)
    at ggml/src/ggml.c:5811
5811	    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
(gdb) up
#3  0x00005555555cbc0e in ggml_compute_backward (ctx=0x555555aaaf88 <g_state+104>, tensor=0x7fff77569200, zero_table=0x7fffffffd530,
    acc_table=0x7fffffffd550) at ggml/src/ggml.c:18550
18550	                        ggml_add_or_set(ctx,
(gdb) up
#4  0x00005555555ce391 in ggml_build_backward_expand (ctx=0x555555aaaf88 <g_state+104>, gf=0x7fff7756d440, gb=0x7fff775816d0,
    accumulate=false, keep=true) at ggml/src/ggml.c:19263
19263	            ggml_compute_backward(ctx, node, &zero_table, &acc_table);
```
And we can can see what node this is (it's index):
```console
(gdb) p i
$114 = 56
```
And this corresponds to the following node in the computation graph:
```console
(gdb) p gf->nodes[56].name
$116 = "lm_head", '\000' <repeats 56 times>
```

This corresponds to the following code in `baby-llama.cpp`:
```
    // lm_head
    // inpL shape [n_vocab,N*n_batch,1,1]
    inpL = ggml_mul_mat(ctx0, model->output, inpL);
    ggml_set_name(inpL, "lm_head");
```
Now, lets inspect what is happening for this node:
```console
(gdb) disable breakpoints 
(gdb) br ggml.c:19263 if i == 56
Note: breakpoint 5 (disabled) also set at pc 0x5555555ce376.
Breakpoint 8 at 0x5555555ce376: file ggml/src/ggml.c, line 19263.
(gdb) r
Breakpoint 8, ggml_build_backward_expand (ctx=0x555555aaaf88 <g_state+104>, gf=0x7fff7756d440, gb=0x7fff775816d0, accumulate=false,
    keep=true) at ggml/src/ggml.c:19263
19263	            ggml_compute_backward(ctx, node, &zero_table, &acc_table);
(gdb) p i
$128 = 56
(gdb) p node->name
$129 = "lm_head", '\000' <repeats 56 times>
```
Stepping into `ggml_compute_backward` for this node will land us in:
```c
        case GGML_OP_MUL_MAT:
            {
                if (src0->grad) {
                    struct ggml_tensor * s1_tg =
                        ggml_out_prod(ctx, // [n,m,qq,rr]
                            src1,          // [n,p,qq,rr]
                            tensor->grad); // [m,p,qq,rr]
                    const int64_t qq = s1_tg->ne[2];
                    const int64_t rr = s1_tg->ne[3];
                    const int64_t q1 = src0->ne[2];
                    const int64_t r1 = src0->ne[3];
                    const bool ne2_broadcasted = qq > q1;
                    const bool ne3_broadcasted = rr > r1;
                    if (ne2_broadcasted || ne3_broadcasted) {
                        // sum broadcast repetitions of s1_tg into shape of src0
                        s1_tg = ggml_repeat_back(ctx, s1_tg, src0);
                    }
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad, // [n,m,q1,r1]
                                s1_tg,      // [n,m,q1,r1]
                                zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_or_set(ctx,
                                src1->grad,                            // [n,p,qq,rr]
                                // ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                                //     ggml_cont(ctx,                  // [m,n,q1,r1]
                                //         ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                                //     tensor->grad),                  // [m,p,qq,rr]

                                // // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                                // // avoid transpose of src0, rather transpose smaller tensor->grad
                                // // and then use ggml_out_prod
                                ggml_out_prod(ctx,                  // [n,p,qq,rr]
                                    src0,                           // [n,m,q1,r1]
                                    ggml_transpose(ctx,             // [p,m,qq,rr]
                                        tensor->grad)),             // [m,p,qq,rr]
                                zero_table, acc_table);
                }
            } break;
```
My understanding is that this is creating a tensor operations to compute the
gradients for the matrix multiplication using the outer product as an
optimization instead of matrix multiplication.

In this case `src0` is the `model-output` tensor which gradient tensor is
getting updated. This is done by passing in src1 and the tensor gradient.
```console
(gdb) p src1.name
$132 = "norm*inpL", '\000' <repeats 54 times>
(gdb) p *src1->grad
$139 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {32, 64, 1, 1}, nb = {4, 128, 8192, 8192}, 
  op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 
    0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff77610c40, name = '\000' <repeats 63 times>, extra = 0x0}
```
So first a tensor is created that will be the operation for computing src1 and
the tensor gradient (`s1_tg`)
```c
                    struct ggml_tensor * s1_tg =
                        ggml_out_prod(ctx, // [n,m,qq,rr]
                            src1,          // [n,p,qq,rr]
                            tensor->grad); // [m,p,qq,rr]
```
Now, tensor is the `lm_head` tensor, and we are passing in it's gradient as
the third argument to `ggml_out_prod`:
```console
(gdb) p *tensor->grad
$135 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0,
ne = {8, 64, 1, 1}, nb = {4, 32, 2048, 2048}, op = GGML_OP_RESHAPE,
op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x7fff77614a30, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0}, view_src = 0x7fff77614a30, view_offs = 0, data = 0x7fff77614b80,
name = " (reshaped)", '\000' <repeats 52 times>, extra = 0x0}

(gdb) p tensor->grad->grad
$136 = (struct ggml_tensor *) 0x0
```
Now, `ggml_out_prod` will check if the src1 (a), or `tensor->grad` (b) has
a gradient:
```c
struct ggml_tensor * ggml_out_prod(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    const int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    result->op   = GGML_OP_OUT_PROD;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
```
So the resulting tensor (`s1_tg`) will look like this:
```console
(gdb) p *s1_tg
$4 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {32, 8, 1, 1}, nb = {4, 128, 1024, 1024},
op = GGML_OP_OUT_PROD, op_params = {0 <repeats 16 times>}, flags = 0,
grad = 0x7fff77615a80, src = {0x7fff77564f20, 0x7fff776153a0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff77615660, name = '\000' <repeats 63 times>, extra = 0x0}
```
A little further down we the set `src0->grad`:
```c

                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad, // [n,m,q1,r1]
                                s1_tg,      // [n,m,q1,r1]
                                zero_table, acc_table);
```
So the gradient for src0 will be set to `s1_tg` which we know also has a
gradient:
```console
(gdb) p *src0->grad
$7 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {32, 8, 1, 1}, nb = {4, 128, 1024, 1024},
op = GGML_OP_OUT_PROD, op_params = {0 <repeats 16 times>}, flags = 0,
grad = 0x7fff77615a80, src = {0x7fff77564f20, 0x7fff776153a0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff77615660,
name = '\000' <repeats 63 times>, extra = 0x0}

(gdb) p *src0->grad->grad
$8 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {32, 8, 1, 1}, nb = {4, 128, 1024, 1024},
op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, grad = 0x0, src = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0}, view_src = 0x0, view_offs = 0, data = 0x7fff77615bd0, name = '\000' <repeats 63 times>, extra = 0x0}
```
My take on this is that the outer product tensor (`s1_tg`) is the actual
gradient computation for src0, but it should not have a gradient itself. And
since this is set to the gradient of src0, this is what is causing the error.
I'm going to purpose a "fix" for this to see if I'm on the right track for this
and if not perhaps others can explain where my reasoning is incorrect.

So how does this relate to the error message that we are seeing?  This node
is now where near the `Qcur` node in the computation graph.

Well this kinda of "trickles" its way down. If we take a look at the grads for
the node we went through above we can see this that after it's source gradients
have been set they are:
```console
(gdb) p tensor->src[0]->grad
$51 = (struct ggml_tensor *) 0x7fff77615510
(gdb) p tensor->src[1]->grad
$52 = (struct ggml_tensor *) 0x7fff77616160
```
If we now take a look at the next node to be processed:
```console
(gdb) p node->name
$53 = "norm*inpL", '\000' <repeats 54 times>
(gdb) p i
$54 = 55
gdb) p tensor->grad
$55 = (struct ggml_tensor *) 0x7fff77616160
```
Notice that this is the same tensor as the second source tensor for the previous
node. And this will be used as an argument of the gradient computation for the
current nodes parents:
```console
        case GGML_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_or_set(ctx,
                                src0->grad,
                                ggml_mul(ctx, src1, tensor->grad),
                                zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_or_set(ctx,
                                src1->grad,
                                ggml_mul(ctx, src0, tensor->grad),
                                zero_table, acc_table);
                }
            } break;
```

With out going through all these nodes this is how this issue is affecting
other nodes int the computation graph.

When I was going to commit a [suggestion](https://github.com/danbev/ggml/tree/out_prod_no_grad)
for fixing this there had been an [update](https://github.com/ggerganov/ggml/pull/966)
to ggml which fixes this.
