## Segmentation fault when using GGML_BACKEND_DL
When using the `GGML_BACKEND_DL` option, the program crashes with a segmentation
fault when trying to load a model. This is because the model loading will need
to access a backend device to load the tensors, but when using `GGML_BACKEND_DL`
option, the device is not initialized properly.

```console
$ cmake -B build -GNinja -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DCMAKE_BUILD_TYPE=Debug
$ ninja -C build
```
```console
$ gdb --args build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav
(gdb) r
Starting program: /home/danbev/work/ai/whisper.cpp/build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
whisper_init_from_file_with_params_no_state: loading model from 'models/ggml-base.en.bin'
whisper_init_with_params_no_state: use gpu    = 1
whisper_init_with_params_no_state: flash attn = 0
whisper_init_with_params_no_state: gpu_device = 0
whisper_init_with_params_no_state: dtw        = 0
whisper_init_with_params_no_state: devices    = 0
whisper_init_with_params_no_state: backends   = 0
whisper_model_load: loading model
whisper_model_load: n_vocab       = 51864
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 512
whisper_model_load: n_audio_head  = 8
whisper_model_load: n_audio_layer = 6
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 512
whisper_model_load: n_text_head   = 8
whisper_model_load: n_text_layer  = 6
whisper_model_load: n_mels        = 80
whisper_model_load: ftype         = 1
whisper_model_load: qntvr         = 0
whisper_model_load: type          = 2 (base)
whisper_model_load: adding 1607 extra tokens
whisper_model_load: n_langs       = 99

Program received signal SIGSEGV, Segmentation fault.
0x00007ffff7eb6804 in ggml_backend_dev_backend_reg (device=0x0)
    at /home/danbev/work/ai/whisper.cpp/ggml/src/ggml-backend.cpp:472
472	    return device->reg;

(gdb) bt
#0  0x00007ffff7eb6804 in ggml_backend_dev_backend_reg (device=0x0)
    at /home/danbev/work/ai/whisper.cpp/ggml/src/ggml-backend.cpp:472
#1  0x00007ffff7d1164b in make_buft_list (params=...) at /home/danbev/work/ai/whisper.cpp/src/whisper.cpp:1419
#2  0x00007ffff7d130bd in whisper_model_load (loader=0x7fffffffcc60, wctx=...)
    at /home/danbev/work/ai/whisper.cpp/src/whisper.cpp:1734
#3  0x00007ffff7d1c2e2 in whisper_init_with_params_no_state (loader=0x7fffffffcc60, params=...)
    at /home/danbev/work/ai/whisper.cpp/src/whisper.cpp:3743
#4  0x00007ffff7d1be03 in whisper_init_from_file_with_params_no_state (
    path_model=0x555555714fa0 "models/ggml-base.en.bin", params=...)
    at /home/danbev/work/ai/whisper.cpp/src/whisper.cpp:3679
#5  0x00007ffff7d1c3b6 in whisper_init_from_file_with_params (path_model=0x555555714fa0 "models/ggml-base.en.bin",
    params=...) at /home/danbev/work/ai/whisper.cpp/src/whisper.cpp:3756
#6  0x00005555555c334d in main (argc=5, argv=0x7fffffffd818)
    at /home/danbev/work/ai/whisper.cpp/examples/cli/cli.cpp:1020
```

The loading is done by calling:
```console
static void whisper_load_backends() {
#ifdef GGML_BACKEND_DL
    static std::once_flag flag;
    std::call_once(flag, []() {
        ggml_backend_load_all();
    });
#endif
}
```
Perhaps I'm being naive but could we not just call this in the whisper_model_load
seeing that it requires a backend?

```console
diff --git a/src/whisper.cpp b/src/whisper.cpp
index cb887d45..7090142d 100644
--- a/src/whisper.cpp
+++ b/src/whisper.cpp
@@ -1507,6 +1507,8 @@ static ggml_backend_buffer_type_t select_weight_buft(const whisper_hparams & hpa
 static bool whisper_model_load(struct whisper_model_loader * loader, whisper_context & wctx) {
     WHISPER_LOG_INFO("%s: loading model\n", __func__);

+    whisper_load_backends();
+
     const int64_t t_start_us = ggml_time_us();

     wctx.t_start_us = t_start_us;

```
