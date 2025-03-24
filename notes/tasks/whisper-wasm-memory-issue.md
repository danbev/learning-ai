## whisper.wasm WebAssembly Memory Issue
Issue: https://github.com/ggerganov/whisper.cpp/issues/2920

The user reported that this would happen when trying to run the whisper.wasm
appplication: https://whisper.ggerganov.com/
I've not been able to reproduce this error on the live demo or locally yet.
```console
WebAssembly.Memory(): could not allocate memory error.
```

The origin of this error is from `main.js:1:7645`. This file is minified so it is
just one line but we can jump to this column
```console
wasmBinaryFile="data:application/octet-stream;base64,AGFzbQEAAAABgAVRYAF/AX9gAX8AYAJ/fwF/YAJ/fwBgA39/fwF/YAN/f34AYAN/f38AYAR/f39/AX9gBH9/f38AYAZ/f39/f38Bf2AAAGAFf39/f38Bf2AFf39/f38AYAh/f39/f39/fwBgBn9/f39/fwBgAAF/YAh/f39/f39/fwF/YAd/f39/f39/AX9gAX0BfWAFf35+fn4AYAN/fn8BfmAHf39/f39/fwBgBX9/f39+AX9gBX9/f39/AXxgA39+fwF/YAR/f35+AX9gBH9+fn8AYAp/f39/f39/f39/AGADf39/AXxgAn9+AGABfAF9YAN/f30Bf2ACfH8BfGAFf39+fn4Bf2AKf39/f39/f39/fwF/YAx/f39/f39/f39/f38Bf2AJf39/f39/f39/AGAPf39/f39/f39/f39/f39/AGALf
"main.js" line 1 of 1 --100%-- col 7645
```
This is just the loading of the base64 encoded wasm file which does not give us much information.

Lets update CMakeList.txt to produce a non-minified version of the main.js file:
```console
set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    --bind \
    --emit-symbol-map -g3 --source-map-base ./ -O0 \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE_STRICT=0 \
    -s INITIAL_MEMORY=2000MB \
    -s TOTAL_MEMORY=2000MB \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap']\" \
    ${EXTRA_FLAGS} \
    ")
```
The WebAssembly.Memory object is created in the main.js file, and we can set this
to a lower value to see if we can reproduce the error:
```javascript
 if (Module['wasmMemory']) {
    wasmMemory = Module['wasmMemory'];
  } else
  {
    var INITIAL_MEMORY = Module['INITIAL_MEMORY'] || 134217728;legacyModuleProp('INITIAL_MEMORY', 'INITIAL_MEMORY');

    assert(INITIAL_MEMORY >= 5242880, 'INITIAL_MEMORY should be larger than STACK_SIZE, was ' + INITIAL_MEMORY + '! (STACK_SIZE=' + 5242880 + ')');
    /** @suppress {checkTypes} */
    wasmMemory = new WebAssembly.Memory({
      //'initial': INITIAL_MEMORY / 65536,
      //'maximum': INITIAL_MEMORY / 65536,
      'initial': 2048,
      'maximum': 2048,
      'shared': true,
    });
  }
```
This is where I think the error originates from, but I can't be sure as the console.log output
does not really have an order. The error is thrown when the WebAssembly.Memory object is created.
I've tried lowering this to see if I could reproduce the error but I could not. I can get it to
throw an error but not at the exact same place as the user reported.

We can then check this in the devtools console:
```console
> console.log((Module.HEAP8.length / (1024 * 1024)).toFixed(2) + " MB");
128.00 MB
```
This is a TypedArrayBuffer (Int8Array), and this is getting the size of the ArrayBuffer
in bytes (because we are using the Int8Array view).

One thing I've tried it to manually trigger an error in main.js where the
WebAssembly.Memory is requested:
build-em/bin/whisper.wasm/main.js:
```javascript
if (ENVIRONMENT_IS_PTHREAD) {
    wasmMemory = Module["wasmMemory"];
    buffer = Module["buffer"];
} else {
    if (Module["wasmMemory"]) {
        wasmMemory = Module["wasmMemory"];
    } else {
        throw Error("danbev simulating memory issue");

        wasmMemory = new WebAssembly.Memory({
            "initial": INITIAL_MEMORY / 65536,
            "maximum": INITIAL_MEMORY / 65536,
            "shared": true
        });
        if (!(wasmMemory.buffer instanceof SharedArrayBuffer)) {
           err("requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag");
            if (ENVIRONMENT_IS_NODE) {                                                   
                console.log("(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and also use a recent version)");
            }
            throw Error("bad memory");
            }
     }
}
```
This will produce and initial error like this when the example is loaded:
```console
main.js:1049 Uncaught Error: danbev simulating memory issue
    at main.js:1049:10
```
When trying to load a model I get:
```console
loadRemote: storage quota: 603520792166 bytes
helpers.js:14 loadRemote: storage usage: 109873656 bytes
helpers.js:14 loadRemote: "https://whisper.ggerganov.com/ggml-model-whisper-tiny.en-q5_1.bin" is already in the IndexedDB
whisper.wasm/:291 Uncaught TypeError: Module.FS_createDataFile is not a function
    at storeFS (whisper.wasm/:291:24)
    at rq.onsuccess (helpers.js:124:17)
```
And loading a sample:
```console
js: loading audio: jfk.wav, size: 352078 bytes
helpers.js:14 js: please wait ...
helpers.js:14 js: audio loaded, size: 176000
```
And then when I try to transcribe the audio I get:
```console
js: loading audio: jfk.wav, size: 352078 bytes
helpers.js:14 js: please wait ...
helpers.js:14 js: audio loaded, size: 176000
```
This at least matches the reported errors. So the the theory is that the
initial memory we request for WebAssembly.Memory is too large, it is currently
2GB. We could reduce this to 256MB and allow it to grow, but still leave the
max/total at 2GB:
```console
set_target_properties(${TARGET} PROPERTIES LINK_FLAGS " \
    --bind \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE_STRICT=0 \
    -s INITIAL_MEMORY=256MB \
    -s TOTAL_MEMORY=2000GB \
    -s ALLOW_MEMORY_GROWTH=1 \
    --emit-symbol-map -g3 --source-map-base ./ -O0 \
    -s FORCE_FILESYSTEM=1 \
    -s EXPORTED_RUNTIME_METHODS=\"['print', 'printErr', 'ccall', 'cwrap']\" \
    ${EXTRA_FLAGS} \
    ")
```
Actually, there is another issue here. The `INITIAL_MEMORY` and `TOTAL_MEMORY`
are actually specifying the same thing: 
https://emscripten.org/docs/tools_reference/settings_reference.html#initial-memory

We should be specifing:
```
    -s INITIAL_MEMORY=256MB \
    -s MAXIMUM_MEMORY=2000MB \
```

Lets try this out using a new chrome brower and restrict the memory it has
available to it:
To try this out we can start a new chrome instance:
```console
$ mkdir -p /tmp/chrome-test-profile
```
First we try using 3200 pages (64KB) which is 200MB:
```console
google-chrome --user-data-dir=/tmp/chrome-test-profile --no-sandbox --js-flags="--wasm-max-mem-pages=3200" http://localhost:8000/whisper.wasm/
```

```console
Uncaught RangeError: WebAssembly.Memory(): could not allocate memory
    at main.js:1:12464Understand this errorAI
favicon.ico:1 
            
            
           Failed to load resource: the server responded with a status of 404 (File not found)Understand this errorAI
helpers.js:14 loadRemote: storage quota: 603520792166 bytes
helpers.js:14 loadRemote: storage usage: 0 bytes
helpers.js:14 loadRemote: "https://whisper.ggerganov.com/ggml-model-whisper-tiny.en.bin" is already in the IndexedDB
whisper.wasm/:291 Uncaught TypeError: Module.FS_createDataFile is not a function
    at storeFS (whisper.wasm/:291:24)
    at rq.onsuccess (helpers.js:124:17)
storeFS @ whisper.wasm/:291
rq.onsuccess @ helpers.js:124Understand this errorAI
helpers.js:14 js: loading audio: jfk.wav, size: 352078 bytes
helpers.js:14 js: please wait ...
helpers.js:14 js: audio loaded, size: 176000
whisper.wasm/:623 Uncaught TypeError: Module.init is not a function
    at onProcess (whisper.wasm/:623:39)
    at HTMLButtonElement.onclick (whisper.wasm/:192:61)
```

Now, lets try with with 4800 pages (300MB) which should be enough::
```console
google-chrome --user-data-dir=/tmp/chrome-test-profile --no-sandbox --js-flags="--wasm-max-mem-pages=4800" http://localhost:8000/whisper.wasm/
```
```console
js: Running...
favicon.ico:1


           Failed to load resource: the server responded with a status of 404 (File not found)Understand this errorAI
helpers.js:14 js:
helpers.js:14 loadRemote: storage quota: 603520792166 bytes
helpers.js:14 loadRemote: storage usage: 0 bytes
helpers.js:14 loadRemote: "https://whisper.ggerganov.com/ggml-model-whisper-tiny.en-q5_1.bin" is not in the IndexedDB
helpers.js:14 fetchRemote: downloading with fetch()...
helpers.js:14 fetchRemote: fetching 0% ...
helpers.js:14 fetchRemote: fetching 10% ...
helpers.js:14 fetchRemote: fetching 20% ...
helpers.js:14 fetchRemote: fetching 30% ...
helpers.js:14 fetchRemote: fetching 40% ...
helpers.js:14 fetchRemote: fetching 50% ...
helpers.js:14 fetchRemote: fetching 60% ...
helpers.js:14 fetchRemote: fetching 70% ...
helpers.js:14 fetchRemote: fetching 80% ...
helpers.js:14 fetchRemote: fetching 90% ...
helpers.js:14 fetchRemote: fetching 100% ...
helpers.js:14 loadRemote: "https://whisper.ggerganov.com/ggml-model-whisper-tiny.en-q5_1.bin" stored in the IndexedDB
helpers.js:14 storeFS: stored model: whisper.bin size: 32166155
helpers.js:14 js: loading audio: jfk.wav, size: 352078 bytes
helpers.js:14 js: please wait ...
helpers.js:14 js: audio loaded, size: 176000
helpers.js:14 whisper_init_from_file_with_params_no_state: loading model from 'whisper.bin'
helpers.js:14 whisper_init_with_params_no_state: use gpu    = 1
helpers.js:14 whisper_init_with_params_no_state: flash attn = 0
helpers.js:14 whisper_init_with_params_no_state: gpu_device = 0
helpers.js:14 whisper_init_with_params_no_state: dtw        = 0
helpers.js:14 whisper_init_with_params_no_state: devices    = 1
helpers.js:14 whisper_init_with_params_no_state: backends   = 1
helpers.js:14 whisper_model_load: loading model
helpers.js:14 whisper_model_load: n_vocab       = 51864
helpers.js:14 whisper_model_load: n_audio_ctx   = 1500
helpers.js:14 whisper_model_load: n_audio_state = 384
helpers.js:14 whisper_model_load: n_audio_head  = 6
helpers.js:14 whisper_model_load: n_audio_layer = 4
helpers.js:14 whisper_model_load: n_text_ctx    = 448
helpers.js:14 whisper_model_load: n_text_state  = 384
helpers.js:14 whisper_model_load: n_text_head   = 6
helpers.js:14 whisper_model_load: n_text_layer  = 4
helpers.js:14 whisper_model_load: n_mels        = 80
helpers.js:14 whisper_model_load: ftype         = 9
helpers.js:14 whisper_model_load: qntvr         = 1
helpers.js:14 whisper_model_load: type          = 1 (tiny)
helpers.js:14 whisper_model_load: adding 1607 extra tokens
helpers.js:14 whisper_model_load: n_langs       = 99
helpers.js:14 whisper_model_load:      CPU total size =    31.57 MB
helpers.js:14 whisper_model_load: model size    =   31.57 MB
helpers.js:14 whisper_backend_init_gpu: no GPU found
helpers.js:14 whisper_init_state: kv self size  =    3.15 MB
helpers.js:14 whisper_init_state: kv cross size =    9.44 MB
helpers.js:14 whisper_init_state: kv pad  size  =    2.36 MB
helpers.js:14 whisper_init_state: compute buffer (conv)   =   12.78 MB
helpers.js:14 whisper_init_state: compute buffer (encode) =   64.38 MB
helpers.js:14 whisper_init_state: compute buffer (cross)  =    3.47 MB
helpers.js:14 ggml_aligned_malloc: insufficient memory (attempted to allocate  20.06 MB)
helpers.js:14 /home/danbev/work/ai/whisper-work/ggml/src/ggml.c:1450: GGML_ASSERT(ctx->mem_buffer != NULL) failed
```
So we did not get the intial WebAssembly.Memory error but we do need a little
more memory it seems.
```console
$ google-chrome --user-data-dir=/tmp/chrome-test-profile --no-sandbox --js-flags="--wasm-max-mem-pages=8192" http://localhost:8000/whisper.wasm/
```
This worked. So perhaps the inital size should be a little higher that 256MB.
Perhaps setting it to 512MB would be a good option.
