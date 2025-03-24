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
