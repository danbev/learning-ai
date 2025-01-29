## llama.cpp server notes
This document contains notes about the llama.cpp server.

### Starting the server
```console
$ ./build/bin/llama-server -m models/llama-2-7b.Q4_K_M.gguf -n 20
```
This server has a GUI which can be accessed at `http://localhost:8080/`.

### Calling the server using curl:
```console
$ ./call-server.sh | jq
{
  "content": " The LoRaWAN Specification\n броја 1.0.2\nThe Low Power",
  "id_slot": 0,
  "stop": true,
  "model": "models/llama-2-7b.Q4_K_M.gguf",
  "tokens_predicted": 20,
  "tokens_evaluated": 6,
  "generation_settings": {
    "n_ctx": 4096,
    "n_predict": 20,
    "model": "models/llama-2-7b.Q4_K_M.gguf",
    "seed": 4294967295,
    "seed_cur": 4203817392,
    "temperature": 0.800000011920929,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "top_k": 40,
    "top_p": 0.949999988079071,
    "min_p": 0.05000000074505806,
    "tfs_z": 1.0,
    "typical_p": 1.0,
    "repeat_last_n": 64,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "mirostat": 0,
    "mirostat_tau": 5.0,
    "mirostat_eta": 0.10000000149011612,
    "penalize_nl": false,
    "stop": [],
    "max_tokens": 20,
    "n_keep": 0,
    "n_discard": 0,
    "ignore_eos": false,
    "stream": false,
    "n_probs": 0,
    "min_keep": 0,
    "grammar": "",
    "samplers": [
      "top_k",
      "tfs_z",
      "typ_p",
      "top_p",
      "min_p",
      "temperature"
    ]
  },
  "prompt": "What is LoRA:",
  "has_new_line": true,
  "truncated": false,
  "stopped_eos": false,
  "stopped_word": false,
  "stopped_limit": true,
  "stopping_word": "",
  "tokens_cached": 25,
  "timings": {
    "prompt_n": 6,
    "prompt_ms": 315.259,
    "prompt_per_token_ms": 52.54316666666667,
    "prompt_per_second": 19.031970538509604,
    "predicted_n": 20,
    "predicted_ms": 876.246,
    "predicted_per_token_ms": 43.8123,
    "predicted_per_second": 22.824640568972644
  },
  "index": 0
}
```

### Walkthrough
This section will step through the server code to understand how it works.

```console
$ lldb ./build/bin/llama-server -- -m models/llama-2-7b.Q4_K_M.gguf -n 20
(lldb) br set -f server.cpp -l 2436
Breakpoint 1: where = llama-server`main + 120 at server.cpp:2436:19, address = 0x000000010000221c

(lldb) r
Process 94087 launched: '/Users/danbev/work/llama.cpp/build/bin/llama-server' (arm64)
Process 94087 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x000000010000221c llama-server`main(argc=5, argv=0x000000016fdff2e0) at server.cpp:2436:19
   2433
   2434	int main(int argc, char ** argv) {
   2435	    // own arguments required by this example
-> 2436	    common_params params;
   2437
   2438	    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
   2439	        return 1;
Target 0: (llama-server) stopped.
```
_wip_

### index_html_gz
In server.cpp we have the following code:
```console
            // using embedded static index.html
            svr->Get("/", [](const httplib::Request & req, httplib::Response & res) {
                if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                    res.set_content("Error: gzip is not supported by this browser", "text/plain");
                } else {
                    res.set_header("Content-Encoding", "gzip");
                    res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                }
                return false;
            });
```

Now, `index_html_gz` gzipped file in `examples/server/public` which is built
by `examples/server/webui/package.json`:
```console
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "analyze": "ANALYZE=1 npx vite-bundle-visualizer"
  },
```
We can inspect the vite configuration which is in `vite.config.js`:
```js
...
      writeBundle() {
        const outputIndexHtml = path.join(config.build.outDir, 'index.html');
        const content = GUIDE_FOR_FRONTEND + '\n' + fs.readFileSync(outputIndexHtml, 'utf-8');
        const compressed = zlib.gzipSync(Buffer.from(content, 'utf-8'), { level: 9 });

        // because gzip header contains machine-specific info, we must remove these data from the header
        // timestamp
        compressed[0x4] = 0;
        compressed[0x5] = 0;
        compressed[0x6] = 0;
        compressed[0x7] = 0;
        // OS
        compressed[0x9] = 0;
```
This is reading the `webui/index.html` file and prepending `GUIDE_FOR_FRONTEND`
warning to it. This is then gzipped and the timestamp and OS fields are zeroed
out.
So when we run `npm run build` in the `webui` directory, the `index.html` file
is built and gzipped and the resulting `index.html.gz` file is copied to the

And then when we build `llama-server` using cmake we can see the following
in `examples/server/CMakeLists.txt`:
```cmake
set(PUBLIC_ASSETS
    index.html.gz
    loading.html
)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
endforeach()
```
Notice that this is actually generateing a `.hpp` file from the `.gz` file:
```console
/home/danbev/work/ai/llama.cpp-debug/build/examples/server/index.html.gz.hpp
```

Now, this is passed to the script `xxd.cmake`:
```
# CMake equivalent of `xxd -i ${INPUT} ${OUTPUT}`
```
xxd is a hexdump/converter util and the `-i` flag is to output C-style arrays.


If we look in includes in server.cpp we find:
```cpp
#include "index.html.gz.hpp"
```

```cpp
unsigned char index_html_gz[] = {0x1f,0x8b,...

unsigned int index_html_gz_len = 1207150;
```
And this is how the `index.html.gz` file is included in the server:
```cpp
    res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
```
