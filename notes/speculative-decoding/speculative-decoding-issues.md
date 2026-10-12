## Speculative Draft decoding issues

###  quantized speculative decoding differs from baseline
Issue: https://github.com/ggml-org/llama.cpp/issues/25618

If we take a quantized target model without using speculative drafting and
have use a temperature of 0 for sampling we should see the target models baseline
greed output for that prompt (if we specify a seed values this should be reproducable as well).

Now, if we enable speculative decoding with the same settings as above and use
greedy sampling (not probabalistic draft sampling) the draft model should produce
the same choice as ordinary decoding from above. Even if the drafter has a low
acceptance rate or speed, that should not matter for the actualy result.

With a quantized target model, speculative decoding disabled, and temperature 0,
we obtain the target's baseline greedy output for a prompt. Repeated runs should
match under the same deterministic execution conditions.

If we enable speculative decoding while keeping the target model, prompt, and
sampling settings unchanged, the final verified output should match that baseline.
The drafter's proposals may differ, but the target should reject proposals that
disagree with its greedy choices. A low acceptance rate should affect
performance, not the output.

The reported issue is that evaluating the target in a multi-token verification
batch can produce different logits than evaluating it one token at a time. Those
differences can change the greedy choice, causing speculative output to diverge
from the baseline.

* https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf
* https://huggingface.co/ggml-org/Qwen3-8B-GGUF/resolve/main/dspark-Qwen3-8B-Q8_0.gguf 

#### Baseline
```console
#!/bin/bash

set -e

#build_dir="build-cpu-debug"
build_dir="build-cuda-89-release"

model=models/Qwen3-8B-Q4_K_M.gguf

cmd=llama-server

cmake --build ${build_dir} --target $cmd -j 12

$build_dir/bin/llama-server \
    -m $model \
    --spec-type none \
    -ngl 99 -c 4096 -np 1 -fa on \
    -ctk f16 -ctv f16 --jinja --port 8088
```

### Draft
```console
#!/bin/bash

set -e

#build_dir="build-cpu-debug"
build_dir="build-cuda-89-release"

model=models/Qwen3-8B-Q4_K_M.gguf
draft=models/dspark-Qwen3-8B-Q8_0.gguf

cmd=llama-server

cmake --build ${build_dir} --target $cmd -j 12

$build_dir/bin/llama-server \
    -m $model \
    -md $draft \
    --spec-type draft-dspark --spec-draft-n-max 7 -ngld 99 \
    -ngl 99 -c 4096 -np 1 -fa on \
    -ctk f16 -ctv f16 --jinja --port 8088
```

#### Probe
```console
$ cat probe.sh
probe() {
  curl -fsS http://127.0.0.1:8088/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "messages": [{
        "role": "user",
        "content": "What happens when a browser requests https://example.com? Five bullet points."
      }],
      "temperature": 0,
      "top_k": 1,
      "seed": 42,
      "max_tokens": 512,
      "cache_prompt": false,
      "chat_template_kwargs": {"enable_thinking": false}
    }'
}
```

Probe baseline server:
```console
$ probe | jq -r '.choices[0].message.content'
- The browser checks if it has a cached copy of the website's SSL/TLS certificate.
- The browser initiates a TLS handshake with the server to establish a secure connection.
- The server responds with its SSL/TLS certificate, which the browser verifies against a trusted root certificate authority.
- If the certificate is valid, the browser and server agree on encryption settings and exchange keys to encrypt data.
- The browser proceeds to request the website's content over the secure connection.
```

Probe draft server:
```console
$ probe | jq -r '.choices[0].message.content'
- The browser checks if it has a cached copy of the website's SSL/TLS certificate.
- The browser initiates a TLS handshake with the server to establish a secure connection.
- The server responds with its SSL/TLS certificate, which the browser verifies against a trusted root certificate authority.
- If the certificate is valid, the browser and server agree on encryption settings and exchange keys to encrypt data.
- The browser proceeds to request the webpage content over the secure connection.
```
Notice the difference:
```console
- The browser proceeds to request the website's content over the secure connection.
                                         ↑
- The browser proceeds to request the webpage content over the secure connection.
```

If we run the CPU backend we get this (baseline):
```console
$ probe | jq -r '.choices[0].message.content' 
- The browser checks if it has a cached copy of the website's SSL/TLS certificate.  
- The browser initiates a TLS handshake with the server to establish a secure connection.  
- The server responds with its SSL/TLS certificate, which the browser verifies against a trusted root certificate authority.  
- If the certificate is valid, the browser and server agree on encryption settings and exchange keys to encrypt the communication.  
- The browser proceeds to request the website's content over the secure connection.
```
And this with speculative drafting:
```console
$ probe | jq -r '.choices[0].message.content'
- The browser checks if it has a cached copy of the website's SSL/TLS certificate.
- The browser initiates a TLS handshake with the server to establish a secure connection.
- The server responds with its SSL/TLS certificate, which the browser verifies against a trusted root certificate authority.
- If the certificate is valid, the browser and server agree on encryption settings and exchange keys to encrypt the communication.
- The browser proceeds to request the website's content over the secure connection.
```
Notice that these are identical which is what we would expect.


So running the speculative draft server and adding some output logging in
sampling.cpp's common_sampler_sample_and_accept_n I was able to print the token
id and the other candidates are for the token that represents `webpage`:
```console
0.10.348.594 I common_sampler_sample_and_accept_n: token id = 44610, str:  webpage
0.10.348.597 I draft = 11223, target = 44610, match = 0, row = 0
```
So the draft proposed token 11233, but this was rejectede (match=0), and the target
logit row used for verification was row 0:
```console
0.10.349.740 I token = 44610, logit = 37.4872627, piece = ' webpage'
0.10.349.742 I token = 3910,  logit = 37.3778076, piece = ' website'
0.10.349.742 I token = 3482,  logit = 35.4710083, piece = ' web'
0.10.349.743 I token = 11223, logit = 33.6743164, piece = ' requested'
0.10.349.743 I token = 3151,  logit = 31.249321,  piece = ' specific'
```
And ' webpage' is above in rank compared to ' website' so the sampling is work
as it should here, the correct token is being selected according to the logits.

Now, we can also see that ' website' has token id 3910. So we can run the baseline
model server and add logging to common_sample_sample when we see that token
and print out information about it:
```console
0.20.521.217 I common_sampler_sample: token id = 3910, str: ' website', row = 0
0.20.522.476 I token = 3910,  logit = 37.2676849, piece = ' website'
0.20.522.477 I token = 25316, logit = 33.4767532, piece = ' SSL'
0.20.522.478 I token = 2747,  logit = 33.0001755, piece = ' site'
0.20.522.479 I token = 15748, logit = 31.2881069, piece = ' certificate'
0.20.522.479 I token = 3538,  logit = 28.3665161, piece = ' server'

0.21.445.915 I common_sampler_sample: token id = 3910, str: ' website', row = 0
0.21.446.872 I token = 3910,  logit = 37.3920135, piece = ' website'
0.21.446.874 I token = 44610, logit = 37.3027039, piece = ' webpage'
0.21.446.874 I token = 3482,  logit = 35.3343048, piece = ' web'
0.21.446.875 I token = 11223, logit = 33.8154259, piece = ' requested'
0.21.446.875 I token = 3151,  logit = 31.3140221, piece = ' specific'
```
So we can see that we have:
```console
 Token           Baseline logit         Speculative logit
website 3910     37.3910135             37.3778076
webpage 44610    37.3027039             37.4872627
```
So there is a difference in the logits values before the sampling is done.


When the target model verifies the drafted tokens it does so in a batch, here
we have 8 tokens:
```console
0.08.345.820 I MMVQ launch:
type = q4_K, ncols_dst = 8, nwarps = 2, rows_per_block = 2, small_k = 0,
halve_iters = 0, fusion = 0, table = 0, cc = 890, grid = [2048,1,1], block = [32,2,1]
```
Notice that this will get 2 warps where as the the 1 token decodes will get 4:
```console
0.28.377.692 I Q projection: MMVQ, tokens = 1, cc = 890
0.28.377.699 I MMVQ launch: type = q4_K, ncols_dst = 1, nwarps = 4, rows_per_block = 1, small_k = 0, halve_iters = 0, fusion = 0, table = 0, cc = 890, grid = [4096,1,1], block = [32,4,1]
```
If we try an force the 8 token decode to use 4 warps the output is:
```console
--- a/ggml/src/ggml-cuda/mmvq.cu
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -451,6 +451,9 @@ static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {

 static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
     if (table_id == MMVQ_PARAMETERS_GENERIC) {
+        if (type == GGML_TYPE_Q4_K && ncols_dst == 8) {
+            return 4;
+        }
         switch (ncols_dst) {
             case 1:
             case 2:

```
The output become:
```console
$ probe | jq -r '.choices[0].message.content' 
- The browser checks if it has a cached copy of the website's SSL/TLS certificate.  
- The browser initiates a TLS handshake with the server to establish a secure connection.  
- The server responds with its SSL/TLS certificate, which the browser verifies against a trusted root certificate authority.  
- If the certificate is valid, the browser and server agree on encryption settings and exchange keys to encrypt data.  
- The browser proceeds to request the website's content over the secure connection.
```
Notice that this now matches the baseline output:
```console
- The browser proceeds to request the website's content over the secure connection.
```

So this seems to be an issue with how floating point addition is not associative
and these minor difference are causing issues.
