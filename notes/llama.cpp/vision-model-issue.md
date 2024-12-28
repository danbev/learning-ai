### Llama 3.2 Vision Instruct Model issues

### Issue with callback in build_mllama
TODO: document the issue that I ran into with the callback in build_mllama.

### Preprocessing
So, what I have so far is that the model works with the image that I used during testing/development
which is this image:

![image](../images/eiffel-tower-3349075_1280.jpg)

Now, while this works and produces a pretty good output I've not been able to get it to
work with all images. I can get a pretty good response for a photo of the Golden Gate bridge
but if I try a close up on an apple I get a very poor response.

I've been looking at the preprocessing as it is somewhat complex and not something that I've
done before. TODO: link to documentation on the preprocessing.

After many iterations and trying various solutions, looking at how huggingface transformers
does their [preprocessing](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/image_processing_mllama.py), and also looking at how Ollama does their
[preprocessing](https://github.com/ollama/ollama/blob/main/model/mllama/imageproc.go) I'm trying
to rule out the preprocessing as the issue. To do this I've added some code to store the
input image bytes to a file after the preprocessing but before these are passed to the vision
encoder.

I've added the following in `src/llama-vision.cpp`

```c++
static int32_t encode_image_with_ca_vision(ca_context & ctx,
        llama_img img, std::vector<float> & output) {
    ...
    const auto & model = *ctx.model;
	size_t total_bytes = ggml_nbytes(inp_raw);
    ...

    // set raw input
    {
        struct ggml_tensor * inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        ggml_backend_tensor_set(inp_raw, img_batch[0].img->data, 0, ggml_nbytes(inp_raw));

        void* tensor_data = malloc(total_bytes);
        if (tensor_data != NULL) {
	    ggml_backend_tensor_get_async(backend, inp_raw, tensor_data, 0, total_bytes);
	    ggml_backend_sched_synchronize(ctx.sched);

	    // Write all bytes to file
	    FILE* fp = fopen("inp_raw.bin", "wb");
	    if (fp != NULL) {
            fwrite(tensor_data, 1, total_bytes, fp);
            fclose(fp);
	    }
	    free(tensor_data);
	}
```
Running this using this apple image:

![image](../images/apple.jpg)

This is the output from llama.cpp using this image:
```console
llama_new_context_with_model: graph splits = 64
token = 128006
token = 882
token = 128007
token = 271
token = 128256
token = 3923
token = 374
token = 304
token = 420
token = 2217
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
Loaded image: w=1500 h=1749 c=3
Chosen canvas: 1120 x 1120
Resized (no pad) to: 1120 x 960
Splitting to tiles => 2 x 2
Supported aspect ratios: size: 8
Aspect ratio ID: 6
n_tiles: 4, n_channels: 3, patch_size: 14, image_size: 560, n_patches: 1600, n_positions: 1601
num_padding_patches: 7
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ggml_gallocr_reserve_n: reallocating Metal buffer from size 0.00 MiB to 2839.12 MiB
ggml_gallocr_reserve_n: reallocating CPU buffer from size 0.00 MiB to 187.62 MiB
inp_raw tensor type: f32
inp_raw backend type: CPU
inp_raw[0] = 1.930336
inp_raw[1] = 1.930336
inp_raw[2] = 1.930336
inp_raw[3] = 1.930336
inp_raw[4] = 1.930336
inp_raw[5] = 1.930336
inp_raw[6] = 1.930336
inp_raw[7] = 1.930336
inp_raw[8] = 1.930336
inp_raw[9] = 1.930336
aspect_ratio_id = 6
output[0] = 6.341918
output[1] = 21.302219
output[2] = -1.246417
output[3] = 3.067833
output[4] = -2.978220
output[5] = -19.076042
output[6] = -2.420478
output[7] = 2.041078
output[8] = -1.945675
output[9] = 1.832796
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
n_img_tokens = 15052800
--------- use ca_patch_embd for K and V and store in kv_cache.layer[3] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[8] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[13] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[18] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[23] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[28] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[33] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[38] ------
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ca_patch_emd[0] = 6.341918
ca_patch_emd[1] = 21.302219
ca_patch_emd[2] = -1.246417
ca_patch_emd[3] = 3.067833
ca_patch_emd[4] = -2.978220
ca_patch_emd[5] = -19.076042
ca_patch_emd[6] = -2.420478
ca_patch_emd[7] = 2.041078
ca_patch_emd[8] = -1.945675
ca_patch_emd[9] = 1.832796
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image shows a picture of a tree with a brown trunk and green leaves.ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)


main: decoded 16 tokens in 2.10 s, speed: 7.61 t/s
```

Produces the following file `inp_raw.bin`:
```console
ls -l inp_raw.bin
-rw-r--r--@ 1 danbev  staff  15052800 Dec 28 10:32 inp_raw.bin

$ shasum inp_raw.bin
3037b8632b350a80a8385cad90e517db83932994  inp_raw.bin
```
Now, lets add the same to Ollama's code and see if its preprocessing produces the same
output. We have to modify the code a little but it is pretty similar and goes in the
file `llama/mllama.cpp`:
```c++
        size_t total_bytes = ggml_nbytes(inp_raw);
        void* tensor_data = malloc(total_bytes);
        if (tensor_data != NULL) {
            ggml_backend_tensor_get(inp_raw, tensor_data, 0, total_bytes);

            // Write all bytes to file
            FILE* fp = fopen("inp_raw.bin", "wb");
            if (fp != NULL) {
                fwrite(tensor_data, 1, total_bytes, fp);
                fclose(fp);
            }
            free(tensor_data);
        }
```

```console
n_tiles: 4, n_channels: 3, patch_size: 14, image_size: 560, num_patches: 1600, num_positions: 1601
num_padding_patches: 7
mllama_model_load: compute allocated memory: 2853.34 MB
time=2024-12-28T10:54:58.635+01:00 level=INFO source=server.go:594 msg="llama runner started in 1.26 seconds"
time=2024-12-28T10:54:58.635+01:00 level=INFO source=prompt.go:95 msg=Preprocess opts="&{Runner:{NumCtx:2048 NumBatch:512 NumGPU:-1 MainGPU:0 LowVRAM:false F16KV:false LogitsAll:false VocabOnly:false UseMMap:<nil> UseMLock:false NumThread:0} NumKeep:4 Seed:-1 NumPredict:-1 TopK:40 TopP:0.9 MinP:0 TypicalP:1 RepeatLastN:64 Temperature:0.6 RepeatPenalty:1.1 PresencePenalty:0 FrequencyPenalty:0 Mirostat:0 MirostatTau:5 MirostatEta:0.1 PenalizeNewline:true Stop:[]}"
time=2024-12-28T10:54:58.635+01:00 level=INFO source=imageproc.go:194 msg="Preprocessing image data"
time=2024-12-28T10:54:58.695+01:00 level=INFO source=imageproc.go:125 msg="Resize image" outputSize=(560,560) maxImageTiles=4
time=2024-12-28T10:54:58.695+01:00 level=INFO source=imageproc.go:45 msg="Get optimal tiled canvas" imageSize=(1500,1749) maxImageTiles=4 tileSize=560
time=2024-12-28T10:54:58.695+01:00 level=INFO source=imageproc.go:19 msg="Get supported aspect ratios" maxTiles=4
time=2024-12-28T10:54:58.695+01:00 level=INFO source=imageproc.go:30 msg="Get supported aspect ratios" ratios="[(1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (3,1) (4,1)]"
time=2024-12-28T10:54:58.695+01:00 level=INFO source=imageproc.go:132 msg="Aspect ratio" aspectRatio=(2,2)
time=2024-12-28T10:54:58.732+01:00 level=INFO source=imageproc.go:204 msg="Resized image" aspectRatio=(2,2) bounds=(0,0)-(960,1120)
time=2024-12-28T10:54:58.732+01:00 level=INFO source=imageproc.go:139 msg="Pad image" aspectRatio=(2,2)
time=2024-12-28T10:54:58.732+01:00 level=INFO source=imageproc.go:144 msg="Padded size" paddedSize=(1120,1120)
time=2024-12-28T10:54:58.732+01:00 level=INFO source=imageproc.go:147 msg="Dst bounds" dstBounds=(0,0)-(1120,1120)
time=2024-12-28T10:54:58.734+01:00 level=INFO source=imageproc.go:206 msg="Padded image" bounds=(0,0)-(1120,1120)
time=2024-12-28T10:54:58.734+01:00 level=INFO source=imageproc.go:176 msg="Pack images" aspectRatio=(2,2)
time=2024-12-28T10:54:58.734+01:00 level=INFO source=imageproc.go:155 msg="Split to tiles" numTilesSize=(2,2) bounds=(0,0)-(1120,1120)
time=2024-12-28T10:54:58.734+01:00 level=INFO source=imageproc.go:178 msg="Sub images" subImages="[0x1400057a0c0 0x1400057a100 0x1400057a140 0x1400057a1c0]"
time=2024-12-28T10:54:58.734+01:00 level=INFO source=images.go:61 msg="Normalize subimage" mean="[0.48145467 0.4578275 0.40821072]" std="[0.26862955 0.2613026 0.2757771]" channelFirst=true
time=2024-12-28T10:54:58.741+01:00 level=INFO source=images.go:61 msg="Normalize subimage" mean="[0.48145467 0.4578275 0.40821072]" std="[0.26862955 0.2613026 0.2757771]" channelFirst=true
time=2024-12-28T10:54:58.748+01:00 level=INFO source=images.go:61 msg="Normalize subimage" mean="[0.48145467 0.4578275 0.40821072]" std="[0.26862955 0.2613026 0.2757771]" channelFirst=true
time=2024-12-28T10:54:58.753+01:00 level=INFO source=images.go:61 msg="Normalize subimage" mean="[0.48145467 0.4578275 0.40821072]" std="[0.26862955 0.2613026 0.2757771]" channelFirst=true
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=0 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=1 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=2 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=3 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=4 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=5 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=6 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=7 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=8 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:211 msg="preprocessed value" index=9 value=1.9303361177444458
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:19 msg="Get supported aspect ratios" maxTiles=4
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:30 msg="Get supported aspect ratios" ratios="[(1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (3,1) (4,1)]"
time=2024-12-28T10:54:58.759+01:00 level=INFO source=imageproc.go:214 msg=AspectRationIndex: aspectRatioIndex=6
time=2024-12-28T10:54:58.768+01:00 level=INFO source=prompt.go:118 msg=image id=0 aspectRatio=6 len=15052800
time=2024-12-28T10:54:58.768+01:00 level=INFO source=routes.go:1544 msg="chat request" images=1 prompt="<|start_header_id|>user<|end_header_id|>\n\n[img-0]<|image|>What is in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
n_tiles: 4, n_channels: 3, patch_size: 14, image_size: 560, num_patches: 1600, num_positions: 1601
num_padding_patches: 7
input[0] = 1.930336
input[1] = 1.930336
input[2] = 1.930336
input[3] = 1.930336
input[4] = 1.930336
input[5] = 1.930336
input[6] = 1.930336
input[7] = 1.930336
input[8] = 1.930336
input[9] = 1.930336
```

The output for this image is:
```console
$ python python-ollama.py
model='llama3.2-vision:latest' created_at='2024-12-28T09:56:27.956901Z' done=True done_reason='stop' total_duration=90617492792 load_duration=1295202000 prompt_eval_count=18 prompt_eval_duration=82535000000 eval_count=68 eval_duration=6570000000 message=Message(role='assistant', content='The image features a red apple with two green leaves, possibly an apple cultivar named "Red Delicious." The apple has a reflective surface and a brown stem. Its deep red coloration suggests that it may be ripe or ready to eat. One of the leaves appears to have been bitten into on its right side, indicating possible consumption.', images=None, tool_calls=None)
```

And we can then inspect and generate a checksum for the generated file:
```console
$ ls -l inp_raw.bin
-rw-r--r--@ 1 danbev  staff  15052800 Dec 28 10:54 inp_raw.bin

shasum inp_raw.bin
ed90d9fd0b967add6f887ac9e65575ae9c73ece6  inp_raw.bin
```
Comparing both files (llama.cpp first followed by Ollama):
```console
3037b8632b350a80a8385cad90e517db83932994  inp_raw.bin

ed90d9fd0b967add6f887ac9e65575ae9c73ece6  inp_raw.bin
```
So we can see that we are not generating identical inputs to the model so there seems to be
something wrong with how we are preprocessing the image.

