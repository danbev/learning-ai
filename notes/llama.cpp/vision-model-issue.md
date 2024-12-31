### Llama 3.2 Vision Instruct Model issues
This is an issue I ran into while exploring the new Vision API in llama.cpp (not merged yet)
and trying to get Llama 3.2 Vision Instruct to work with the model.

While I can get the model to work with an image of the Eiffel Tower and also one of the Golden Gate
bridge, I'm having issues with other images.

### Issue with callback in build_mllama
TODO: document the issue that I ran into with the callback in build_mllama.

### Preprocessing
So, what I have so far is that the model works with the image that I used during testing/development
which is this image:

![image](../images/eiffel-tower-3349075_1280.jpg)

Now, while this works and produces a pretty good output:
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
Loaded image: w=1280 h=853 c=3
Chosen canvas: 1120 x 1120
Resized (no pad) to: 746 x 1120
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
inp_raw[0] = 1.156620
inp_raw[1] = 1.156620
inp_raw[2] = 1.156620
inp_raw[3] = 1.156620
inp_raw[4] = 1.156620
inp_raw[5] = 1.171218
inp_raw[6] = 1.171218
inp_raw[7] = 1.156620
inp_raw[8] = 1.171218
inp_raw[9] = 1.171218
aspect_ratio_id = 6
output[0] = 10.172008
output[1] = 15.932920
output[2] = -3.465006
output[3] = 5.908316
output[4] = -1.494109
output[5] = -14.418842
output[6] = -0.452144
output[7] = 1.189293
output[8] = -8.067196
output[9] = -0.785143
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
ca_patch_emd[0] = 10.172008
ca_patch_emd[1] = 15.932920
ca_patch_emd[2] = -3.465006
ca_patch_emd[3] = 5.908316
ca_patch_emd[4] = -1.494109
ca_patch_emd[5] = -14.418842
ca_patch_emd[6] = -0.452144
ca_patch_emd[7] = 1.189293
ca_patch_emd[8] = -8.067196
ca_patch_emd[9] = -0.785143
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image shows a photograph of the Eiffel Tower in Paris, France.
The tower is in the center of the image and is made of metal with a light brown color. It has a long, thin, rectangular shape and is standing on a square base. The background of the image is a
main: decoded 60 tokens in 7.80 s, speed: 7.70 t/s
```
I've not been able to get it to work with all images.

I can get a pretty good response for a photo of the Golden Gate bridge also but if I try a close
up on an apple I get a very poor response (and other images that I've tried).

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
(llama.cpp)
3037b8632b350a80a8385cad90e517db83932994  inp_raw.bin

(Ollama's)
ed90d9fd0b967add6f887ac9e65575ae9c73ece6  inp_raw.bin
```
_So we can see that we are not generating identical inputs to the model so there seems to be
something wrong with how we are preprocessing the image_.

The llama.cpp code that was used for the pre-processing was ported by looking at the Huggingface transformers
code. Lets try doing the same but using Ollama's preprocessing code and see if we can get an exact match.

```console
Image loaded, width=1500, height=1749, channels=3
Calculating optimal canvas for image 1500x1749 with max_tiles=4, tile_size=560
Possible ratios and their canvas sizes:
  Ratio 1x1 -> Canvas 560x560 (scale=0.320)
  Ratio 1x2 -> Canvas 560x1120 (scale=0.373)
  Ratio 1x3 -> Canvas 560x1680 (scale=0.373)
  Ratio 1x4 -> Canvas 560x2240 (scale=0.373)
  Ratio 2x1 -> Canvas 1120x560 (scale=0.320)
  Ratio 2x2 -> Canvas 1120x1120 (scale=0.640)
  Ratio 3x1 -> Canvas 1680x560 (scale=0.320)
  Ratio 4x1 -> Canvas 2240x560 (scale=0.320)
Scale selection (has_upscale=0, selected_scale=0.640):
Selected canvas 1120x1120 (area=1254400)
Canvas size: 1120 x 1120
Scaled size: 1120 x 1120
Selected aspect ratio index: 6
Subdividing into 2x2 tiles (tile_size=560)
Processing tile at 0,0
Processing tile at 1,0
Processing tile at 0,1
Processing tile at 1,1
Aspect ratio: 6
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
output[0] = 6.445750
output[1] = 20.882206
output[2] = -2.809249
output[3] = 1.198682
output[4] = -3.665206
output[5] = -18.842869
output[6] = -3.300013
output[7] = -0.508817
output[8] = -0.843601
output[9] = 1.558197
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
ca_patch_emd[0] = 6.445750
ca_patch_emd[1] = 20.882206
ca_patch_emd[2] = -2.809249
ca_patch_emd[3] = 1.198682
ca_patch_emd[4] = -3.665206
ca_patch_emd[5] = -18.842869
ca_patch_emd[6] = -3.300013
ca_patch_emd[7] = -0.508817
ca_patch_emd[8] = -0.843601
ca_patch_emd[9] = 1.558197
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image shows a picture of a tree in a field. The tree is in the center of the image and is surrounded by a field of grass. The tree is a light brown color with a darker brown trunk and branches. The leaves are a lighter shade of brown. The tree is standing in a
main: decoded 60 tokens in 7.79 s, speed: 7.70 t/s
```

And the checksum for the file:
```console
$ shasum inp_raw.bin
(llama.cpp)
bb09af72e28d1c837ade0011ad33df5183520569  inp_raw.bin

(Ollama's)
ed90d9fd0b967add6f887ac9e65575ae9c73ece6  inp_raw.bin
```
Now, I think we should be able to get the exact same input (output from the preprocessing) for
our model. Things that can effect this are the resizing, tiling, normalization, channels used, channel
order.

I've tried more variant but still not been able to get the same output from the preprocessing.
The following is debug information for llama.cpp:
```console
cat debug_info.txt
Canvas size: 1120 x 1120
Tile size: 560
Number of tiles: 4

Tile 0:
  Raw size: 940800 bytes
  Normalized size: 940800 floats
  First 10 raw values:
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
  First 10 normalized values:
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336

Tile 1:
  Raw size: 940800 bytes
  Normalized size: 940800 floats
  First 10 raw values:
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
  First 10 normalized values:
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336

Tile 2:
  Raw size: 940800 bytes
  Normalized size: 940800 floats
  First 10 raw values:
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
  First 10 normalized values:
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336
    1.930336

Tile 3:
  Raw size: 940800 bytes
  Normalized size: 940800 floats
  First 10 raw values:
    [153,   6,  14]
    [155,   6,  16]
    [155,   5,  16]
    [154,   4,  15]
    [153,   3,  14]
    [153,   3,  14]
    [152,   2,  13]
    [151,   1,  12]
    [151,   1,  12]
    [151,   1,  12]
  First 10 normalized values:
    0.441297
    0.470494
    0.470494
    0.455895
    0.441297
    0.441297
    0.426698
    0.412100
    0.412100
    0.412100

(venv) $ shasum debug_canvas.bin
34bb725f3ef4eded56c969239f6283c2b37fcfe4  debug_canvas.bin
(venv) $ shasum debug_tile_0.bin
63bb47b7635e137ae60899405711888287a11b4a  debug_tile_0.bin
(venv) $ shasum debug_tile_1.bin
b0fb51b0cf34d89569f91b18a505b53ed04b6765  debug_tile_1.bin
(venv) $ shasum debug_tile_2.bin
60e9e573b4818cfe87848a97cdfc002c49852e33  debug_tile_2.bin
(venv) $ shasum debug_tile_3.bin
d598d3fdacbb3d57128f569311df3fb60ddf798c  debug_tile_3.bin
(venv) $ shasum debug_tile_norm_0.bin
68363050417b54b506f586735e26aa23013bb692  debug_tile_norm_0.bin
(venv) $ shasum debug_tile_norm_1.bin
4cef3b1ee410c2c9c9cd3036fc5a9b48f287f5cf  debug_tile_norm_1.bin
(venv) $ shasum debug_tile_norm_2.bin
ad26957a4c35818854f982754dbf27092d78d303  debug_tile_norm_2.bin
(venv) $ shasum debug_tile_norm_3.bin
fe722e798e03dab253d65086a154219cc3da67a8  debug_tile_norm_3.bin
```

```console
(venv) $ cat ollama_debug_info.txt
Original image size: 1500x1749
Number of tiles: 4

Tile 0:
  Size: 560x560
  First 10 pixels (RGB):
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]
    [255, 255, 255]

Tile 1:
  Size: 1120x560
  First 10 pixels (RGB):
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]

Tile 2:
  Size: 560x1120
  First 10 pixels (RGB):
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]

Tile 3:
  Size: 1120x1120
  First 10 pixels (RGB):
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]
    [  0,   0,   0]

First 30 normalized values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336
  [10] = 1.930336
  [11] = 1.930336
  [12] = 1.930336
  [13] = 1.930336
  [14] = 1.930336
  [15] = 1.930336
  [16] = 1.930336
  [17] = 1.930336
  [18] = 1.930336
  [19] = 1.930336
  [20] = 1.930336
  [21] = 1.930336
  [22] = 1.930336
  [23] = 1.930336
  [24] = 1.930336
  [25] = 1.930336
  [26] = 1.930336
  [27] = 1.930336
  [28] = 1.930336
  [29] = 1.930336

(venv) $ shasum ollama_normalized.bin
ed90d9fd0b967add6f887ac9e65575ae9c73ece6  ollama_normalized.bin
(venv) $ shasum ollama_original.bin
2b876bed1d6c15e5bd387639568ff9d51572de18  ollama_original.bin
(venv) $ shasum ollama_resized.bin
001377e709666e03d4bd769e8f3c9599e989274f  ollama_resized.bin
(venv) $ shasum ollama_tile_0.bin
213faeea9d8bb8b246409e460f142058fa698006  ollama_tile_0.bin
(venv) $ shasum ollama_tile_1.bin
067bc8866e9b775675331204330e53271612c860  ollama_tile_1.bin
(venv) $ shasum ollama_tile_2.bin
250e64e774db72e369cef752956cfdbcbaa5dea3  ollama_tile_2.bin
(venv) $ shasum ollama_tile_3.bin
e279c68f44521a35940c3df87c125916af7a9436  ollama_tile_3.bin
```

So I've compared the resized image:
```console
(venv) $ ls -l resized_image.bin ~/work/ai/ollama/ollama_resized.bin
-rw-r--r--@ 1 danbev  staff  3225600 Dec 29 08:57 /Users/danbev/work/ai/ollama/ollama_resized.bin
-rw-r--r--@ 1 danbev  staff  3225600 Dec 29 08:59 resized_image.bin
```
And these only differ for a single pixel:
```console
(venv) $ cmp resized_image.bin ~/work/ai/ollama/ollama_resized.bin
resized_image.bin /Users/danbev/work/ai/ollama/ollama_resized.bin differ: char 59430, line 1
```
So I'm going to move on to the next preprocessing step which is the padding:
```console

### Preprocessing alignment with Ollama
I've gone through how the preprocessing is done in [Ollama](../ollama-image-preprocessing.md) and
the function that does this looks like this:
```go
func Preprocess(imageData io.Reader) ([]float32, map[string]any, error) {
        slog.Info("Preprocessing image data")
        outputSize := image.Point{560, 560}
        maxTiles := 4

        img, format, err := image.Decode(imageData)
        if err != nil {
                return nil, nil, fmt.Errorf("failed to decode image: %w", err)
        }

        newImage, aspectRatio := resizeImage(img, format, outputSize, maxTiles)
        slog.Info("Resized image", "aspectRatio", aspectRatio, "bounds", newImage.Bounds())
        newImage = padImage(newImage, outputSize, aspectRatio)
        slog.Info("Padded image", "bounds", newImage.Bounds())

        data := packImages(newImage, aspectRatio)
        // Print first 10 float32 values
        for i := 0; i < 10 && i < len(data); i++ {
            slog.Info("preprocessed value", "index", i, "value", data[i])
        }
        aspectRatioIndex := slices.Index(getSupportedAspectRatios(maxTiles), aspectRatio) + 1
        slog.Info("AspectRationIndex:", "aspectRatioIndex", aspectRatioIndex)

        opts := map[string]any{
                "aspectRatioIndex": aspectRatioIndex,
        }

        return data, opts, nil
}
```
`resizeImage` does the 

```
msg="Canvas size" canvasSize=(1120,1120)
```

This is the output of the first 10 values of the preprocessed image in Ollama:
```console
Tile 0 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 1 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 2 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 3 first 10 values:
  [0] = 0.485092
  [1] = 0.455895
  [2] = 0.426698
  [3] = 0.426698
  [4] = 0.426698
  [5] = 0.426698
  [6] = 0.426698
  [7] = 0.426698
  [8] = 0.485092
  [9] = 0.514289
```
And this is for llama.cpp:
```console
Tile 0 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 1 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 2 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 3 first 10 values:
  [0] = 0.441297
  [1] = 0.470494
  [2] = 0.470494
  [3] = 0.470494
  [4] = 0.455895
  [5] = 0.441297
  [6] = 0.441297
  [7] = 0.426698
  [8] = 0.412100
  [9] = 0.412100
```
The image in question is the apple.jpg image which has a white background so it is not that surprising that the first 10 values are the same.

```
  Ollama         llama.cpp
  [0] = 0.485092 [0] = 0.441297
  [1] = 0.455895 [1] = 0.470494
  [2] = 0.426698 [2] = 0.470494
  [3] = 0.426698 [3] = 0.470494
  [4] = 0.426698 [4] = 0.455895
  [5] = 0.426698 [5] = 0.441297
  [6] = 0.426698 [6] = 0.441297
  [7] = 0.426698 [7] = 0.426698
  [8] = 0.485092 [8] = 0.412100
  [9] = 0.514289 [9] = 0.412100
```
Why are these different?  These are the normalized values so perhaps we should look at how
that is being done.

  
```console
llama.cpp:
output[0] = 6.445750
output[1] = 20.882206
output[2] = -2.809249
output[3] = 1.198682
output[4] = -3.665206
output[5] = -18.842869
output[6] = -3.300013
output[7] = -0.508817
output[8] = -0.843601
output[9] = 1.558197

Ollama:
ca_patch_emd[0] = 6.947874
ca_patch_emd[1] = 17.099964
ca_patch_emd[2] = -2.731302
ca_patch_emd[3] = 1.561556
ca_patch_emd[4] = -0.641592
ca_patch_emd[5] = -18.051983
ca_patch_emd[6] = -0.249803
ca_patch_emd[7] = 1.927986
ca_patch_emd[8] = -1.114733
ca_patch_emd[9] = 0.568076
```

Hmm, I notice something while looking at original sizes input images:
```
apple: width:1500, height: 1749
canvas: 1120x1120

Eiffel: width:1280, height: 853
canvas: 1120x1120

Golden: width:1024, height: 683
```

```console
Calculating optimal canvas for image 669x446 with max_tiles=4, tile_size=560
Possible ratios and their canvas sizes:
  Ratio 1x1 -> Canvas 560x560 (scale_w=0.837 scale_h=1.256   selected=0.837)
  Ratio 1x2 -> Canvas 560x1120 (scale_w=0.837 scale_h=2.511  selected=0.837)
  Ratio 1x3 -> Canvas 560x1680 (scale_w=0.837 scale_h=3.767  selected=0.837)
  Ratio 1x4 -> Canvas 560x2240 (scale_w=0.837 scale_h=5.022  selected=0.837)
  Ratio 2x1 -> Canvas 1120x560 (scale_w=1.674 scale_h=1.256  selected=1.256)
  Ratio 2x2 -> Canvas 1120x1120 (scale_w=1.674 scale_h=2.511 selected=1.674)
  Ratio 3x1 -> Canvas 1680x560 (scale_w=2.511 scale_h=1.256  selected=1.256)
  Ratio 4x1 -> Canvas 2240x560 (scale_w=3.348 scale_h=1.256  selected=1.256)
Selected scale: 1.255605 (upscale=1)
Candidate canvas 1120x560 (area=627200)
Candidate canvas 1680x560 (area=940800)
Candidate canvas 2240x560 (area=1254400)
Final selected canvas 1120x560
Get image size fit to canvas: img=669x446, canvas=1120x560, tile=560
After clamp: target=669x560
After aspect adjustment: 669x446
Final size: 1120x560

```
```console
msg="Possible canvas size" pcs=(560,560) scaleHeight=1.2556053811659194 scaleWidth=0.8370702541106129
msg="Possible canvas size" pcs=(560,1120) scaleHeight=2.5112107623318387 scaleWidth=0.8370702541106129
msg="Possible canvas size" pcs=(560,1680) scaleHeight=3.766816143497758 scaleWidth=0.8370702541106129
msg="Possible canvas size" pcs=(560,2240) scaleHeight=5.022421524663677 scaleWidth=0.8370702541106129
msg="Possible canvas size" pcs=(1120,560) scaleHeight=1.2556053811659194 scaleWidth=1.6741405082212257
msg="Possible canvas size" pcs=(1120,1120) scaleHeight=2.5112107623318387 scaleWidth=1.6741405082212257
msg="Possible canvas size" pcs=(1680,560) scaleHeight=1.2556053811659194 scaleWidth=2.5112107623318387
msg="Possible canvas size" pcs=(2240,560) scaleHeight=1.2556053811659194 scaleWidth=3.3482810164424515
msg=Scale index=0 value=0.8370702541106129
msg=Scale index=1 value=0.8370702541106129
msg=Scale index=2 value=0.8370702541106129
msg=Scale index=3 value=0.8370702541106129
msg=Scale index=4 value=1.2556053811659194
msg=Scale index=5 value=1.6741405082212257
msg=Scale index=6 value=1.2556053811659194
msg=Scale index=7 value=1.2556053811659194
"Selected scale" selectedScale=1.2556053811659194
msg="Selected canvas" selectedCanvas=(1120,560)
msg="Canvas size" canvasSize=(1120,560)
```
So the canvas selected is the same. And the select scale also match.

```console
msg="Get image size fit to canvas" imageSize=(669,446) canvasSize=(1120,560) tileSize=560
msg="Get image size fit to canvas" imageSize=(669,446) canvasSize=(1120,560) tileSize=560 targetWidth=669 targetHeight=56"
msg="Get image size fit to canvas" w=669 h=446
msg="New scaled image size" newSize=(669,446)
```
```console
Get image size fit to canvas: img=669x446, canvas=1120x560, tile=560
After clamp: target=669x560
After aspect adjustment: 669x446
```

So ollama seem to use the 669x446 image size when resizing:
```conosle
msg="Resize image" method=0 newSize=(669,446) src=(0,0)-(669,446)
msg="Resize image" method=0 newSize=(669,446) dst=(0,0)-(669,446)
msg="Resized image" aspectRatio=(2,1) bounds=(0,0)-(669,446)
```
But I'm currently doing:
```c++
    int final_width = ((target_width + tile_size - 1) / tile_size) * tile_size;
    int final_height = ((target_height + tile_size - 1) / tile_size) * tile_size;
```
If I don't do this I get NaN values but that might an issue later in the code.

```console
msg="Pad image" aspectRatio=(2,1)
msg="Padded size" paddedSize=(1120,560)
```

```console
msg="Split to tiles" numTilesSize=(2,1) bounds=(0,0)-(1120,560)
```
```console
Splitting into 2x1 tiles
```

Tile 0 first 10 values:
  [0] = -1.222924
  [1] = -1.222924
  [2] = -1.222924
  [3] = -1.222924
  [4] = -1.208326
  [5] = -1.208326
  [6] = -1.208326
  [7] = -1.193727
  [8] = -1.222924
  [9] = -1.222924

Tile 0 first 10 values:
  [0] = -1.222924
  [1] = -1.222924
  [2] = -1.222924
  [3] = -1.208326
  [4] = -1.208326
  [5] = -1.208326
  [6] = -1.208326
  [7] = -1.193727
  [8] = -1.222924
  [9] = -1.222924
OK

Tile 1 first 10 values:
  [0] = 0.394014
  [1] = 0.394014
  [2] = 0.394014
  [3] = 0.394014
  [4] = 0.409022
  [5] = 0.409022
  [6] = 0.409022
  [7] = 0.409022
  [8] = 0.424029
  [9] = 0.424029

  Tile 1 first 10 values:
  [0] = -1.047743
  [1] = -1.047743
  [2] = -1.047743
  [3] = -1.047743
  [4] = -1.033144
  [5] = -1.033144
  [6] = -1.033144
  [7] = -1.033144
  [8] = -1.018546
  [9] = -1.018546
NOK - The values from llama.cpp (the first values) are way off. We should also bre getting negative values
here and not positive values.


```console
msg="Pack images" aspectRatio=(2,1)
msg="Split to tiles" numTilesSize=(2,1) bounds=(0,0)-(1120,560)
msg="Image size" width=1120 height=560
msg="Tile size" tileWidth=560 tileHeight=560
msg="Processing tile row" row=0
msg="Processing tile column" column=0
msg="Tile rect" rect=(0,0)-(560,560)
msg="Logging pixel data" tileBounds=(0,0)-(560,560)
msg="Tile bounds" bounds=(0,0)-(560,560)
msg=Pixel index=0 x=0 y=0 R=39 G=137 B=210 A=255
msg=Pixel index=1 x=1 y=0 R=39 G=137 B=210 A=255
msg=Pixel index=2 x=2 y=0 R=39 G=137 B=210 A=255
msg=Pixel index=3 x=3 y=0 R=39 G=137 B=210 A=255
msg=Pixel index=4 x=4 y=0 R=40 G=138 B=211 A=255
msg=Pixel index=5 x=5 y=0 R=40 G=138 B=211 A=255
msg=Pixel index=6 x=6 y=0 R=40 G=138 B=211 A=255
msg=Pixel index=7 x=7 y=0 R=41 G=139 B=212 A=255
msg=Pixel index=8 x=8 y=0 R=39 G=135 B=209 A=255
msg=Pixel index=9 x=9 y=0 R=39 G=135 B=209 A=255
msg="Processing tile column" column=1
msg="Tile rect" rect=(560,0)-(1120,560)
msg="Logging pixel data" tileBounds=(560,0)-(1120,560)
msg="Tile bounds" bounds=(560,0)-(1120,560)
msg=Pixel index=0 x=560 y=0 R=51 G=143 B=206 A=255
msg=Pixel index=1 x=561 y=0 R=51 G=143 B=206 A=255
msg=Pixel index=2 x=562 y=0 R=51 G=143 B=206 A=255
msg=Pixel index=3 x=563 y=0 R=51 G=143 B=206 A=255
msg=Pixel index=4 x=564 y=0 R=52 G=144 B=207 A=255
msg=Pixel index=5 x=565 y=0 R=52 G=144 B=207 A=255
msg=Pixel index=6 x=566 y=0 R=52 G=144 B=207 A=255
msg=Pixel index=7 x=567 y=0 R=52 G=144 B=207 A=255
msg=Pixel index=8 x=568 y=0 R=53 G=145 B=208 A=255
msg=Pixel index=9 x=569 y=0 R=53 G=145 B=208 A=255
```console

```console
Processing tile [0,0], source region: x=0-559, y=0-559
  Tile[0,0] at (0,0): src=(39,137,210) -> dst=(39,137,210)
  Tile[0,0] at (1,0): src=(39,137,210) -> dst=(39,137,210)
  Tile[0,0] at (2,0): src=(39,137,210) -> dst=(39,137,210)
  Tile[0,0] at (0,1): src=(39,137,210) -> dst=(39,137,210)
  Tile[0,0] at (1,1): src=(39,137,210) -> dst=(39,137,210)
  Tile[0,0] at (2,1): src=(40,138,211) -> dst=(40,138,211)
  Tile[0,0] at (0,2): src=(40,138,211) -> dst=(40,138,211)
  Tile[0,0] at (1,2): src=(40,138,211) -> dst=(40,138,211)
  Tile[0,0] at (2,2): src=(40,138,211) -> dst=(40,138,211)

Processing tile [1,0], source region: x=560-1119, y=0-559
  Tile[1,0] at (0,0): src=(51,143,206) -> dst=(51,143,206)
  Tile[1,0] at (1,0): src=(51,143,206) -> dst=(51,143,206)
  Tile[1,0] at (2,0): src=(51,143,206) -> dst=(51,143,206)
  Tile[1,0] at (0,1): src=(52,144,207) -> dst=(52,144,207)
  Tile[1,0] at (1,1): src=(52,144,207) -> dst=(52,144,207)
  Tile[1,0] at (2,1): src=(52,144,207) -> dst=(52,144,207)
  Tile[1,0] at (0,2): src=(53,145,208) -> dst=(53,145,208)
  Tile[1,0] at (1,2): src=(53,145,208) -> dst=(53,145,208)
  Tile[1,0] at (2,2): src=(53,145,208) -> dst=(53,145,208)
```
Notice that the values are the same.


Processing tile 0
Raw pixel at (0,0):  R=39 G=137 B=210
After norm at (0,0): R=-1.222924 G=0.303967 B=1.505994
Raw pixel at (1,0):  R=39 G=137 B=210
After norm at (1,0): R=-1.222924 G=0.303967 B=1.505994
Raw pixel at (2,0):  R=39 G=137 B=210
After norm at (2,0): R=-1.222924 G=0.303967 B=1.505994
Raw pixel at (3,0):  R=40 G=138 B=211
After norm at (3,0): R=-1.208326 G=0.318975 B=1.520214
Raw pixel at (4,0):  R=40 G=138 B=211
After norm at (4,0): R=-1.208326 G=0.318975 B=1.520214

Processing tile 1
Raw pixel at (0,0):  R=51 G=143 B=206
After norm at (0,0): R=-1.047743 G=0.394014 B=1.449114
Raw pixel at (1,0):  R=51 G=143 B=206
After norm at (1,0): R=-1.047743 G=0.394014 B=1.449114
Raw pixel at (2,0):  R=51 G=143 B=206
After norm at (2,0): R=-1.047743 G=0.394014 B=1.449114
Raw pixel at (3,0):  R=51 G=143 B=206
After norm at (3,0): R=-1.047743 G=0.394014 B=1.449114
Raw pixel at (4,0):  R=52 G=144 B=207
After norm at (4,0): R=-1.033144 G=0.409022 B=1.463334

Notice that the Red values in Tile 1 are the same as the values in 
Ollamas Tile 1:
  [0] = -1.047743
  [1] = -1.047743
  [2] = -1.047743
  [3] = -1.047743
  [4] = -1.033144
  [5] = -1.033144
  [6] = -1.033144
  [7] = -1.033144
  [8] = -1.018546
  [9] = -1.018546

But when we look at the final tiles we are getting the green values for 
tile 1 instead:
  [0] = 0.394014
  [1] = 0.394014
  [2] = 0.394014
  [3] = 0.394014
  [4] = 0.409022
  [5] = 0.409022
  [6] = 0.409022
  [7] = 0.409022
  [8] = 0.424029
  [9] = 0.424029

### Eiffel Tower preprocessing
llama.cpp:
```console
Tile 0 first 10 values:
  [0] = 1.156620
  [1] = 1.156620
  [2] = 1.156620
  [3] = 1.156620
  [4] = 1.156620
  [5] = 1.171218
  [6] = 1.171218
  [7] = 1.156620
  [8] = 1.171218
  [9] = 1.171218

Tile 1 first 10 values:
  [0] = -0.565995
  [1] = -0.565995
  [2] = -0.565995
  [3] = -0.565995
  [4] = -0.580593
  [5] = -0.580593
  [6] = -0.580593
  [7] = -0.580593
  [8] = -0.580593
  [9] = -0.580593

Tile 2 first 10 values:
  [0] = 0.966840
  [1] = 0.952242
  [2] = 0.937643
  [3] = 0.966840
  [4] = 0.966840
  [5] = 0.937643
  [6] = 0.893848
  [7] = 0.879250
  [8] = 0.893848
  [9] = 0.908446

Tile 3 first 10 values:
  [0] = 1.682163
  [1] = 1.638368
  [2] = 1.463187
  [3] = 1.565376
  [4] = 1.536179
  [5] = 1.638368
  [6] = 1.594572
  [7] = 1.579974
  [8] = 1.550777
  [9] = 1.609171
```
Ollama:
```console
Tile 0 first 10 values:
  [0] = 1.156620
  [1] = 1.156620
  [2] = 1.156620
  [3] = 1.156620
  [4] = 1.156620
  [5] = 1.171218
  [6] = 1.171218
  [7] = 1.156620
  [8] = 1.171218
  [9] = 1.171218

Tile 1 first 10 values:
  [0] = -0.565995
  [1] = -0.565995
  [2] = -0.565995
  [3] = -0.565995
  [4] = -0.580593
  [5] = -0.580593
  [6] = -0.580593
  [7] = -0.580593
  [8] = -0.580593
  [9] = -0.580593

Tile 2 first 10 values:
  [0] = 0.981439
  [1] = 0.952242
  [2] = 0.952242
  [3] = 0.966840
  [4] = 0.966840
  [5] = 0.937643
  [6] = 0.893848
  [7] = 0.893848
  [8] = 0.893848
  [9] = 0.908446

Tile 3 first 10 values:
  [0] = 1.667565
  [1] = 1.579974
  [2] = 1.521580
  [3] = 1.521580
  [4] = 1.492383
  [5] = 1.536179
  [6] = 1.492383
  [7] = 1.579974
  [8] = 1.536179
  [9] = 1.594572

```

### Empire State Building preprocessing
llama.cpp
```console
Tile 0 first 10 values:
  [0] = -1.222924
  [1] = -1.222924
  [2] = -1.222924
  [3] = -1.208326
  [4] = -1.208326
  [5] = -1.208326
  [6] = -1.208326
  [7] = -1.193727
  [8] = -1.222924
  [9] = -1.222924

Tile 1 first 10 values:
  [0] = -1.047743
  [1] = -1.047743
  [2] = -1.047743
  [3] = -1.047743
  [4] = -1.033144
  [5] = -1.033144
  [6] = -1.033144
  [7] = -1.033144
  [8] = -1.018546
  [9] = -1.018546

Tile 2 first 10 values:
  [0] = 0.000000
  [1] = 0.000000
  [2] = 0.000000
  [3] = 0.000000
  [4] = 0.000000
  [5] = 0.000000
  [6] = 0.000000
  [7] = 0.000000
  [8] = 0.000000
  [9] = 0.000000

Tile 3 first 10 values:
  [0] = 0.000000
  [1] = 0.000000
  [2] = 0.000000
  [3] = 0.000000
  [4] = 0.000000
  [5] = 0.000000
  [6] = 0.000000
  [7] = 0.000000
  [8] = 0.000000
  [9] = 0.000000
n_positions bytes: 6404
aspect_ratio_id = 5
```
Ollama:
```console
Tile 0 first 10 values:
  [0] = -1.222924
  [1] = -1.222924
  [2] = -1.222924
  [3] = -1.222924
  [4] = -1.208326
  [5] = -1.208326
  [6] = -1.208326
  [7] = -1.193727
  [8] = -1.222924
  [9] = -1.222924

Tile 1 first 10 values:
  [0] = -1.047743
  [1] = -1.047743
  [2] = -1.047743
  [3] = -1.047743
  [4] = -1.033144
  [5] = -1.033144
  [6] = -1.033144
  [7] = -1.033144
  [8] = -1.018546
  [9] = -1.018546

Tile 2 first 10 values:
  [0] = 0.000000
  [1] = 0.000000
  [2] = 0.000000
  [3] = 0.000000
  [4] = 0.000000
  [5] = 0.000000
  [6] = 0.000000
  [7] = 0.000000
  [8] = 0.000000
  [9] = 0.000000

Tile 3 first 10 values:
  [0] = 0.000000
  [1] = 0.000000
  [2] = 0.000000
  [3] = 0.000000
  [4] = 0.000000
  [5] = 0.000000
  [6] = 0.000000
  [7] = 0.000000
  [8] = 0.000000
  [9] = 0.000000
num_positions bytes: 6404
aspect_ratio_id: 5
```

### Apple preprocessing
llama.cpp:
```console
Tile 0 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 1 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 2 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 3 first 10 values:
  [0] = 0.485092
  [1] = 0.470494
  [2] = 0.441297
  [3] = 0.426698
  [4] = 0.426698
  [5] = 0.426698
  [6] = 0.426698
  [7] = 0.426698
  [8] = 0.499690
  [9] = 0.514289
n_positions bytes: 6404
aspect_ratio_id = 6
```
Ollama.:
```console
Tile 0 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 1 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 2 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 3 first 10 values:
  [0] = 0.485092
  [1] = 0.455895
  [2] = 0.426698
  [3] = 0.426698
  [4] = 0.426698
  [5] = 0.426698
  [6] = 0.426698
  [7] = 0.426698
  [8] = 0.485092
  [9] = 0.514289
num_positions bytes: 6404
aspect_ratio_id: 6
```

### Vision Model tensors
I had quantized the tensors for the model, both the vision models tensors and the language models
tensors. But I noticed that Ollama, which has two models, does not quantize the vision models tensors.
Fixing this in the quantization logic enabled me to sort of get a response for the apple image which
I was not able to previously.

Ollama (Eiffel Tower):
```console
output[0] = 10.534190
output[1] = 13.177908
output[2] = -4.533195
output[3] = 4.155879
output[4] = -1.235688
output[5] = -14.886117
output[6] = -1.362531
output[7] = 0.795458
output[8] = -4.832182
output[9] = -1.020501
```
llama.cpp (Eiffel Tower):
```console
output[0] = 9.534625
output[1] = 15.786053
output[2] = -3.500089
output[3] = 5.841105
output[4] = -1.771145
output[5] = -13.647954
output[6] = -0.700998
output[7] = 0.940367
output[8] = -8.735542
output[9] = -0.055729
```

llama.cpp (Empirs State Building):
```console
Tile 0 first 10 values:
  [0] = -1.222924
  [1] = -1.222924
  [2] = -1.222924
  [3] = -1.208326
  [4] = -1.208326
  [5] = -1.208326
  [6] = -1.208326
  [7] = -1.193727
  [8] = -1.222924
  [9] = -1.222924

Tile 1 first 10 values:
  [0] = -1.047743
  [1] = -1.047743
  [2] = -1.047743
  [3] = -1.047743
  [4] = -1.033144
  [5] = -1.033144
  [6] = -1.033144
  [7] = -1.033144
  [8] = -1.018546
  [9] = -1.018546

Tile 2 first 10 values:
  [0] = 0.000000
  [1] = 0.000000
  [2] = 0.000000
  [3] = 0.000000
  [4] = 0.000000
  [5] = 0.000000
  [6] = 0.000000
  [7] = 0.000000
  [8] = 0.000000
  [9] = 0.000000

Tile 3 first 10 values:
  [0] = 0.000000
  [1] = 0.000000
  [2] = 0.000000
  [3] = 0.000000
  [4] = 0.000000
  [5] = 0.000000
  [6] = 0.000000
  [7] = 0.000000
  [8] = 0.000000
  [9] = 0.000000
n_positions bytes: 6404
aspect_ratio_id = 5

output[0] = 11.987567
output[1] = 18.092655
output[2] = -1.181066
output[3] = 5.768115
output[4] = 0.949257
output[5] = -16.738150
output[6] = 1.604087
output[7] = -0.944634
output[8] = -10.563751
output[9] = -1.891053
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
n_img_tokens = 1
--------- use ca_patch_embd for K and V and store in kv_cache.layer[3] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[8] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[13] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[18] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[23] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[28] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[33] ------
--------- use ca_patch_embd for K and V and store in kv_cache.layer[38] ------
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ca_patch_emd[0] = 11.987567
ca_patch_emd[1] = 18.092655
ca_patch_emd[2] = -1.181066
ca_patch_emd[3] = 5.768115
ca_patch_emd[4] = 0.949257
ca_patch_emd[5] = -16.738150
ca_patch_emd[6] = 1.604087
ca_patch_emd[7] = -0.944634
ca_patch_emd[8] = -10.563751
ca_patch_emd[9] = -1.891053
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image shows a landscape of a cityscape of a park with a walkwayggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 and a tree. The main body of the text is not visible and the text is in the background. The text is in white and the background is dark grayggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
. The background of the text is dark brown. The text
main: decoded 60 tokens in 7.77 s, speed: 7.72 t/s

```
ollama (Empire State Building):
```console
Tile 0 first 10 values:
  [0] = -1.222924
  [1] = -1.222924
  [2] = -1.222924
  [3] = -1.222924
  [4] = -1.208326
  [5] = -1.208326
  [6] = -1.208326
  [7] = -1.193727
  [8] = -1.222924
  [9] = -1.222924

Tile 1 first 10 values:
  [0] = -1.047743
  [1] = -1.047743
  [2] = -1.047743
  [3] = -1.047743
  [4] = -1.033144
  [5] = -1.033144
  [6] = -1.033144
  [7] = -1.033144
  [8] = -1.018546
  [9] = -1.018546

Tile 2 first 10 values:
  [0] = 0.000000
  [1] = 0.000000
  [2] = 0.000000
  [3] = 0.000000
  [4] = 0.000000
  [5] = 0.000000
  [6] = 0.000000
  [7] = 0.000000
  [8] = 0.000000
  [9] = 0.000000

Tile 3 first 10 values:
  [0] = 0.000000
  [1] = 0.000000
  [2] = 0.000000
  [3] = 0.000000
  [4] = 0.000000
  [5] = 0.000000
  [6] = 0.000000
  [7] = 0.000000
  [8] = 0.000000
  [9] = 0.000000
num_positions bytes: 6404
aspect_ratio_id: 5

output[0] = 11.099115
output[1] = 15.679549
output[2] = -2.832836
output[3] = 3.733549
output[4] = -0.777896
output[5] = -14.934703
output[6] = 0.122263
output[7] = -2.404936
output[8] = -7.385611
output[9] = 0.015843

"The image depicts the New York City skyline, including several iconic buildings and landmarks. The most prominent building in the image is the Empire State Building, which is located in Midtown Manhattan and stands at a height of 1,454 feet (443 meters). It was completed in 1931 and held the title of the world's tallest building for nearly four decades.\n\nOther notable buildings visible in the image include:\n\n* The Chrysler Building: A 77-story skyscraper located just across the street from the Empire State Building. It was completed in 1930 and is known for its distinctive art deco design.\n* The One World Trade Center: A 104-story skyscraper located at the site of the former World Trade Center, which was destroyed in the terrorist attacks on September 11, 2001. It was completed in 2014 and serves as the main building of the redeveloped World Trade Center complex.\n\nIn addition to these buildings, the image also shows several other notable landmarks, including:\n\n* The Statue of Liberty: A colossal neoclassical sculpture located on Liberty Island in New York Harbor. It was a gift from France to the United States and was dedicated in 1886.\n* The Brooklyn Bridge: A hybrid cable-stayed/suspension bridge that connects the boroughs of Manhattan and Brooklyn over the East River. It was completed in 1883 and is one of the oldest suspension bridges in the world.\n\nOverall, the image provides a stunning view of some of New York City's most iconic landmarks and buildings, showcasing the city's rich history and architectural heritage.
```

llama.cpp (Apple):
```console
Tile 0 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 1 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 2 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 3 first 10 values:
  [0] = 0.485092
  [1] = 0.470494
  [2] = 0.441297
  [3] = 0.426698
  [4] = 0.426698
  [5] = 0.426698
  [6] = 0.426698
  [7] = 0.426698
  [8] = 0.499690
  [9] = 0.514289
n_positions bytes: 6404
aspect_ratio_id = 6
output[0] = 5.889554
output[1] = 20.887268
output[2] = -1.525568
output[3] = 2.030161
output[4] = -2.887129
output[5] = -18.866877
output[6] = -3.165283
output[7] = 1.574872
output[8] = -2.236907
output[9] = 2.396866

The image shows a tree with green leaves and branches.
```
ollama (Apple):
```console
Tile 0 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 1 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 2 first 10 values:
  [0] = 1.930336
  [1] = 1.930336
  [2] = 1.930336
  [3] = 1.930336
  [4] = 1.930336
  [5] = 1.930336
  [6] = 1.930336
  [7] = 1.930336
  [8] = 1.930336
  [9] = 1.930336

Tile 3 first 10 values:
  [0] = 0.485092
  [1] = 0.455895
  [2] = 0.426698
  [3] = 0.426698
  [4] = 0.426698
  [5] = 0.426698
  [6] = 0.426698
  [7] = 0.426698
  [8] = 0.485092
  [9] = 0.514289
num_positions bytes: 6404
aspect_ratio_id: 6

output[0] = 6.947874
output[1] = 17.099964
output[2] = -2.731302
output[3] = 1.561556
output[4] = -0.641592
output[5] = -18.051983
output[6] = -0.249803
output[7] = 1.927986
output[8] = -1.114733
output[9] = 0.568076

There appears to be a single, shiny red apple with a green leaf and brown stem protruding from its top. The apple has yellow spots scattered across it as well.', images=None, tool_calls=None)
```

llama.cpp (Apollo 11):
```console
Tile 0 first 10 values:
  [0] = -1.792263
  [1] = -1.792263
  [2] = -1.792263
  [3] = -1.792263
  [4] = -1.792263
  [5] = -1.792263
  [6] = -1.792263
  [7] = -1.792263
  [8] = -1.777664
  [9] = -1.777664

Tile 1 first 10 values:
  [0] = -1.777664
  [1] = -1.777664
  [2] = -1.777664
  [3] = -1.792263
  [4] = -1.792263
  [5] = -1.792263
  [6] = -1.792263
  [7] = -1.792263
  [8] = -1.792263
  [9] = -1.792263

Tile 2 first 10 values:
  [0] = -1.792263
  [1] = -1.792263
  [2] = -1.617081
  [3] = -1.733869
  [4] = -0.347018
  [5] = 0.470494
  [6] = 0.032541
  [7] = -0.390814
  [8] = -0.405412
  [9] = -0.201034

Tile 3 first 10 values:
  [0] = -0.055050
  [1] = 0.149328
  [2] = 0.514289
  [3] = -0.084247
  [4] = -0.697380
  [5] = -0.930955
  [6] = -1.573286
  [7] = -1.792263
  [8] = -0.887160
  [9] = -1.003947
n_positions bytes: 6404
aspect_ratio_id = 6

ca_patch_emd[0] = 11.866544
ca_patch_emd[1] = 18.326988
ca_patch_emd[2] = -2.006176
ca_patch_emd[3] = 4.276202
ca_patch_emd[4] = -2.578590
ca_patch_emd[5] = -11.743117
ca_patch_emd[6] = -1.049956
ca_patch_emd[7] = 1.673604
ca_patch_emd[8] = -8.623837
ca_patch_emd[9] = -4.596409
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image shows a photograph of a person's feet, with the focus being onggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 their feet and the clothing they are wearing. The person is standing in a room with a light-colored floor and a dark-colored wall. The person is wearing aggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 long-sleeved dress or skirt that is light-colored and
main: decoded 60 tokens in 7.80 s, speed: 7.69 t/s
``` 

ollama.cpp (Apollo 11):
```console
input[0] = -1.792263
input[1] = -1.792263
input[2] = -1.792263
input[3] = -1.792263
input[4] = -1.792263
input[5] = -1.792263
input[6] = -1.792263
input[7] = -1.792263
input[8] = -1.792263
input[9] = -1.792263

Tile 0 first 10 values:
  [0] = -1.792263
  [1] = -1.792263
  [2] = -1.792263
  [3] = -1.792263
  [4] = -1.792263
  [5] = -1.792263
  [6] = -1.792263
  [7] = -1.792263
  [8] = -1.792263
  [9] = -1.792263

Tile 1 first 10 values:
  [0] = -1.792263
  [1] = -1.792263
  [2] = -1.792263
  [3] = -1.792263
  [4] = -1.792263
  [5] = -1.792263
  [6] = -1.792263
  [7] = -1.792263
  [8] = -1.792263
  [9] = -1.792263

Tile 2 first 10 values:
  [0] = -1.792263
  [1] = -1.792263
  [2] = -1.631680
  [3] = -1.733869
  [4] = -0.361617
  [5] = 0.470494
  [6] = 0.032541
  [7] = -0.390814
  [8] = -0.405412
  [9] = -0.201034

Tile 3 first 10 values:
  [0] = -0.055050
  [1] = 0.149328
  [2] = 0.499690
  [3] = -0.084247
  [4] = -0.697380
  [5] = -0.930955
  [6] = -1.587885
  [7] = -1.792263
  [8] = -0.887160
  [9] = -1.003947
num_positions bytes: 6404
aspect_ratio_id: 6

ca_patch_emd[0] = 11.147110
ca_patch_emd[1] = 14.811555
ca_patch_emd[2] = -3.044152
ca_patch_emd[3] = 3.978172
ca_patch_emd[4] = -1.647257
ca_patch_emd[5] = -11.960491
ca_patch_emd[6] = 0.248256
ca_patch_emd[7] = 1.465993
ca_patch_emd[8] = -5.544159
ca_patch_emd[9] = -3.971370

'This is an image of an astronaut on the moon. This photograph was taken during the Apollo 11 mission when Neil Armstrong and Edwin "Buzz" Aldrin became the first humans to walk on the Moon\'s surface.\n\nThe image shows one of the astronauts planting a U.S. flag in the lunar soil, while the other astronaut prepares the equipment for sampling. The background features a vast expanse of gray, rocky terrain with craters and hills stretching out to the horizon.'
```

Now, with the same preprocessed image I should be able to get similar image patch embeddings
for my model but I don't, mine are quite different. And I'm also noticing that setting tensors
as inputs effects my patch embeddings which indicates that those tensors are being reused by
the model and producing different results.

I noticed that reducing the number of tensors that get offloaded to the GPU caused changes to the
output of the model, but the image patch embeddings remained pretty stable.
25 layers offloaded (empire.jpg):
```console
ca_patch_emd[0] = 11.991608
ca_patch_emd[1] = 18.090160
ca_patch_emd[2] = -1.171960
ca_patch_emd[3] = 5.761043
ca_patch_emd[4] = 0.946982
ca_patch_emd[5] = -16.728817
ca_patch_emd[6] = 1.600705
ca_patch_emd[7] = -0.943010
ca_patch_emd[8] = -10.564150
ca_patch_emd[9] = -1.892998
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
This image is of a tree. It appears to be a tree with branches andggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 leaves, rather than an actual tree. The tree is likely a tree of some sort, but its actual type is unclear. It is likely a^C
```

35 layers offloaded (empire.jpg):
```console
ca_patch_emd[0] = 11.993979
ca_patch_emd[1] = 18.090450
ca_patch_emd[2] = -1.176173
ca_patch_emd[3] = 5.761998
ca_patch_emd[4] = 0.948003
ca_patch_emd[5] = -16.730900
ca_patch_emd[6] = 1.601181
ca_patch_emd[7] = -0.943453
ca_patch_emd[8] = -10.561062
ca_patch_emd[9] = -1.893219
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image is of a small, green, tree-lined area that is part ofggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 a larger park or garden. The grassy area is covered with a variety of plants and grasses, including a prominent, light-colored, leafy, greenggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 plant with a long, light-colored, leafy, green
main: decoded 60 tokens in 28.21 s, speed: 2.13 t/s
```

38 layers offloaded (empire.jpg):
```console
ca_patch_emd[0] = 11.991747
ca_patch_emd[1] = 18.090778
ca_patch_emd[2] = -1.178204
ca_patch_emd[3] = 5.768996
ca_patch_emd[4] = 0.949145
ca_patch_emd[5] = -16.735151
ca_patch_emd[6] = 1.604748
ca_patch_emd[7] = -0.945413
ca_patch_emd[8] = -10.561853
ca_patch_emd[9] = -1.890782
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image is of a small, dark-colored tree with a wide, flat,ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 textured canopy. The tree is a dark gray-brown color, with a wide, flat, textured canopy. The tree is surrounded by a light gray, texturedggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
, and textured, with a textured, textured, textured,
main: decoded 60 tokens in 20.01 s, speed: 3.00 t/s

```

41 layers offloaded (empire.jpg):
```console
ca_patch_emd[0] = 11.987567
ca_patch_emd[1] = 18.092655
ca_patch_emd[2] = -1.181066
ca_patch_emd[3] = 5.768115
ca_patch_emd[4] = 0.949257
ca_patch_emd[5] = -16.738150
ca_patch_emd[6] = 1.604087
ca_patch_emd[7] = -0.944634
ca_patch_emd[8] = -10.563751
ca_patch_emd[9] = -1.891053
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image shows a stone fountain in a park, with a statue of a womanggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 holding a bowl, which is likely to be a water fountain. The fountain is made of stone and has a light-colored, textured surface. The statue of theggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 woman is also made of stone and has a light-colored,
main: decoded 60 tokens in 7.60 s, speed: 7.90 t/s
``` 

42 layers offloaded (empire.jpg):
```console
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ca_patch_emd[0] = 11.987567
ca_patch_emd[1] = 18.092655
ca_patch_emd[2] = -1.181066
ca_patch_emd[3] = 5.768115
ca_patch_emd[4] = 0.949257
ca_patch_emd[5] = -16.738150
ca_patch_emd[6] = 1.604087
ca_patch_emd[7] = -0.944634
ca_patch_emd[8] = -10.563751
ca_patch_emd[9] = -1.891053
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image appears to be a photograph of a building or structure, but it isggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 unclear due to the blurriness of the photo. There is no clear structure or details visible in the photo.

main: decoded 39 tokens in 5.02 s, speed: 7.77 t/s
```

This caused me to look closer at the language model graph, inspecting the graph splits and I noticed
that I'd forced the patch embeddings to be on the CPU backend:
```c++
static struct ggml_tensor * llm_build_ca_patch_embd(
        struct ggml_context * ctx,
       struct llama_context & lctx,
        const llama_hparams & hparams,
         const llm_build_cb & cb) {
    const int64_t n_embd = hparams.n_embd;

    // Cross Attention Patch embeddings.
    // 1601 is the number of positions per frame, plus one for the CLS token.
    // 4 is the number of tiles. 1601 * 4 = 6404
    struct ggml_tensor * ca_patch_embd = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1601, 4);
    cb(ca_patch_embd, "ca_patch_embd", -1);
    ggml_set_input(ca_patch_embd);
    lctx.ca_patch_embd = ca_patch_embd;

    //ggml_backend_sched_set_tensor_backend(lctx.sched.get(), ca_patch_embd, lctx.backend_cpu);

   return ca_patch_embd;
}
```
And then I changed the following in `build_mllama`:
```c++
            } else {
                // Use KV-Cache values for K and V.
                //printf("--------- use values from KV cache for kv_cache.layer[%d] ------ \n", il);
                //Kcur = ggml_dup_tensor(ctx0, kv_self.k_l[il]);
		    Kcur = ggml_view_tensor(ctx0, kv_self.k_l[il]);
                    cb(Kcur, "Kcur", il);

                    //Vcur = ggml_dup_tensor(ctx0, kv_self.v_l[il]);
		    Vcur = ggml_view_tensor(ctx0, kv_self.v_l[il]);
                    cb(Vcur, "Vcur", il);
		    //ggml_set_input(gf, Vcur);
                }
```
I also removed all `ggml_set_input/outputs` in `build_mllama` and `build_llama`.

With the above changed I was able to get the following results:

With 42 layers offloaded (empire.jpg):
```console
ca_patch_emd[0] = 11.987567
ca_patch_emd[1] = 18.092655
ca_patch_emd[2] = -1.181066
ca_patch_emd[3] = 5.768115
ca_patch_emd[4] = 0.949257
ca_patch_emd[5] = -16.738150
ca_patch_emd[6] = 1.604087
ca_patch_emd[7] = -0.944634
ca_patch_emd[8] = -10.563751
ca_patch_emd[9] = -1.891053
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image is of the New York City skyline. It includes the Empire State Buildingggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
, which is the tallest building in the image and is the most well-known building in the skyline. It is situated in the center of the skyline and stands outggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 due to its height and unique architecture.
```

apollo.jgp:
```console
ca_patch_emd[0] = 11.866544
ca_patch_emd[1] = 18.326988
ca_patch_emd[2] = -2.006176
ca_patch_emd[3] = 4.276202
ca_patch_emd[4] = -2.578590
ca_patch_emd[5] = -11.743117
ca_patch_emd[6] = -1.049956
ca_patch_emd[7] = 1.673604
ca_patch_emd[8] = -8.623837
ca_patch_emd[9] = -4.596409
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image depicts an astronaut on the moon, with a flag planted in the foregroundggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
. The astronaut is wearing a white spacesuit and is standing on the moon's surface, with the flag planted in the ground nearby. The background of the imageggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 shows the vast, barren landscape of the moon's surface.
main: decoded 60 tokens in 5.90 s, speed: 10.17 t/s
```

apple.jpg:
```console
ca_patch_emd[0] = 5.889554
ca_patch_emd[1] = 20.887268
ca_patch_emd[2] = -1.525568
ca_patch_emd[3] = 2.030161
ca_patch_emd[4] = -2.887129
ca_patch_emd[5] = -18.866877
ca_patch_emd[6] = -3.165283
ca_patch_emd[7] = 1.574872
ca_patch_emd[8] = -2.236907
ca_patch_emd[9] = 2.396866
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image contains a red apple with a green leaf.

main: decoded 11 tokens in 1.10 s, speed: 10.03 t/s
```

black cat:
```console
ca_patch_emd[0] = 10.687916
ca_patch_emd[1] = 26.104683
ca_patch_emd[2] = -5.388736
ca_patch_emd[3] = 1.556743
ca_patch_emd[4] = 1.007764
ca_patch_emd[5] = -19.128231
ca_patch_emd[6] = 4.583585
ca_patch_emd[7] = 2.020725
ca_patch_emd[8] = -2.042757
ca_patch_emd[9] = 0.310236
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
This image is a stock photo featuring a black cat's head and neck, withggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 a dark background. The cat is positioned on the right side of the image, with its head turned to the left, and its body is not visible. Theggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 cat has yellow eyes and a long, thin whisker on
main: decoded 60 tokens in 5.97 s, speed: 10.06 t/s
```

Eiffel Tower:
```console
ca_patch_emd[0] = 9.534625
ca_patch_emd[1] = 15.786053
ca_patch_emd[2] = -3.500089
ca_patch_emd[3] = 5.841105
ca_patch_emd[4] = -1.771145
ca_patch_emd[5] = -13.647954
ca_patch_emd[6] = -0.700998
ca_patch_emd[7] = 0.940367
ca_patch_emd[8] = -8.735542
ca_patch_emd[9] = -0.055729
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
The image shows the Eiffel Tower, a famous landmark in Paris, Franceggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
```

Golden Gate Bridge:
```console
ca_patch_emd[0] = 10.487551
ca_patch_emd[1] = 19.631279
ca_patch_emd[2] = -0.940176
ca_patch_emd[3] = 5.795629
ca_patch_emd[4] = -2.099150
ca_patch_emd[5] = -14.099861
ca_patch_emd[6] = 3.142638
ca_patch_emd[7] = -0.667919
ca_patch_emd[8] = -5.454469
ca_patch_emd[9] = -1.247946
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
This image is of the Golden Gate Bridge in San Francisco, California. The bridgeggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 is a suspension bridge that spans the Golden Gate Strait and connects the city of San Francisco to Marin County. It is a famous landmark and a popular tourist destination,ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 known for its stunning views of the San Francisco Bay and the
main: decoded 60 tokens in 5.82 s, speed: 10.31 t/s
```
