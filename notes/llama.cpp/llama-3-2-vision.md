### Llama 3.2 Vision 
This document contains notes about llama 3.2 vision and about supporting this
model in llama.cpp.

The model architecture is similar to Llama 3.1 but with addition of a vision
model in addition to the text model. The architecture is named `mllama` for
multi-modal llama.

So this model has a variant of a vision transformer (Vit). Now, one thing that
I need to keep in mind is that there is support being added to llama.cpp for
[~vision api support~](https://github.com/ggerganov/llama.cpp/pull/9687). This is
currently based on the Llava example which uses CLIP (which also uses ViT).

> There is now a new PR for [Vision API support](https://github.com/ggerganov/llama.cpp/pull/11292)
> and this document might contain outdated information as the first iteration was
> based on the former Vision API PR.

* [Arxiv Paper](https://arxiv.org/pdf/2407.21783)
* [Vision API PR](https://github.com/ggerganov/llama.cpp/pull/11292)
* [Discussing about multi-model .gguf models](https://github.com/ggerganov/llama.cpp/discussions/11139?sort=old)

### Vocab size issue
One interesting thing with this model is that is has a vocab size specified as:
```console
"vocab_size": 128256 
```
But the special token `<|image|>` is at index 128256, so the actual vocab size
is 128257. We can see this by inspecting the actual vocabulary array in
convert_hf_to_gguf.py:
```python
    tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=is_cli_non_interactive)
    print(f'tokenizer len: {len(tokenizer.vocab)}')
    vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
    assert max(tokenizer.vocab.values()) <= vocab_size
```
```console
tokenizer len: 128257
```
This causes problems as there is a tensor that depend on the vocab size being
128256:
```console
      1:  525336576 |  4096, 128256,    1,     1 | Q6_K    | output.weight
```

The image token needs to be in our models vocab, in `vocab.id_to_token` that is,
so that it is resolved correctly and the correct token id passed to the model.

So as far as I can tell we need to have the additional image token in the
actual vocab list, `id_to_token` in llama.cpp. The vocabulary size is determined
by calling:
```c++
int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab) {
    return vocab->n_tokens();
}

uint32_t llama_vocab::n_tokens() const {
    return (uint32_t) pimpl->id_to_token.size();
}
```
And notice that this is using the size of the `id_to_token` vector
to determine the vocab size. Now, this vector is resized in llama-vocab.cpp:
```c++
    uint32_t n_tokens = gguf_get_arr_n(ctx, token_idx);
    id_to_token.resize(n_tokens);
```
```console
(gdb) p  n_tokens
$1 = 128256
```

I think a way to handle this is to leave the vocab size as 128256 when
converting the model, so that id_to_token will have the correct size. And then
add a special token for the image token.
So adding the following to the converted .gguf model:
```console
     60: UINT32     |        1 | tokenizer.ggml.image_token_id = 128256
```
And then adding this to the vocab special tokens in llama-arch.cpp:
```c++
enum llm_kv {
    ...
    LLM_KV_TOKENIZER_IMAGE_ID,
    ...
```
And the in llama-vocab.cpp:
```c++
struct llama_vocab::impl {
    ...
    llama_token special_image_id = LLAMA_TOKEN_NULL;
    ...
}
```
And the update the handling of special tokens in llama-vocab.cpp:
```c++
void llama_vocab::impl::load(llama_model_loader & ml, const LLM_KV & kv) {
   ...
   // special tokens                                                               
    {                                                                               
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,     special_bos_id     },                    
            { LLM_KV_TOKENIZER_EOS_ID,     special_eos_id     },                    
            { LLM_KV_TOKENIZER_EOT_ID,     special_eot_id     },                    
            { LLM_KV_TOKENIZER_EOM_ID,     special_eom_id     },                    
            { LLM_KV_TOKENIZER_UNK_ID,     special_unk_id     },                    
            { LLM_KV_TOKENIZER_SEP_ID,     special_sep_id     },                    
            { LLM_KV_TOKENIZER_PAD_ID,     special_pad_id     },                    
            { LLM_KV_TOKENIZER_MASK_ID,    special_mask_id    },                    
            { LLM_KV_TOKENIZER_IMAGE_ID,   special_image_id   },                    
            { LLM_KV_TOKENIZER_FIM_PRE_ID, special_fim_pre_id },                    
            { LLM_KV_TOKENIZER_FIM_SUF_ID, special_fim_suf_id },                    
            { LLM_KV_TOKENIZER_FIM_MID_ID, special_fim_mid_id },                    
            { LLM_KV_TOKENIZER_FIM_PAD_ID, special_fim_pad_id },                    
            { LLM_KV_TOKENIZER_FIM_REP_ID, special_fim_rep_id },                    
            { LLM_KV_TOKENIZER_FIM_SEP_ID, special_fim_sep_id },                    
                                                                                    
            // deprecated                                                           
            { LLM_KV_TOKENIZER_PREFIX_ID, special_fim_pre_id },                     
            { LLM_KV_TOKENIZER_SUFFIX_ID, special_fim_suf_id },                     
            { LLM_KV_TOKENIZER_MIDDLE_ID, special_fim_mid_id },                 
        };
```
Hmm, this will still not work as if we print out the tokens for the following
prompt we will see that it will not use the correct image token id:
```console
prompt: <|image|>What is in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

token = 27
token = 91
token = 1843
token = 91
token = 29
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
```
So perhaps we should let the vocabulary size be 128257 so that the image token
is included in `id_to_token` and then modify the shape of `output.weight` that
depends on the size being 128256. 

```python
    tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=is_cli_non_interactive)
    vocab_size = max(self.hparams.get("vocab_size", 0), len(tokenizer.vocab))
    print(f'tokenizer len: {len(tokenizer.vocab)}')
    print(f'vocab_size: {vocab_size}')
    assert max(tokenizer.vocab.values()) <= vocab_size
```
So allowing the vocabulary size to have the actual size of 128257 and then
in `load_tensors` we read the 'vocab_size' from the configuration for this 
model:
```c++
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false);
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab + 8}, 0);
```
With these changes the tokens look like this:
```console
prefix prompt: <|start_header_id|>user<|end_header_id|>


token = 128006
token = 882
token = 128007
token = 271
prompt: <|image|>What is in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


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
```
We can now see that the `<|image|>` token is correctly resolved to the correct
token id 128256.

So that worked when I inspected the tokens which is great. But after processing
the output will be copied:
```c++
            if (n_outputs_new) {
                GGML_ASSERT( n_outputs_prev + n_outputs_new <= n_outputs);
                GGML_ASSERT((n_outputs_prev + n_outputs_new)*n_vocab <= (int64_t) lctx.logits_size);
                ggml_backend_tensor_get_async(backend_res, res, logits_out, 0, n_outputs_new*n_vocab*sizeof(float));
            }
```
```console
(gdb) p res->ne
$4 = {128256, 1, 1, 1}
(gdb) p n_vocab
$7 = 128257
```
In this case the above call will cause an error:
```console
/danbev/work/ai/new-vision-api/ggml/src/ggml-backend.cpp:245:
GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds") failed
```

The `n_vocab` is earlier in `llama_decode_impl`:
```c++
static int llama_decode_impl(
         llama_context & lctx,
           llama_batch   inp_batch) {
           ...

    const int64_t n_vocab = vocab.n_tokens();
```
What I've done just to see if I can get the model working is the following
change:
```c++
    int64_t n_vocab;
    if (model.arch == LLM_ARCH_MLLAMA) {
        n_vocab = 128256;
    } else {
        n_vocab = vocab.n_tokens();
    }
```
This is obviously not a solution but it will allow me to test the model. I'm
going to ask for input about what best way to handle this is.

### New Vision API mllama issue
So I've modified the mllama version that worked with the first new vision api
and I've verified that the pre-processing produces the same output, and I've
also added the same logging to the new version to make sure it is identical.

This is the output from the old version:
```console
token = 27
token = 91
token = 1843
token = 91
token = 29
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
Calculating optimal canvas for image 1280x748 with max_tiles=4, tile_size=560
Possible ratios and their canvas sizes:
  Ratio 1x1 -> Canvas 560x560 (scale_w=0.438 scale_h=0.749 selected=0.438)
  Ratio 1x2 -> Canvas 560x1120 (scale_w=0.438 scale_h=1.497 selected=0.438)
  Ratio 1x3 -> Canvas 560x1680 (scale_w=0.438 scale_h=2.246 selected=0.438)
  Ratio 1x4 -> Canvas 560x2240 (scale_w=0.438 scale_h=2.995 selected=0.438)
  Ratio 2x1 -> Canvas 1120x560 (scale_w=0.875 scale_h=0.749 selected=0.749)
  Ratio 2x2 -> Canvas 1120x1120 (scale_w=0.875 scale_h=1.497 selected=0.875)
  Ratio 3x1 -> Canvas 1680x560 (scale_w=1.312 scale_h=0.749 selected=0.749)
  Ratio 4x1 -> Canvas 2240x560 (scale_w=1.750 scale_h=0.749 selected=0.749)
Selected scale: 0.875000 (upscale=0)
Candidate canvas 1120x1120 (area=1254400)
Final selected canvas 1120x1120
Get image size fit to canvas: img=1280x748, canvas=1120x1120, tile=560
Now resize image to size: 1120x654
Padding image to size 560x560 with aspect ratio 2x2
Padded image to size 1120x1120
Splitting into 2x2 tiles
split_to_tiles: img_width=1120, img_height=1120, tile_width=560, tile_height=560, tiles_x=2, tiles_y=2

Processing tile [0,0], source region: x=0-559, y=0-559
  Tile[0,0] at (0,0): src=(16,147,193) -> dst=(16,147,193)
  Tile[0,0] at (1,0): src=(15,146,192) -> dst=(15,146,192)
  Tile[0,0] at (2,0): src=(12,145,192) -> dst=(12,145,192)
  Tile[0,0] at (0,1): src=(15,148,194) -> dst=(15,148,194)
  Tile[0,0] at (1,1): src=(14,148,193) -> dst=(14,148,193)
  Tile[0,0] at (2,1): src=(10,147,192) -> dst=(10,147,192)
  Tile[0,0] at (0,2): src=(8,145,189) -> dst=(8,145,189)
  Tile[0,0] at (1,2): src=(7,145,190) -> dst=(7,145,190)
  Tile[0,0] at (2,2): src=(5,145,191) -> dst=(5,145,191)

Processing tile [1,0], source region: x=560-1119, y=0-559
  Tile[1,0] at (0,0): src=(195,221,236) -> dst=(195,221,236)
  Tile[1,0] at (1,0): src=(195,221,236) -> dst=(195,221,236)
  Tile[1,0] at (2,0): src=(197,220,236) -> dst=(197,220,236)
  Tile[1,0] at (0,1): src=(192,217,232) -> dst=(192,217,232)
  Tile[1,0] at (1,1): src=(194,218,233) -> dst=(194,218,233)
  Tile[1,0] at (2,1): src=(196,219,235) -> dst=(196,219,235)
  Tile[1,0] at (0,2): src=(192,216,230) -> dst=(192,216,230)
  Tile[1,0] at (1,2): src=(194,217,231) -> dst=(194,217,231)
  Tile[1,0] at (2,2): src=(195,218,232) -> dst=(195,218,232)

Processing tile [0,1], source region: x=0-559, y=560-1119
  Tile[0,1] at (0,0): src=(38,34,35) -> dst=(38,34,35)
  Tile[0,1] at (1,0): src=(25,21,23) -> dst=(25,21,23)
  Tile[0,1] at (2,0): src=(0,0,0) -> dst=(0,0,0)
  Tile[0,1] at (0,1): src=(24,20,21) -> dst=(24,20,21)
  Tile[0,1] at (1,1): src=(18,14,15) -> dst=(18,14,15)
  Tile[0,1] at (2,1): src=(0,0,0) -> dst=(0,0,0)
  Tile[0,1] at (0,2): src=(13,9,10) -> dst=(13,9,10)
  Tile[0,1] at (1,2): src=(11,7,8) -> dst=(11,7,8)
  Tile[0,1] at (2,2): src=(16,11,13) -> dst=(16,11,13)

Processing tile [1,1], source region: x=560-1119, y=560-1119
  Tile[1,1] at (0,0): src=(126,124,129) -> dst=(126,124,129)
  Tile[1,1] at (1,0): src=(216,214,220) -> dst=(216,214,220)
  Tile[1,1] at (2,0): src=(177,176,181) -> dst=(177,176,181)
  Tile[1,1] at (0,1): src=(109,107,112) -> dst=(109,107,112)
  Tile[1,1] at (1,1): src=(223,221,227) -> dst=(223,221,227)
  Tile[1,1] at (2,1): src=(182,181,186) -> dst=(182,181,186)
  Tile[1,1] at (0,2): src=(109,108,113) -> dst=(109,108,113)
  Tile[1,1] at (1,2): src=(225,224,230) -> dst=(225,224,230)
  Tile[1,1] at (2,2): src=(185,184,189) -> dst=(185,184,189)
Processing tile 0
Processing tile 1
Processing tile 2
Processing tile 3
nx=560, ny=2240
aspect_ratio=6
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 0.00 MiB to 2864.12 MiB
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 0.00 MiB to 376.05 MiB

Tile 0 first 10 values:
  [0] = -1.558688
  [1] = -1.573286
  [2] = -1.617081
  [3] = -1.675475
  [4] = -1.719270
  [5] = -1.733869
  [6] = -1.748467
  [7] = -1.763066
  [8] = -1.792263
  [9] = -1.792263

Tile 1 first 10 values:
  [0] = 1.054431
  [1] = 1.054431
  [2] = 1.083627
  [3] = 1.083627
  [4] = 1.083627
  [5] = 1.098226
  [6] = 1.127423
  [7] = 1.142021
  [8] = 1.127423
  [9] = 1.112824

Tile 2 first 10 values:
  [0] = -1.237522
  [1] = -1.427302
  [2] = -1.792263
  [3] = -0.288625
  [4] = -0.098845
  [5] = -1.047743
  [6] = -0.040451
  [7] = -1.164530
  [8] = -1.660877
  [9] = -1.558688

Tile 3 first 10 values:
  [0] = 0.047139
  [1] = 1.360998
  [2] = 0.791659
  [3] = 0.587281
  [4] = 0.879250
  [5] = 0.061738
  [6] = -1.587885
  [7] = -1.704672
  [8] = -1.792263
  [9] = -1.792263
n_positions bytes: 6404, n_positions: 1601
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [embeddings_after_position_embd] [1280 1601 4 1]
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [embeddings_after_tile_position_embd] [1280 1601 4 1]
update_cuda_graph_executable: CUDA graph update failed
update_cuda_graph_executable: CUDA graph update failed
update_cuda_graph_executable: CUDA graph update failed
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to too many consecutive updates
vision encoder output[0] = 9.583341
vision encoder output[1] = 14.313586
vision encoder output[2] = -3.192569
vision encoder output[3] = 5.813879
vision encoder output[4] = 0.386942
vision encoder output[5] = -13.529299
vision encoder output[6] = -2.128806
vision encoder output[7] = 3.152669
vision encoder output[8] = -7.955503
vision encoder output[9] = -4.424203
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 16.01 MiB to 25.66 MiB
n_img_tokens = 1
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 25.66 MiB to 100.21 MiB
ca_patch_emd[0] = 9.583341
ca_patch_emd[1] = 14.313586
ca_patch_emd[2] = -3.192569
ca_patch_emd[3] = 5.813879
ca_patch_emd[4] = 0.386942
ca_patch_emd[5] = -13.529299
ca_patch_emd[6] = -2.128806
ca_patch_emd[7] = 3.152669
ca_patch_emd[8] = -7.955503
ca_patch_emd[9] = -4.424203
The image depicts a cityscape, with a large body of water in the background. The
city appears to be densely populated, with many tall buildings and skyscrapers. In
the background, there is a large body of water, possibly an ocean or a lake. The
sky above is cloudy and h
main: decoded 60 tokens in 19.11 s, speed: 3.14 t/s
```
One thing that is different is the `<|image|>` token is not resovled correctly
with this version but that is something I've fixed in the newest version.

This is the ouput from the newest version:
```console
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
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [l_out-9] [4096 4 1 1]
Decoded prefix prompt
loaded image examples/vision-mllama/ny.jpg, size = 1280 x 748
Calculating optimal canvas for image 1280x748 with max_tiles=4, tile_size=560
Possible ratios and their canvas sizes:
  Ratio 1x1 -> Canvas 560x560 (scale_w=0.438 scale_h=0.749 selected=0.438)
  Ratio 1x2 -> Canvas 560x1120 (scale_w=0.438 scale_h=1.497 selected=0.438)
  Ratio 1x3 -> Canvas 560x1680 (scale_w=0.438 scale_h=2.246 selected=0.438)
  Ratio 1x4 -> Canvas 560x2240 (scale_w=0.438 scale_h=2.995 selected=0.438)
  Ratio 2x1 -> Canvas 1120x560 (scale_w=0.875 scale_h=0.749 selected=0.749)
  Ratio 2x2 -> Canvas 1120x1120 (scale_w=0.875 scale_h=1.497 selected=0.875)
  Ratio 3x1 -> Canvas 1680x560 (scale_w=1.312 scale_h=0.749 selected=0.749)
  Ratio 4x1 -> Canvas 2240x560 (scale_w=1.750 scale_h=0.749 selected=0.749)
Selected scale: 0.875000 (upscale=0)
Candidate canvas 1120x1120 (area=1254400)
Final selected canvas 1120x1120
Get image size fit to canvas: img=1280x748, canvas=1120x1120, tile=560
Now resize image to size: 1120x654
Padding image to size 560x560 with aspect ratio 2x2
Padded image to size 1120x1120
Splitting into 2x2 tiles
split_to_tiles: img_width=1120, img_height=1120, tile_width=560, tile_height=560, tiles_x=2, tiles_y=2

Processing tile [0,0], source region: x=0-559, y=0-559
  Tile[0,0] at (0,0): src=(16,147,193) -> dst=(16.00,147.00,193.00)
  Tile[0,0] at (1,0): src=(15,146,192) -> dst=(15.00,146.00,192.00)
  Tile[0,0] at (2,0): src=(12,145,192) -> dst=(12.00,145.00,192.00)
  Tile[0,0] at (0,1): src=(15,148,194) -> dst=(15.00,148.00,194.00)
  Tile[0,0] at (1,1): src=(14,148,193) -> dst=(14.00,148.00,193.00)
  Tile[0,0] at (2,1): src=(10,147,192) -> dst=(10.00,147.00,192.00)
  Tile[0,0] at (0,2): src=(8,145,189) -> dst=(8.00,145.00,189.00)
  Tile[0,0] at (1,2): src=(7,145,190) -> dst=(7.00,145.00,190.00)
  Tile[0,0] at (2,2): src=(5,145,191) -> dst=(5.00,145.00,191.00)

Processing tile [1,0], source region: x=560-1119, y=0-559
  Tile[1,0] at (0,0): src=(195,221,236) -> dst=(195.00,221.00,236.00)
  Tile[1,0] at (1,0): src=(195,221,236) -> dst=(195.00,221.00,236.00)
  Tile[1,0] at (2,0): src=(197,220,236) -> dst=(197.00,220.00,236.00)
  Tile[1,0] at (0,1): src=(192,217,232) -> dst=(192.00,217.00,232.00)
  Tile[1,0] at (1,1): src=(194,218,233) -> dst=(194.00,218.00,233.00)
  Tile[1,0] at (2,1): src=(196,219,235) -> dst=(196.00,219.00,235.00)
  Tile[1,0] at (0,2): src=(192,216,230) -> dst=(192.00,216.00,230.00)
  Tile[1,0] at (1,2): src=(194,217,231) -> dst=(194.00,217.00,231.00)
  Tile[1,0] at (2,2): src=(195,218,232) -> dst=(195.00,218.00,232.00)

Processing tile [0,1], source region: x=0-559, y=560-1119
  Tile[0,1] at (0,0): src=(38,34,35) -> dst=(38.00,34.00,35.00)
  Tile[0,1] at (1,0): src=(25,21,23) -> dst=(25.00,21.00,23.00)
  Tile[0,1] at (2,0): src=(0,0,0) -> dst=(0.00,0.00,0.00)
  Tile[0,1] at (0,1): src=(24,20,21) -> dst=(24.00,20.00,21.00)
  Tile[0,1] at (1,1): src=(18,14,15) -> dst=(18.00,14.00,15.00)
  Tile[0,1] at (2,1): src=(0,0,0) -> dst=(0.00,0.00,0.00)
  Tile[0,1] at (0,2): src=(13,9,10) -> dst=(13.00,9.00,10.00)
  Tile[0,1] at (1,2): src=(11,7,8) -> dst=(11.00,7.00,8.00)
  Tile[0,1] at (2,2): src=(16,11,13) -> dst=(16.00,11.00,13.00)

Processing tile [1,1], source region: x=560-1119, y=560-1119
  Tile[1,1] at (0,0): src=(126,124,129) -> dst=(126.00,124.00,129.00)
  Tile[1,1] at (1,0): src=(216,214,220) -> dst=(216.00,214.00,220.00)
  Tile[1,1] at (2,0): src=(177,176,181) -> dst=(177.00,176.00,181.00)
  Tile[1,1] at (0,1): src=(109,107,112) -> dst=(109.00,107.00,112.00)
  Tile[1,1] at (1,1): src=(223,221,227) -> dst=(223.00,221.00,227.00)
  Tile[1,1] at (2,1): src=(182,181,186) -> dst=(182.00,181.00,186.00)
  Tile[1,1] at (0,2): src=(109,108,113) -> dst=(109.00,108.00,113.00)
  Tile[1,1] at (1,2): src=(225,224,230) -> dst=(225.00,224.00,230.00)
  Tile[1,1] at (2,2): src=(185,184,189) -> dst=(185.00,184.00,189.00)
Processing tile 0
Processing tile 1
Processing tile 2
Processing tile 3
n_px=40, n_py=40
px=560, py=2240
aspect_ratio=6
vision_image_encode_mllama: image_size = 560
vision_image_encode_mllama: num_positions = 1601
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 669.48 MiB to 2864.12 MiB
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 9.01 MiB to 376.05 MiB

Tile 0 first 10 values:
  [0] = -1.558688
  [1] = -1.573286
  [2] = -1.617081
  [3] = -1.675475
  [4] = -1.719270
  [5] = -1.733869
  [6] = -1.748467
  [7] = -1.763066
  [8] = -1.792263
  [9] = -1.792263

Tile 1 first 10 values:
  [0] = 1.054431
  [1] = 1.054431
  [2] = 1.083627
  [3] = 1.083627
  [4] = 1.083627
  [5] = 1.098226
  [6] = 1.127423
  [7] = 1.142021
  [8] = 1.127423
  [9] = 1.112824

Tile 2 first 10 values:
  [0] = -1.237522
  [1] = -1.427302
  [2] = -1.792263
  [3] = -0.288625
  [4] = -0.098845
  [5] = -1.047743
  [6] = -0.040451
  [7] = -1.164530
  [8] = -1.660877
  [9] = -1.558688

Tile 3 first 10 values:
  [0] = 0.047139
  [1] = 1.360998
  [2] = 0.791659
  [3] = 0.587281
  [4] = 0.879250
  [5] = 0.061738
  [6] = -1.587885
  [7] = -1.704672
  [8] = -1.792263
  [9] = -1.792263
n_positions bytes: 6404, n_positions: 1601
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [embeddings_after_position_embd] [1280 1601 4 1]
check_node_graph_compatibility_and_refresh_copy_ops: disabling CUDA graphs due to batch size > 1 [embeddings_after_tile_position_embd] [1280 1601 4 1]
update_cuda_graph_executable: CUDA graph update failed
update_cuda_graph_executable: CUDA graph update failed
update_cuda_graph_executable: CUDA graph update failed
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to too many consecutive updates
vision encoder output[0] = 9.583341
vision encoder output[1] = 14.313586
vision encoder output[2] = -3.192569
vision encoder output[3] = 5.813879
vision encoder output[4] = 0.386942
vision encoder output[5] = -13.529299
vision encoder output[6] = -2.128806
vision encoder output[7] = 3.152669
vision encoder output[8] = -7.955503
vision encoder output[9] = -4.424203
encoded image
image patch embeddings are in ctx_vision.vctx.output:
name: img_patch_embd
shape: [4096, 1601, 4, 1]
embd_tensor[0] = 9.583341
embd_tensor[1] = 14.313586
embd_tensor[2] = -3.192569
embd_tensor[3] = 5.813879
embd_tensor[4] = 0.386942
embd_tensor[5] = -13.529299
embd_tensor[6] = -2.128806
embd_tensor[7] = 3.152669
embd_tensor[8] = -7.955503
embd_tensor[9] = -4.424203
The image is a picture of a city skyline, specifically the New York City skyline.

main: decoded 17 tokens in 3.72 s, speed: 4.57 t/s
```
Now, the issue is that the output above was mostly "lucky" as other times it 
will generate:
```console
I don't see an image, but I can try to help you if you describe the image or
tell me what it's supposed to be.
```
I've inspected the output of the vision encoder and as far as I can tell they
are identical:
```
Previous version:
vision encoder output[0] = 9.583341
vision encoder output[1] = 14.313586
vision encoder output[2] = -3.192569
vision encoder output[3] = 5.813879
vision encoder output[4] = 0.386942
vision encoder output[5] = -13.529299
vision encoder output[6] = -2.128806
vision encoder output[7] = 3.152669
vision encoder output[8] = -7.955503
vision encoder output[9] = -4.424203
Latest version:
vision encoder output[0] = 9.583341
vision encoder output[1] = 14.313586
vision encoder output[2] = -3.192569
vision encoder output[3] = 5.813879
vision encoder output[4] = 0.386942
vision encoder output[5] = -13.529299
vision encoder output[6] = -2.128806
vision encoder output[7] = 3.152669
vision encoder output[8] = -7.955503
vision encoder output[9] = -4.424203
```
Hmm, but it could also be that it is only the first tile that is identical so
perhaps I should print out the first 10 values of all 4 tiles. Lets start by
printing out the tiles for the vision encoder output.

Vision encoder output for old version:
```console
vision encoder output Tile 0 first 10 values:
  [0] = 9.583341
  [1] = 14.313586
  [2] = -3.192569
  [3] = 5.813879
  [4] = 0.386942
  [5] = -13.529299
  [6] = -2.128806
  [7] = 3.152669
  [8] = -7.955503
  [9] = -4.424203

vision encoder output Tile 1 first 10 values:
  [0] = 5.986829
  [1] = -2.915241
  [2] = -2.784132
  [3] = -4.247492
  [4] = 6.727473
  [5] = 10.927721
  [6] = -6.980994
  [7] = -1.603015
  [8] = 9.635002
  [9] = -24.777727

vision encoder output Tile 2 first 10 values:
  [0] = 11.259818
  [1] = 11.602535
  [2] = -3.990987
  [3] = 10.948430
  [4] = 8.536315
  [5] = -1.765288
  [6] = 10.040323
  [7] = 4.448214
  [8] = 9.211788
  [9] = 8.241113

vision encoder output Tile 3 first 10 values:
  [0] = 0.649771
  [1] = 0.371095
  [2] = -0.332472
  [3] = -2.569907
  [4] = 1.415616
  [5] = -0.114935
  [6] = 0.485733
  [7] = -1.081182
  [8] = 0.368833
  [9] = 0.020522
```

Vision encoder output for new version:
```console
vision encoder output Tile 0 first 10 values:
  [0] = 9.583341
  [1] = 14.313586
  [2] = -3.192569
  [3] = 5.813879
  [4] = 0.386942
  [5] = -13.529299
  [6] = -2.128806
  [7] = 3.152669
  [8] = -7.955503
  [9] = -4.424203

vision encoder output Tile 1 first 10 values:
  [0] = 5.986829
  [1] = -2.915241
  [2] = -2.784132
  [3] = -4.247492
  [4] = 6.727473
  [5] = 10.927721
  [6] = -6.980994
  [7] = -1.603015
  [8] = 9.635002
  [9] = -24.777727

vision encoder output Tile 2 first 10 values:
  [0] = 11.259818
  [1] = 11.602535
  [2] = -3.990987
  [3] = 10.948430
  [4] = 8.536315
  [5] = -1.765288
  [6] = 10.040323
  [7] = 4.448214
  [8] = 9.211788
  [9] = 8.241113

vision encoder output Tile 3 first 10 values:
  [0] = 0.649771
  [1] = 0.371095
  [2] = -0.332472
  [3] = -2.569907
  [4] = 1.415616
  [5] = -0.114935
  [6] = 0.485733
  [7] = -1.081182
  [8] = 0.368833
  [9] = 0.020522
```

Things that are different are how the image patch embeddings are handled in the
newst version. The actual embedding tensor are copied to the context like this:
```c++
    struct ggml_tensor * embeddings = ggml_graph_get_tensor(gf, "mmproj");

    // copy image patch embedding tensor to context
    if (ctx.ctx_ggml) {
        ggml_free(ctx.ctx_ggml);
    }
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ctx.ctx_ggml = ggml_init(params);
    ctx.output = ggml_dup_tensor(ctx.ctx_ggml, embeddings);
    ggml_set_name(ctx.output, "img_patch_embd");
    ggml_backend_alloc_ctx_tensors_from_buft(ctx.ctx_ggml, ctx.model->buft);
    ggml_backend_tensor_copy(embeddings, ctx.output);
    ggml_backend_sched_reset(ctx.sched);
```
In the previous version they the image patch embeddings were copied into a
vector<float> and returned.

_work in progress_

### Model conversion
So we first need to convert the model to GGUF format which is done by the
`convert_hf_to_gguf.py` script. This model consists of not just one model but
it has two which is also reflected in the config.json file of the model. The
language model is in a `text_config` attribute, and the vision model is a
`vision_config` attribute:
```console
{
  "architectures": [
    "MllamaForConditionalGeneration"
  ],
  "image_token_index": 128256,
  "model_type": "mllama",
  "text_config": {
      ...
  }
  "vision_config": {
      ...
  }
```
And we can see the architecture is `MllamaForConditionalGeneration`  and the
model type is `mllama`. 

If we inspect `model.safetensors.index.json` we can find a few tensor that
were not in previous version of Llama:
```console
    "language_model.model.layers.13.cross_attn.k_norm.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.k_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.o_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.q_norm.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.q_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.v_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn_attn_gate": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn_mlp_gate": "model-00002-of-00005.safetensors",
```
These tensors exist for blocks 3, 8, 13, 18, 23, 28, 33, 38. Which also matches
the following attribute in the config.json file:
```console
    "cross_attention_layers": [
      3,
      8,
      13,
      18,
      23,
      28,
      33,
      38
    ],
```
As far as I know there are currently no tensor names like this so we need to
add them to the `gguf-py/gguf/constants.py` file:
```console
    MODEL_TENSOR.CROSS_ATTN_K_NORM          "blk.{bid}.cross_attn_k_norm",
    MODEL_TENSOR.CROSS_ATTN_K_PROJ          "blk.{bid}.cross_attn_k_proj",
    MODEL_TENSOR.CROSS_ATTN_Q_NORM          "blk.{bid}.cross_attn_q_norm",
    MODEL_TENSOR.CROSS_ATTN_Q_PROJ          "blk.{bid}.cross_attn_q_proj",
    MODEL_TENSOR.CROSS_ATTN_O_PROJ          "blk.{bid}.cross_attn_o_proj",
    MODEL_TENSOR.CROSS_ATTN_V_PROJ          "blk.{bid}.cross_attn_v_proj",
    MODEL_TENSOR.CROSS_ATTN_ATTN_GATE       "blk.{bid}.cross_attn_attn_gate",
    MODEL_TENSOR.CROSS_ATTN_MPL_GATE        "blk.{bid}.cross_attn_mpl_gate",
```

### Vision model layer (tensors)
The vision model has 8 global layers and 32 hidden layers:
```console
    "num_global_layers": 8,
    "num_hidden_layers": 32,
```

The model has 8 (`num_global_layers`) global layers:
```console
"vision_model.global_transformer.layers.{bid}.gate_attn"
"vision_model.global_transformer.layers.{bid}.gate_ffn"
"vision_model.global_transformer.layers.{bid}.input_layernorm.bias"
"vision_model.global_transformer.layers.{bid}.input_layernorm.weight"
"vision_model.global_transformer.layers.{bid}.mlp.fc1.bias"
"vision_model.global_transformer.layers.{bid}.mlp.fc1.weight"
"vision_model.global_transformer.layers.{bid}.mlp.fc2.bias"
"vision_model.global_transformer.layers.{bid}.mlp.fc2.weight"
"vision_model.global_transformer.layers.{bid}.post_attention_layernorm.bias"
"vision_model.global_transformer.layers.{bid}.post_attention_layernorm.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.k_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.o_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.q_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.v_proj.weight"

fc = fully connected.
```

And 32 (`num_hidden_layers`) hidden layers:
```console
"vision_model.transformer.layers.{bid}.input_layernorm.bias"
"vision_model.transformer.layers.{bid}.input_layernorm.weight"
"vision_model.transformer.layers.{bid}.mlp.fc1.bias"
"vision_model.transformer.layers.{bid}.mlp.fc1.weight"
"vision_model.transformer.layers.{bid}.mlp.fc2.bias"
"vision_model.transformer.layers.{bid}.mlp.fc2.weight"
"vision_model.transformer.layers.{bid}.post_attention_layernorm.bias"
"vision_model.transformer.layers.{bid}.post_attention_layernorm.weight"
"vision_model.transformer.layers.{bid}.self_attn.k_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.o_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.q_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.v_proj.weight"
}
```

I initially thougth that having a single model for both the language and vision
model was a good idea, simpler to manage for users. But I had not considered
that it might not optimal from a performance perspective. If we have separate
models.  
The following is from a [discussion](https://github.com/ggerganov/llama.cpp/discussions/11139#discussioncomment-11783418)
on this topic:
```
Having separate models allows to create separate contexts for the encoder and
decoder which gives more fine-grained control over the computation - how many
layers to offload, which devices to use, how much memory to reserve, etc.
Also, computations of the encoder and the decoder could be interleaved which is
important for high-performance scenarios - for example, while we are decoding
the response for an image we could be already encoding the next images.

Having a single GGUF for the entire vision model is definitely more convenient
for users and distribution. But maybe this can be achieved by extending GGUF to
allow packing multiple GGUFs (like an archive).
```
So I'm going to create two models for Llama 3.2 Vision Instruct and then take
a look at how packaging multiple GGUFs could be done. Actually, using one .gguf
for both will work so we will be converting into a single model.



### Language model layers (tensors)
```console
"language_model.lm_head.weight"
"language_model.model.embed_tokens.weight"
"language_model.model.norm.weight"
```

The language model has 40 hidden layers (`num_hidden_layers`):
```console
"language_model.model.layers.{bid}.input_layernorm.weight"
"language_model.model.layers.{bid}.mlp.down_proj.weight"
"language_model.model.layers.{bid}.mlp.gate_proj.weight"
"language_model.model.layers.{bid}.mlp.up_proj.weight"
"language_model.model.layers.{bid}.post_attention_layernorm.weight"
"language_model.model.layers.{bid}.self_attn.k_proj.weight"
"language_model.model.layers.{bid}.self_attn.o_proj.weight"
"language_model.model.layers.{bid}.self_attn.q_proj.weight"
"language_model.model.layers.{bid}.self_attn.v_proj.weight"
```


#### Model notes
So we first need to convert the model to GGUF format which is done by the
`convert_hf_to_gguf.py` script.

This model consists of not just one model but it has two which is also reflected
in the `config.json` file of the model. The language model is in a `text_config`
attribute, and the vision model is a `vision_config` attribute.

### Vision API in llama.cpp
The current Vision API in llama.cpp includes an example where a llava model is
used. This model contains both the language model and the vision model (like
Llava 1.6 does). So the arch type of this model is `llama`:
```console
(venv) $ ./inspect-model.sh ~/work/ai/llama.cpp/models/llava-1.5.7b-hf.gguf
INFO:gguf-dump:* Loading: /home/danbev/work/ai/llama.cpp/models/llava-1.5.7b-hf.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 49 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 684
      3: UINT64     |        1 | GGUF.kv_count = 46
      4: STRING     |        1 | general.architecture = 'llama'
      ...
     23: STRING     |        1 | vision.type = 'clip-vit'
    ...
```
Now, this `vision_type` will be checked in `llm_load_hparams` and this will
set `model.has_vision` to true. This flag will later be used in
`llm_load_tensors` to load the vision tensors. And since the arch of the
model is `llama` this means that the switch-case for `LLM_ARCH_LLAMA` in 
`llm_load_hparams` will also be executed, and similarly for `llm_load_tensors`.
the tensors for the `llama` architecture will also be loaded.

### Prompt-based models
This type of model converts an image into an embedding that is similar to a
textual embeddings. These embeddings are then prepended or appended to a text
token embeddings and input (prompt therefor the name prompt-based) to the LLM.
This does not require any change to the LLM model. 
So in the case of LLava we had in the previous section it can use the standard
`llama` model architecture.

### Cross-attention based models
This type also has an encoder that converts an image into an embedding but
instead of passing these embeddings with the text token embeddings the model
in modified to include them in cross-attention layers.


### Vision model
So I think this is similar to the llava example in llama.cpp where we have the
LLM model and the image encoder model. For example, with llava1.6 that model
contains both the vision encoder and the LLM model, and we have to extract
the vision encoder from the model to use it with the llava example. For example,
when we run `llama-llava-cli` we specify both the model and the projection 
models::
```console
./build/bin/llama-llava-cli -m models/vicuna-7b-q5_k.gguf \
    --mmproj models/mmproj-vicuna7b-f16.gguf \
    --image ~/work/ai/learning-ai/notes/apollo11.jpg -c 4096 -ngl 15
```
But I believe that for the new Vision API in llama.cpp is will be possible to
just pass a single model to llama.cpp and not have to have two separate models.

Is we inspect the tensors that are in `model.safetensors.index.json` we can see
it has both the text language model tensors, and the vision model tensors.

So I think that the llama3.2-vision model will work in a similar way. First one
or more images would be read and split into patches which would then be
passed to the projector model (the vision model). This model will pass the
patches through the vision model, going through the self-attention for the
patches (all the layers in the model). The output of this will be
patch embeddings that can then be passed into the language model.

### Vision model layer (tensors)
The vision model has 8 global layers and 32 hidden layers:
```console
    "num_global_layers": 8,
    "num_hidden_layers": 32,
```

```console
"vision_model.patch_embedding.weight"
"vision_model.class_embedding"

"vision_model.pre_tile_positional_embedding.embedding.weight" 
"vision_model.pre_tile_positional_embedding.gate"

"vision_model.layernorm_pre.weight"
"vision_model.layernorm_pre.bias"

"vision_model.gated_positional_embedding.embedding"
"vision_model.gated_positional_embedding.gate"

"vision_model.gated_positional_embedding.tile_embedding.weight"
"vision_model.post_tile_positional_embedding.embedding.weight"
"vision_model.post_tile_positional_embedding.gate"

"vision_model.layernorm_post.bias"
"vision_model.layernorm_post.weight"
```

Tiling is done to support higher resolution images. In a standard vision
tranformer an image would be split into patches, perhaps 16x16 pixels and then
processed as a sequence. When tiling is used the image is first split into
smaller tiles, perhaps 224x224 each, and then each tile is processed in the same
way as a standared vision transformer (each tile is split into patches and then
processed as a sequence).

The model has 8 (`num_global_layers`) global layers:
```console
"vision_model.global_transformer.layers.{bid}.input_layernorm.weight"
"vision_model.global_transformer.layers.{bid}.input_layernorm.bias"

"vision_model.global_transformer.layers.{bid}.self_attn.q_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.k_proj.weight"
"vision_model.global_transformer.layers.{bid}.self_attn.v_proj.weight"

"vision_model.global_transformer.layers.{bid}.self_attn.o_proj.weight"

"vision_model.global_transformer.layers.{bid}.post_attention_layernorm.weight"
"vision_model.global_transformer.layers.{bid}.post_attention_layernorm.bias"

"vision_model.global_transformer.layers.{bid}.mlp.fc1.weight"
"vision_model.global_transformer.layers.{bid}.mlp.fc1.bias"

"vision_model.global_transformer.layers.{bid}.gate_attn"
"vision_model.global_transformer.layers.{bid}.gate_ffn"

"vision_model.global_transformer.layers.{bid}.mlp.fc2.bias"
"vision_model.global_transformer.layers.{bid}.mlp.fc2.weight"

fc = fully connected.
```

And 32 (`num_hidden_layers`) hidden layers:
```console
"vision_model.transformer.layers.{bid}.input_layernorm.bias"
"vision_model.transformer.layers.{bid}.input_layernorm.weight"

"vision_model.transformer.layers.{bid}.self_attn.k_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.q_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.v_proj.weight"
"vision_model.transformer.layers.{bid}.self_attn.o_proj.weight"

"vision_model.transformer.layers.{bid}.post_attention_layernorm.bias"
"vision_model.transformer.layers.{bid}.post_attention_layernorm.weight"

"vision_model.transformer.layers.{bid}.mlp.fc1.bias"
"vision_model.transformer.layers.{bid}.mlp.fc1.weight"

"vision_model.transformer.layers.{bid}.mlp.fc2.bias"
"vision_model.transformer.layers.{bid}.mlp.fc2.weight"
}
```

So I think that when we convert a llama3.2-vision model we should produce two
.gguf files, one for the vision model and one for the language model.

### Language model layers (tensors)
```console
"language_model.lm_head.weight"
"language_model.model.embed_tokens.weight"
"language_model.model.norm.weight"
```

The language model has 40 hidden layers (`num_hidden_layers`):
```console
"language_model.model.layers.{bid}.input_layernorm.weight"
"language_model.model.layers.{bid}.mlp.down_proj.weight"
"language_model.model.layers.{bid}.mlp.gate_proj.weight"
"language_model.model.layers.{bid}.mlp.up_proj.weight"
"language_model.model.layers.{bid}.post_attention_layernorm.weight"
"language_model.model.layers.{bid}.self_attn.k_proj.weight"
"language_model.model.layers.{bid}.self_attn.o_proj.weight"
"language_model.model.layers.{bid}.self_attn.q_proj.weight"
"language_model.model.layers.{bid}.self_attn.v_proj.weight"
```
All blocks have the above tensors, but there are also blocks that have
additional tensors. For example block 13 has:
```console
    "language_model.model.layers.13.cross_attn.k_norm.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.k_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.o_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.q_norm.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.q_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn.v_proj.weight": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn_attn_gate": "model-00002-of-00005.safetensors",
    "language_model.model.layers.13.cross_attn_mlp_gate": "model-00002-of-00005.safetensors",
```
These tensors exist for blocks 3, 8, 13, 18, 23, 28, 33, 38 as well. Which also
matches the following attribute in the config.json file:
```console
    "cross_attention_layers": [
      3,
      8,
      13,
      18,
      23,
      28,
      33,
      38
    ],
```

### Hyperparameters

#### supported_aspect_ratios
This is defined for the vision model and looks like this:
```console
  "vision_config": {
    ...
    "supported_aspect_ratios": [
      [ 1, 1 ],
      [ 1, 2 ],
      [ 1, 3 ],
      [ 1, 4 ],
      [ 2, 1 ],
      [ 2, 2 ],
      [ 3, 1 ],
      [ 4, 1 ]
    ],
```
These are ratios that the model is designed to handle efficiently. Each sublist
specified a width to height ration:
* 1, 1  - square
* 1, 2  - portrait
* 2, 1  - landscape

Though the name is very different in CLIP the hyperparameter called
`image_grid_pinpoints` seems to serve the same purpose if we look at the code
in llama.cpp's clip implementation ([notes](https://github.com/danbev/learning-ai/blob/main/notes/vision/llava.md)).
This is stored like this for CLIP models:
```console
     20: [INT32]    |       10 | clip.vision.image_grid_pinpoints
```
So this is just a list of integers and they are in pairs which is how llama.cpp
/clip.cpp handles them. So I think we can store `supported_aspect_ratios` the
same way.

#### image_token_index
This is a property in config.json and looks like this:
```console
{
  "architectures": [
    "MllamaForConditionalGeneration"
  ],
  "image_token_index": 128256,
  "model_type": "mllama",
```
This is special token used to identify image tokens when we combine text and
image. For example:
```
text = "This image <|image|> shows a cat"
```
And this would get tokenized into:
```
["This", "image", 128256, "shows", "a", "cat"]
```
Where the other words would also be integer tokens which are indices into
the vocabulary.

This is what this token looks like in the `tokenizer_config.json`:
```
    "128256": {
      "content": "<|image|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    }
```

Now, the vocabulary is stored in the `text_config` attribute:
```
"vocab_size": 128256
```
Notice that this is the same value as the `image_token_index` value.
This value should be part of the model but I missed this originally. 

The issue is that when we pass in the vocabulary size to create the tensor
for `language_model.model.embed_tokens.weight` this does not match the value
of the actual tensor in the model:
```c++
model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
```
In our case we know that the `vocab_size` is 128256, so this is what will be
passed.

If we inspect the model we can see that the actual value of the tensor is infact
128264:
```console
  2:  525369344 |  4096, 128264,     1,     1 | Q6_K    | token_embd.weight
```
So this is something that happens and I think the cause is that the actual
tensor has this shape:
```console
(venv) $ python ~/work/ai/learning-ai/fundamentals/python/src/list-safetensors.py model-00001-of-00005.safetensors

Tensor shapes in the file:
--------------------------------------------------
language_model.model.embed_tokens.weight: torch.Size([128264, 4096])
...
```
Notice that the shape of `language_model.model.embed_tokens.weight` is
`torch.Size([128264, 4096])` which is 8 more than the `vocab_size` value. So
we need to make sure that this value is correct when loading the tensor in
llama.cpp `llm_load_tensor`. We don't have any control over the values of the
tensor in the model, but we can acount for this when loading this tensor.

### max position embedding
The maxium position embedding is calculated using the image size and the patch
size as follows:
```
max_pos_embd = (image_size // patch_size)^2 + 1

The +1 is for the CLS token.
And we have both width and height so we have to square.
```
In the python conversion script this is done using:
```console
        max_pos_embd = (self.vision_config["image_size"] // self.vision_config["patch_size"])**2 + 1
        self.gguf_writer.add_vision_clip_max_position_embeddings(max_pos_embd)
```
The actual values are the following:
```console
image_size 560
patch_size 14
max_pos_embd 1601
```


### Image preprocessing
The image encoder is a pre-trained ViT-H/14 model which produces image patch
embeddings. These embeddings are then used in the cross-attention for the
language model. This is a little different from other models where the image
patch embeddings are projected into the same embedding space as the text
embeddings and then are used as normal tokens to the transformer model.

### Pre-processor config
Some models have this information in `preprocessor_config.json` which is needed
for the pre-processing of images, for example Llama 3.2 Vision Instruct has the
following:
```console
{
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_pad": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "MllamaImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "max_image_tiles": 4,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 560,
    "width": 560
  }
}
```
Using this information we can see that rescaling is used:
```
"do_rescale": true,
"rescale_factor": 0.00392156862745098,
```
This is 1/255 which we can verify using:
```console
1/255 = 0.00392156862745098
```
This will recale to a range of [0,1] from [0,255].

This means the preprocessing pipeline does:
1. Convert to RGB if needed (`do_convert_rgb`: true)
2. Rescale pixels using `rescale_factor`(`do_rescale`: true)
```c++
    pixel = pixel * 0.00392156862745098  // Same as pixel / 255.0f
    or 
    pixel = pixel / 255.0f
```
3. Normalize using mean/std (`do_normalize`: true)
```c++
    normalized = (pixel - mean) / std
```
4. Resize/pad to specified size (`do_resize": true, `do_pad`: true)

### Image loading
Image loading in llama.cpp uses stb_image.h:
```c++
    int inW, inH, inC;
    unsigned char * input = stbi_load(fname, &inW, &inH, &inC, 3);
```
The last parameter is the number of channels, which is always 3 for RGB images.
And the width, height, and the channels are stored in inW, inH, and inC respectively. So
inC would be 1 for a grayscale image, and 3 for an RGB image, and 4 for an RGBA image.


### Tiling
Tiling is something that is used for large images so that they don't get scaled down
into to an image that is too small, which might distort the image information and cause
the model to not be able to process it properly, or with enough accuracy. Wide images
can become squished, and tall images can become stretched and become unrecognizable. Text
migth become unreadable, and objects might become unrecognizable.

So what is done is the larger image is split into multiple images of the size that the model
expects. For example, if the model expects (was trained on) 560x560 images, images that
are larger can be split into multiple 560x560 images. This is where the concept of
tiling comes in. This allows us to keep the original proportions of the image and keep
a natural view of the image.

As an example, in Llama 3.2 Vision Instruct, the model expects 560x560 images and has
the following aspect ration configuration:
```
"supported_aspect_ratios": [
      [ 1, 1 ],
      [ 1, 2 ],
      [ 1, 3 ],
      [ 1, 4 ],
      [ 2, 1 ],
      [ 2, 2 ],
      [ 3, 1 ],
      [ 4, 1 ]
    ],
```
So it has 8 different aspect ratio configurations that it can handle. The first number
is the width, and the second number is the height. So for example, the first configuration
is 1x1, which means that the image is square. The second configuration is 1x2, which means
that the image is twice as tall as it is wide. 

Aspect Ratio Configurations (each box represents a 560x560 tile)
```

1. 1x1 (Index 1)     2. 1x2 (Index 2)     3. 1x3 (Index 3)     4. 1x4 (Index 4)
+-------+            +-------+            +-------+            +-------+
|       |            |   1   |(560x560)   |   1   |(560x560)   |   1   |(560x560)
|   1   |            +-------+            +-------+            +-------+
|       |            |   2   |(560x560)   |   2   |(560x560)   |   2   |(560x560)
+-------+            +-------+            +-------+            +-------+
(560x560)                                 |   3   |(560x560)   |   3   |(560x560)
                                          +-------+            +-------+
                                                               |   4   |(560x560)
                                                               +-------+

5. 2x1 (Index 5)     6. 2x2 (Index 6)     7. 3x1 (Index 7)
+-------+-------+    +-------+-------+    +-------+-------+-------+
|   1   |   2   |    |   1   |   2   |    |   1   |   2   |   3   |
+-------+-------+    +-------+-------+    +-------+-------+-------+
                     |   3   |   4   |
                     +-------+-------+

8. 4x1 (Index 8)
+-------+-------+-------+-------+
|   1   |   2   |   3   |   4   |
+-------+-------+-------+-------+

```
Each tile represents a 560x560 pixel area and in the code the `tile_size` represents the dimension
so 560 in this case. Numbers indicate processing order (left-to-right, top-to-bottom).

```c++
static std::pair<int,int> get_optimal_canvas(int w, int h, int n_tiles, int tile_size) {
    printf("get_optimal_canvas: w=%d, h=%d, n_tiles=%d, tile_size=%d\n", w, h, n_tiles, tile_size);
    // This is the size that the model expects.
    int model_dim = tile_size;

    // Calculate the width to height ratio.
    //        10
    // +-----------------+
    // |                 |
    // |                 | 5         10 / 5 = 2
    // |                 |
    // +-----------------+
    //     5
    // +--------+
    // |        |
    // |        |          10         5 / 10 = 0.5
    // |        |
    // |        |
    // |        |
    // |        |
    // +--------+
    float aspect = static_cast<float>(w) / h;

    // If the width or height is less than the min dimension
    if (w < model_dim || h < model_dim) {
        if (aspect > 1.0f) {
	        //      Wide image
	        //  +-----------------+          Example: 300x200 image
	        //  |                 |          aspect = 300 / 200 = 1.5
	        //  |                 |          w = 560 (max of 300 and 560)
	        //  |                 |          h = 373 (560 / 1.5)
	        //  +-----------------+
	        // Set width to model_dim or width.
            w = std::max(w, model_dim);
	        // Calculate a new height based on the aspect ratio.
            h = static_cast<int>(w / aspect);
        } else {
            //	    Tall image
            //	    +-----+
            //	    |     |
            //	    |     |
            //	    |     |
            //	    |     |
            //	    +-----+
            //
            // Set height to model_dim or height.
            h = std::max(h, model_dim);
            // Calculate a new width based on the aspect ratio.
            w = static_cast<int>(h * aspect);
        }
    }

    int max_canvas_w = std::min(w, n_tiles * tile_size);
    int max_canvas_h = std::min(h, n_tiles * tile_size);

    return {max_canvas_w, max_canvas_h};
}
```
So lets say we have 4 tiles and the model expects images to be 560x560. So the maximum
canvas would be 4 * 560 = 2240 pixels.

A canvas is like a drawing canvas where the tiles will be placed. The size of the canvas
is determined by how many tiles we need. 
For example, I have an image that is 1280x853 and 3 channels, the canvas for this would be
1280x853 (since it is smaller than 2240).
```
                  1280
     +------------------------------+
     |                              |
     |                              |
     |                              |  853
     |                              |
     |                              |
     |                              |
     +------------------------------+
```

```c++
static std::pair<int,int> scale_to_fit_canvas(int w, int h, int canvas_w, int canvas_h, int tile_size) {
    double scale = std::min(double(canvas_w) / double(w), double(canvas_h) / double(h));

    if (scale > 1.0) {
	// Don't upscale the image if it is smaller than the canvas.
        scale = 1.0;
    }

    // Apply scaling to get new width and height.
    int new_w = int(std::floor(w * scale));
    int new_h = int(std::floor(h * scale));

    // Round down to multiples of tile_size
    new_w = (new_w / tile_size) * tile_size;
    new_h = (new_h / tile_size) * tile_size;

    // Ensure that the new dimensions are at least tile_size(model dimension size)
    if (new_w < tile_size) new_w = tile_size;
    if (new_h < tile_size) new_h = tile_size;

    return {new_w, new_h};
}
```
So for the example of the 1280x853 image, the scaling factor will be 1. Notice that we will
round this down:
```
width:
  1280 / 560 = 2
  2 * 560    = 1120
height:
  853 / 560 = 1
  1 * 560   = 560

                  1120
     +------------------------------+
     |                              |
     |                              |
     |                              |  560
     |                              |
     |                              |
     |                              |
     +------------------------------+

Scaled size: 1120 x 560
```

```c++
    // Resize to Width/Height/Channel
    std::vector<unsigned char> resized_hwc(final_w * final_h * 3);
    stbir_resize_uint8_linear(
        input, i_w, i_h, i_w * 3,
        resized_hwc.data(),
        final_w, final_h, final_w * 3,
        STBIR_RGB
    );
```
This is creating a vector of size 1280x560x3 = 1881600. Notice that we then call `stbir_resize_uint8_linear`
and we specify the input image, the input width, height, and channels. Then we specify where output
data should be stored and the width and height, and channels for the output image. So this is where
the scaling of the image happens.

Then we are going to split this resized image into tiles;
```c++
static std::vector<std::vector<unsigned char>> subdivide_into_tiles(
    const unsigned char* resized_hwc,
    int final_w,
    int final_h,
    int tile_size
) {
    // Number horizontal tiles (x-axis)
    int tiles_x = final_w / tile_size;
    // Number of vertical tiles (y-axis)
    int tiles_y = final_h / tile_size;

    std::vector<std::vector<unsigned char>> tiles;
    tiles.reserve(tiles_x * tiles_y);

    // iterate over the y axis (rows)
    for (int ty = 0; ty < tiles_y; ty++) {
        // iterate over the x axis (columns)
        for (int tx = 0; tx < tiles_x; tx++) {
            // Create a vector to store the one tile
            std::vector<unsigned char> tile_data(tile_size * tile_size * 3);

            for (int tile_row = 0; tile_row < tile_size; tile_row++) {
                for (int tile_col = 0; tile_col < tile_size; tile_col++) {
                    int src_x = tx * tile_size + tile_col;
                    int src_y = ty * tile_size + tile_row;

                    int src_idx = (src_y * final_w + src_x) * 3;
                    int dst_idx = (tile_row * tile_size + tile_col) * 3;

                    // copy 3 channels
                    tile_data[dst_idx + 0] = resized_hwc[src_idx + 0]; // Red
                    tile_data[dst_idx + 1] = resized_hwc[src_idx + 1]; // Green
                    tile_data[dst_idx + 2] = resized_hwc[src_idx + 2]; // Blue
                }
            }
            tiles.push_back(std::move(tile_data));
        }
    }

    return tiles;
}
```
So after this we will have a vector of vectors of unsigned chars, where each vector represents
a tile of the image. And the size of each tile is 560x560x3 = 940800 bytes. And the values will
be in `[R G B] [R G B]...`.


### Tensors
So as we discussed earlier the model has two parts, the language model and the
vision model. 

#### Language model tensors
The tensors that are for the LLM are named with a prefix `language_model` prefix,
for example 
```console
language_model.lm_head.weight
language_model.model.embed_tokens.weight
language_model.model.norm.weight
```

And we have 32 normal layers which each have 9 tensors:
```console
language_model.model.layers.0.input_layernorm.weight
language_model.model.layers.0.mlp.down_proj.weight
language_model.model.layers.0.mlp.gate_proj.weight
language_model.model.layers.0.mlp.up_proj.weight
language_model.model.layers.0.post_attention_layernorm.weight
language_model.model.layers.0.self_attn.k_proj.weight
language_model.model.layers.0.self_attn.o_proj.weight
language_model.model.layers.0.self_attn.q_proj.weight
language_model.model.layers.0.self_attn.v_proj.weight
```

And 8 layers that are involved in the cross-attention and each have 13 tensors:
```console
language_model.model.layers.3.cross_attn.k_norm.weight
language_model.model.layers.3.cross_attn.k_proj.weight
language_model.model.layers.3.cross_attn.o_proj.weight
language_model.model.layers.3.cross_attn.q_norm.weight    
language_model.model.layers.3.cross_attn.q_proj.weight
language_model.model.layers.3.cross_attn.v_proj.weight
language_model.model.layers.3.cross_attn_attn_gate
language_model.model.layers.3.cross_attn_mlp_gate
language_model.model.layers.3.input_layernorm.weight
language_model.model.layers.3.mlp.down_proj.weight
language_model.model.layers.3.mlp.gate_proj.weight
language_model.model.layers.3.mlp.up_proj.weight
language_model.model.layers.3.post_attention_layernorm.weight
```

```
multi_modal_projector.bias
multi_modal_projector.weight
````

#### Vision model tensors
The tensors that are for the LLM are named with a prefix `vision_model` prefix,

The following are the tensors that are not part of the layers:
```console
vision_model.class_embedding
vision_model.patch_embedding.weight
vision_model.gated_positional_embedding.embedding
vision_model.gated_positional_embedding.gate
vision_model.gated_positional_embedding.tile_embedding.weight
vision_model.layernorm_post.bias
vision_model.layernorm_post.weight
vision_model.layernorm_pre.bias
vision_model.layernorm_pre.weight
vision_model.post_tile_positional_embedding.embedding.weight
vision_model.post_tile_positional_embedding.gate
vision_model.pre_tile_positional_embedding.embedding.weight
vision_model.pre_tile_positional_embedding.gate
vision_model.transformer.layers.0.input_layernorm.bias
```

##### Graph
```c++
inp_raw =
struct ggml_tensor *inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32,
                                                 image_size_width,
                                                 image_size_height,
                                                 num_channels,
                                                 num_tiles);
```
So we have the raw image (w*h) with 3 channels each and 4 tiles. A tile is a
special segment in the image.
```console
channels  = 3
num_tiles = 4
   
t_0
    c_0
         [0        ...      width]
         ...
         ...
         ...
         [height   ...      width]

    ...

    c_2
         [0        ...      width]
         ...
         ...
         ...
         [height   ...      width]

...

t_3
    c_0
         [0        ...      width]
         ...
         ...
         ...
         [height   ...      width]

    ...

    c_2
         [0        ...      width]
         ...
         ...
         ...
         [height   ...      width]
```
So this is bascially representing the image 4 times.

Next a 2d convolution is applied over the image which is using the
`patch_embedding` tensor as the kernel:
```c++
struct ggml_tensor *inp = ggml_conv_2d(ctx0, model.patch_embeddings, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
```

```c++
 struct ggml_tensor *aspect_ratios = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, imgs->size);
    ggml_set_name(aspect_ratios, "aspect_ratios");
    ggml_set_input(aspect_ratios);

    if (model.pre_tile_position_embeddings != nullptr) {
        struct ggml_tensor *pre_tile_position_embeddings = ggml_get_rows(ctx0, model.pre_tile_position_embeddings, aspect_ratios);
        ggml_set_name(pre_tile_position_embeddings, "pre_tile_position_embeddings");

        pre_tile_position_embeddings = ggml_reshape_3d(ctx0, pre_tile_position_embeddings, hidden_size, 1, num_tiles);
        if (model.pre_tile_position_embeddings_gate != nullptr) {
            pre_tile_position_embeddings = ggml_mul_inplace(ctx0, pre_tile_position_embeddings, model.pre_tile_position_embeddings_gate);
        }

        inp = ggml_add(ctx0, inp, pre_tile_position_embeddings);
    }
```

### Ollama 3.2 Vision model
I've not been able to get the model to work locally and the only output it
results in when passing an image is a bunch of question marks. I've downloaded
the model that ollama uses (which is publicly available which was a little
surprising) and I've been able to inspect it. In ollama they have split the
model into a the llm part and the projector (vision) part much like what is
done for the llava 1.5 models in llama.cpp.
The models are availble in the blobs directory so we can inspect it.

The projector contains 512 tensors:
```console
 6       2: UINT64     |        1 | GGUF.tensor_count = 512
```
And the llm model contains 396 tensors:
```console
  5       2: UINT64     |        1 | GGUF.tensor_count = 396
```
That is a total of 908 tensors.

In the model I converted I only have 907:
```console
       2: UINT64     |        1 | GGUF.tensor_count = 907
```
So this was good to find out and I need to investigate which tensor is missing.

One thing I can also do is use the language model from ollama and run it without
the updates made for vision and see if that works (so just a pure chat and
not image data).

I found that the missing tensor is:
```console
      5:          1 |     1,     1,     1,     1 | F32     | v.tile_position_embd.gate
```

### Image related hyperparameters
These are the parameters related to images:
```console
  "vision_config": {
    "attention_heads": 16,
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "image_size": 560,
    "intermediate_size": 5120,
    "max_length": 20,
    "max_num_tiles": 4,
    "min_length": 0,
    "model_type": "mllama_vision_model",
    "num_channels": 3,
    "num_global_layers": 8,
    "num_hidden_layers": 32,
    "patch_size": 14,
    "supported_aspect_ratios": [
      [ 1, 1 ], [ 1, 2 ], [ 1, 3 ], [ 1, 4 ], [ 2, 1 ], [ 2, 2 ], [ 3, 1 ], [ 4, 1 ]
    ],
    "vision_output_dim": 7680
  }
}
```
And we also have the following pre-processor config:
```console
{
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_pad": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "MllamaImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "max_image_tiles": 4,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 560,
    "width": 560
  }
}
```
So we can see that the width and height of the image is 560 x 560.
```
image_size / patch_size = 560 / 14 = 40
And we have both width and height so we have to square.
40 x 40 = 1600

And optionally we have a CLS token:
1600 + 1 = 1601

If we have 4 tiles then we get:
1601 x 4 = 6404

We can write this as:
(560 / 14)^2 + 1 x 4 = 6404
```


### Prompting
So the prompt for the vision model has a special token for images which is
`<|image|>` and this has token id 128256. But when I run the example I have
and tokenize the following prompt:
```console
(gdb) p params.prompt
$8 = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\n<|image|>Describe this image in two sentences<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
I get the following tokens:
```console
(gdb) p tokens_list
$1 = std::vector of length 20, capacity 160 = {128000, 128006, 882, 128007, 1432, 27, 91, 1843, 91, 29, 75885, 420, 
  2217, 304, 1403, 23719, 128009, 128006, 78191, 128007}
(gdb) p model.vocab.id_to_token[304]
$2 = {text = "Ġin", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[128007]
$3 = {text = "<|end_header_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p model.vocab.id_to_token[1432]
$4 = {text = "ĊĊĊ", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[27]
$5 = {text = "<", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[91]
$6 = {text = "|", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p model.vocab.id_to_token[1843]
$7 = {text = "image", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
Notice that the image token is not recognized and it should be a single token
and on multiple tokens.
```console
(gdb) p model.vocab.id_to_token[12856]
$9 = {text = "_window", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
```
So there must be something wrong with how I've configured the tokenizer
when converting the model I think.

```console
  25: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     26: STRING     |        1 | tokenizer.ggml.pre = 'llama-bpe'
     27: [STRING]   |   128257 | tokenizer.ggml.tokens
     28: [INT32]    |   128257 | tokenizer.ggml.token_type
     29: [STRING]   |   280147 | tokenizer.ggml.merges
     30: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     31: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     32: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     33: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n
     ```
```
This issue seems to have been related to the size of the vocabulary in the
model which is specified as 128256 but the actual size of the vocabulary is:
```console
(venv) $ python src/inspect-token-config.py
Loading tokenizer from: /home/danbev/work/ai/llama-models/Llama-3.2-11B-Vision-Instruct
Vocabulary size: 128257
Max token ID: 128256
Last token (by max ID): <|image|>, ID: 128256

Tokenized text: {'input_ids': [128000, 9906, 1917, 0, 1115, 374, 264, 1296, 13], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

Decoded text: <|begin_of_text|>Hello world! This is a test.

Sample vocabulary items:
ID 128247: <|reserved_special_token_238|>
ID 128248: <|reserved_special_token_239|>
ID 128249: <|reserved_special_token_240|>
ID 128250: <|reserved_special_token_241|>
ID 128251: <|reserved_special_token_242|>
ID 128252: <|reserved_special_token_243|>
ID 128253: <|reserved_special_token_244|>
ID 128254: <|reserved_special_token_245|>
ID 128255: <|reserved_special_token_246|>
ID 128256: <|image|>
```
So the actual size of the vocabulary is 128257 and not 128256. I've corrected
this (in a poor way but I will fix this later) and now the tokens look like
this:
```console
(gdb) p tokens_list
$1 = std::vector of length 16, capacity 159 = {128000, 128006, 882, 128007, 271, 128256, 75885, 420, 2217, 304,
  1403, 23719, 128009, 128006, 78191, 128007}
```
Compared to before:
```console
$1 = std::vector of length 20, capacity 160 = {128000, 128006, 882, 128007, 1432, 27, 91, 1843, 91, 29, 75885, 420, 
  2217, 304, 1403, 23719, 128009, 128006, 78191, 128007}
```
And we can check the image token:
```console
(gdb) p ctx.model.vocab.id_to_token[128256]
$2 = {text = "<|image|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
```
This seems to be a known issue and is mentioned [here](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mllama.md#usage-tips).


### Instruction prompting
Running locally I've been able to verify what the prompt should look like:
```console
(venv) $ python src/llama-3.2-instruct.py
The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████| 5/5 [00:11<00:00,  2.25s/it]
Some parameters are on the meta device because they were offloaded to the cpu.
prompt: ['<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What does the image show?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n']
<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>What does the image show?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The image shows the Eiffel Tower in Paris, France.<|eot_id|>
```

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What does the image show?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

With the following prompt I can get a somewhat reasonable response from the
model I converted (the unquantized model)::
```console
prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is LoRA?<|eot_id|><|start_header_id|>assistant<|end_header_id|>";
```

### Image types
The new vision api has the following types:
```c++
    // represent an RGB image
    // size of data must be equal to 3*nx*ny
    typedef struct llama_img {
        uint32_t nx;
        uint32_t ny;
        unsigned char * data;
    } llama_img;

    // Input data for llama_vision_decode
    typedef struct llama_batch_img {
        int32_t      n_imgs;
        llama_img ** imgs;
        llama_pos *  pos;
    } llama_batch_img;
```
So if we have one image this would be something like the following:
```console
llama_img img = {
    .nx = 560,
    .ny = 560,
    .data = some_rgb_data_pointer
};

llama_img* img_array[1] = { &img };
llama_pos positions[1] = { 0 };

// Create a batch with one image
llama_batch_img img_batch = {
    .n_imgs = 1,
    .imgs = img_array,
    .pos = positions
};

To access the first image in the batch:
img_batch.imgs[0]->nx
```

There are significant differences between the llava based one and mllama. The
llava one is a "prompt" based model where the image is first encoded into patch
embedding and then projected inte same space as the text embeddings, and these
are then passed to the llava model. In mllama cross-attention is used. So we
still have to encode the image into patch embeddings but these are not passed
back to the caller, instead they are used for the next when the next decode
happend and where the special image token ('<|image|>') is used. This is then
passed to the to the model and the cross-attention layers are used to combine
the image embeddings with the text embeddings.

So the embeddings for the image encoding has the following shape (the final
tenosor:
```console
(gdb) p embeddings->ne
$12 = {1280, 1601, 4, 1}
(gdb) p 1280 * 1601 * 4
$13 = 8197120

z_0
   0    [0                        1279]
        ...
        ...
        ...
        ...
   1600 [0                        1279]

z_1
   0    [0                        1279]
        ...
        ...
        ...
        ...
   1600 [0                        1279]

z_2
   0    [0                        1279]
        ...
        ...
        ...
        ...
   1600 [0                        1279]

z_3
   0    [0                        1279]
        ...
        ...
        ...
        ...
   1600 [0                        1279]
```
And `inp_raw` has the following shape:
```console
(gdb) p inp_raw->ne
$11 = {560, 560, 3, 4}
(gdb) p 560*560*3*4
$6 = 3763200
```

```console
(gdb) p ggml_nbytes(inp_raw)
$10 = 15052800
```

The size of the actual image is:
```console
(gdb) p nx
$7 = 1280
(gdb) p ny
$8 = 853
(gdb) p n
$9 = 1091840
```

When we build the graph we have the following code:
```c++
    const int num_padding_patches = 8 - (embeddings->ne[1] % 8) % 8;
                                                                                
    embeddings = ggml_pad(ctx0, embeddings, 0, num_padding_patches, 0, 0);          
    embeddings = ggml_view_3d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1] * embeddings->ne[2], batch_size, embeddings->nb[1], embeddings->nb[2] * embeddings->ne[3], 0)
```
```console
(gdb) p num_padding_patches
$21 = 7
(gdb) p embeddings->ne
$22 = {1280, 1601, 4, 1}
```
Lets try to visualize this:
```console
z_0   
        0 [0                        1279]
        ...
        ...
        ...
      1600[0                        1279]

z_1   
        0 [0                        1279]
        ...
        ...
        ...
      1600[0                        1279]

z_2   
        0 [0                        1279]
        ...
        ...
        ...
      1600[0                        1279]

z_3   
        0 [0                        1279]
        ...
        ...
        ...
      1600[0                        1279]
```
So this is shape of the embedding tensor before the padding. And we are only
going to pad the second dimension by 7.
```
z_0   
        0 [0                        1279]
        ...
        ...
        ...
      1607[0                        1279]

z_1   
        0 [0                        1279]
        ...
        ...
        ...
      1607[0                        1279]

z_2   
        0 [0                        1279]
        ...
        ...
        ...
      1607[0                        1279]

z_3   
        0 [0                        1279]
        ...
        ...
        ...
      1607[0                        1279]
```
And this is the shape after the padding operation:
```console
(gdb) p embeddings->ne
$23 = {1280, 1608, 4, 1}
```
Now lets look at the reshaping operation after the padding:
```console
embeddings = ggml_view_3d(ctx0,
    embeddings,
    embeddings->ne[0],
    embeddings->ne[1] * embeddings->ne[2],
    batch_size,
    embeddings->nb[1],
    embeddings->nb[2] * embeddings->ne[3],
    0);
```
So the first dimension is kept the same. The second dimension (1608) is
multiplied by the third dimension (4) which is 6432. The third dimension
is set to the batch_size which is currently 1. The strides are set and
the offset is zero resulting in:
```console
(gdb) p embeddings->ne
$28 = {1280, 6432, 1, 1}
```

Output from ollama:
```console
danbev] num_positions: 1601
[danbev] image data length: 15052800
[danbev] image width: 560
[danbev] image height: 560
[danbev] inp_raw[0]: 560
[danbev] inp_raw[1]: 560
[danbev] inp_raw[2]: 3
[danbev] inp_raw[3]: 4
[danbev] data size:  15052800
[danbev] ggml_nbytes: 15052800
[danbev] copy embeddings size: 104923136


[danbev] mllama_image_load_data n: 15052800
[danbev] mllama_image_load_data aspect_ratio_id: 6
[danbev] mllama_n_positions: 1601
[danbev] mllama_n_tiles: 4
[danbev] numTokens: 6404
[danbev] numEmbed: 4096
[danbev] Total size (numTokens * numEmbed): 26230784
```

```console
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:357 msg="[danbev] sequence loop...." seqIdx=0
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.952+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128006 pos=0 seqIds=[0]
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.952+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=882 pos=1 seqIds=[0]
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.952+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128007 pos=2 seqIds=[0]
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.952+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=271 pos=3 seqIds=[0]
time=2024-11-23T10:07:22.952+01:00 level=INFO source=runner.go:432 msg="[danbev] before decode" batch="&{c:{n_tokens:4 token:0x78ceec1a1f00 embd:<nil> n_embd:0 pos:0x78ceec1a2710 n_seq_id:0x78ceec1a2f20 seq_id:0x78ceec28bab0 logits:0x78cecc707bc0 all_pos_0:0 all_pos_1:0 all_seq_id:0 _:[0 0 0 0]} batchSize:512 maxSeq:1 embedSize:0}"
[danbev] llama.cpp set_inputs NO batch.embd

time=2024-11-23T10:07:22.968+01:00 level=INFO source=runner.go:438 msg="[danbev] after decode"

time=2024-11-23T10:07:22.968+01:00 level=INFO source=runner.go:357 msg="[danbev] sequence loop...." seqIdx=0
time=2024-11-23T10:07:22.968+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:22.968+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=true batchSize=1 maxSeq=1 embedSize=26230784 token=0 pos=4 seqIds=[0]
time=2024-11-23T10:07:23.013+01:00 level=INFO source=runner.go:432 msg="[danbev] before decode" batch="&{c:{n_tokens:1 token:<nil> embd:0x78ca97a00010 n_embd:26230784 pos:0x78ceec018d20 n_seq_id:0x78ceec018d40 seq_id:0x78ceec018d60 logits:0x78ceec018da0 all_pos_0:0 all_pos_1:0 all_seq_id:0 _:[0 0 0 0]} batchSize:1 maxSeq:1 embedSize:26230784}"
[danbev] llama.cpp set_inputs batch.embd
[danbev] llama.cpp --------- cross_attn_state from batch.embd -------------

time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:438 msg="[danbev] after decode"

time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:357 msg="[danbev] sequence loop...." seqIdx=0
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128256 pos=5 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=3923 pos=6 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=374 pos=7 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=304 pos=8 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=420 pos=9 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=2217 pos=10 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=30 pos=11 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128009 pos=12 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128006 pos=13 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=78191 pos=14 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=128007 pos=15 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:414 msg="[danbev] adding embeddings (input.embed) to batch as embed"
time=2024-11-23T10:07:23.167+01:00 level=INFO source=llama.go:375 msg="[danbev] Add token" isEmbedding=false batchSize=512 maxSeq=1 embedSize=0 token=271 pos=16 seqIds=[0]
time=2024-11-23T10:07:23.167+01:00 level=INFO source=runner.go:432 msg="[danbev] before decode" batch="&{c:{n_tokens:12 token:0x78ceec1a1f00 embd:<nil> n_embd:0 pos:0x78ceec1a2710 n_seq_id:0x78ceec1a2f20 seq_id:0x78ceec28bab0 logits:0x78cecc707bc0 all_pos_0:0 all_pos_1:0 all_seq_id:0 _:[0 0 0 0]} batchSize:512 maxSeq:1 embedSize:0}"
```

```
(gdb) p ctx.model.vocab.id_to_token[128006]
$1 = {text = "<|start_header_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p ctx.model.vocab.id_to_token[882]
$2 = {text = "user", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p ctx.model.vocab.id_to_token[128007]
$3 = {text = "<|end_header_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p ctx.model.vocab.id_to_token[271]
$5 = {text = "ĊĊ", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}  
```
prompt="<|start_header_id|>user<|end_header_id|>\n\n[img-0]<|image|>What is in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


### Inspect ollama Llama-3.2-11B-Vision-Instruct model
Inspecting the model:
```console
(venv) $ cat ~/.ollama/models/manifests/registry.ollama.ai/x/llama3.2-vision/latest | jq '.layers[0].digest'
"sha256:652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9"

(venv) $ ./inspect-model.sh /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9
INFO:gguf-dump:* Loading: /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9
```
Inspecting the projector:
```console
(venv) $ ./inspect-model.sh /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73
```

### Model issues

#### pre_tile_position_embeddings weights
This tensor is defined as
```
46080 |  5120,     9,     1,     1 | F16     | v.enc.pre_tile_position_embd.weight
```
And in ollama's model is is:
```
 46080 |  5120,     9,     1,     1 | F32     | v.pre_tile_position_embd.weight
```
Notice that the type if F32 and not F16. And this is also quantized if we
quantize our model.

ours:
```console
 503:   31457280 |    7680,   4096,     1,     1 | Q4_1    | v.enc.mmproj.weight           <--- wrong data type   
    504:     752640 |    14,    14,     3,  1280 | F16     | v.enc.patch_embd.weight
    505:    2049280 |  1280,  1601,     1,     1 | F32     | v.enc.position_embd
    506:          1 |     1,     1,     1,     1 | F32     | v.enc.position_gate
    507:       1280 |  1280,     1,     1,     1 | F32     | v.enc.post_ln.bias
    508:       1280 |  1280,     1,     1,     1 | F32     | v.enc.post_ln.weight
    509:      46080 |  5120,     9,     1,     1 | F32     | v.enc.post_tile_position_embd.weight
    510:          1 |     1,     1,     1,     1 | F32     | v.enc.post_tile_position_gate
    511:       1280 |  1280,     1,     1,     1 | F32     | v.enc.pre_ln.bias
    512:       1280 |  1280,     1,     1,     1 | F32     | v.enc.pre_ln.weight
    513:      46080 |  5120,     9,     1,     1 | F32     | v.enc.pre_tile_position_embd.weight
    514:          1 |     1,     1,     1,     1 | F32     | v.enc.pre_tile_position_gate
    515:   73774080 | 8197120,   9,     1,     1 | F16     | v.enc.tile_position_embd.weight <--- wrong data type

```
ollama:
```console
      1:   31457280 |  7680,  4096,     1,     1 | F16     | mm.0.weight
      2:       4096 |  4096,     1,     1,     1 | F32     | mm.0.bias
      3:       1280 |  1280,     1,     1,     1 | F32     | v.class_embd
      4:     752640 |    14,    14,     3,  1280 | F16     | v.patch_embd.weight
      5:          1 |     1,     1,     1,     1 | F32     | v.tile_position_embd.gate
      6:          1 |     1,     1,     1,     1 | F32     | v.position_embd.gate
      7:    2049280 |  1280,  1601,     1,     1 | F16     | v.position_embd.weight
      8:   73774080 | 8197120,   9,     1,     1 | F32     | v.tile_position_embd.weight
      9:          1 |     1,     1,     1,     1 | F32     | v.pre_tile_position_embd.gate
     10:      46080 |  5120,     9,     1,     1 | F32     | v.pre_tile_position_embd.weight
     11:          1 |     1,     1,     1,     1 | F32     | v.post_tile_position_embd.gate
     12:      46080 |  5120,     9,     1,     1 | F32     | v.post_tile_position_embd.weight
     13:       1280 |  1280,     1,     1,     1 | F32     | v.pre_ln.weight
     14:       1280 |  1280,     1,     1,     1 | F32     | v.pre_ln.bias
     15:       1280 |  1280,     1,     1,     1 | F32     | v.post_ln.weight
     16:       1280 |  1280,     1,     1,     1 | F32     | v.post_ln.bias
```

llama.cpp:
```console
 4:  525369344 |  4096, 128264,     1,     1 | Q4_1    | token_embd.weight
```

#### `v.patch_embd`
ollama:
```console
752640 |    14,    14,     3,  1280 | F16     | v.patch_embd.weight
```

```console
752640 |    14,    14,     3,  1280 | F16     | v.enc.patch_embd.weight
```

### issue
When I run the Llava example the tensor for the conv2d operation will be
placed on the GPU:
```console
Backend type: CUDA0
inp_conv2d[0] = 0.000000 (isnan=0)
inp_conv2d[1] = 0.000000 (isnan=0)
inp_conv2d[2] = 0.000000 (isnan=0)
inp_conv2d[3] = 0.000000 (isnan=0)
inp_conv2d[4] = 0.000000 (isnan=0)
inp_conv2d[5] = 0.000000 (isnan=0)
inp_conv2d[6] = 0.000000 (isnan=0)
inp_conv2d[7] = 0.000000 (isnan=0)
inp_conv2d[8] = 0.000000 (isnan=0)
inp_conv2d[9] = 0.000000 (isnan=0)
```
The above was printed before the computation graph was executed by the
scheduler.

```console
(gdb) p *model.patch_embeddings
$4 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555605e5a0, ne = {14, 14, 3, 1024}, nb = {
    2, 28, 392, 1176}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ffd553947a0,
  name = "v.enc.embd.patch.weight", '\000' <repeats 40 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```

```console
(venv) $ ./inspect-model.sh ../llama.cpp/models/llava-1.5.7b-hf.gguf | grep patch
INFO:gguf-dump:* Loading: ../llama.cpp/models/llava-1.5.7b-hf.gguf
     25: UINT32     |        1 | vision.patch_size = 14
     34: STRING     |        1 | vision.clip.patch_merge_type = 'flat'
     98:     602112 |    14,    14,     3,  1024 | F16     | v.enc.embd.patch.weight
```

```console
(gdb) p *inp_raw
$1 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x555555ddeda0, ne = {336, 336, 3, 1}, nb = {
    4, 1344, 451584, 1354752}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 1, src = {0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ffab2441000,
  name = "inp_raw", '\000' <repeats 56 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}

Backend type: CPU
inp_raw[0] = 1.171218 (isnan=0)
inp_raw[1] = 1.171218 (isnan=0)
inp_raw[2] = 1.171218 (isnan=0)
inp_raw[3] = 1.171218 (isnan=0)
inp_raw[4] = 1.171218 (isnan=0)
inp_raw[5] = 1.171218 (isnan=0)
inp_raw[6] = 1.127423 (isnan=0)
inp_raw[7] = 1.112824 (isnan=0)
inp_raw[8] = 1.069029 (isnan=0)
inp_raw[9] = 1.025234 (isnan=0)

Backend type: CUDA0
inp_conv2d[0] = 0.000000 (isnan=0)
inp_conv2d[1] = 0.000000 (isnan=0)
inp_conv2d[2] = 0.000000 (isnan=0)
inp_conv2d[3] = 0.000000 (isnan=0)
inp_conv2d[4] = 0.000000 (isnan=0)
inp_conv2d[5] = 0.000000 (isnan=0)
inp_conv2d[6] = 0.000000 (isnan=0)
inp_conv2d[7] = 0.000000 (isnan=0)
inp_conv2d[8] = 0.000000 (isnan=0)
inp_conv2d[9] = 0.000000 (isnan=0)

Backend type: CPU
patch_embeddings[0] = 0.000000 (isnan=0)
patch_embeddings[1] = 0.000000 (isnan=0)
patch_embeddings[2] = 0.000000 (isnan=0)
patch_embeddings[3] = 0.000000 (isnan=0)
patch_embeddings[4] = 0.000000 (isnan=0)
patch_embeddings[5] = 0.000000 (isnan=0)
patch_embeddings[6] = 0.000000 (isnan=0)
patch_embeddings[7] = -0.000000 (isnan=0)
patch_embeddings[8] = -0.000000 (isnan=0)
patch_embeddings[9] = 0.000000 (isnan=0)
```


MLlama example:
```console
inp_raw Backend type: CPU
inp_raw[0] = 1.156620 (isnan=0)
inp_raw[1] = 1.156620 (isnan=0)
inp_raw[2] = 1.171218 (isnan=0)
inp_raw[3] = 1.171218 (isnan=0)
inp_raw[4] = 1.171218 (isnan=0)
inp_raw[5] = 1.156620 (isnan=0)
inp_raw[6] = 1.171218 (isnan=0)
inp_raw[7] = 1.156620 (isnan=0)
inp_raw[8] = 1.112824 (isnan=0)
inp_raw[9] = 1.098226 (isnan=0)

inp_after_conv2d Backend type: CUDA0
inp_after_conv2d[0] = 0.000000 (isnan=0)
inp_after_conv2d[1] = 0.000000 (isnan=0)
inp_after_conv2d[2] = 0.000000 (isnan=0)
inp_after_conv2d[3] = 0.000000 (isnan=0)
inp_after_conv2d[4] = 0.000000 (isnan=0)
inp_after_conv2d[5] = 0.000000 (isnan=0)
inp_after_conv2d[6] = 0.000000 (isnan=0)
inp_after_conv2d[7] = 0.000000 (isnan=0)
inp_after_conv2d[8] = 0.000000 (isnan=0)
inp_after_conv2d[9] = 0.000000 (isnan=0)

Backend type: CPU
patch_embeddings[0] = 0.000000 (isnan=0)
patch_embeddings[1] = -0.000000 (isnan=0)
patch_embeddings[2] = 0.000000 (isnan=0)
patch_embeddings[3] = 0.000000 (isnan=0)
patch_embeddings[4] = 0.000000 (isnan=0)
patch_embeddings[5] = 0.000000 (isnan=0)
patch_embeddings[6] = 0.000000 (isnan=0)
patch_embeddings[7] = 0.000000 (isnan=0)
patch_embeddings[8] = -0.000000 (isnan=0)
patch_embeddings[9] = 0.000000 (isnan=0)


```
```console
(gdb) p *model.patch_embeddings
$1 = {type = GGML_TYPE_F16, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x55555a4830e0, ne = {14, 14, 3, 1280}, nb = {
    2, 28, 392, 1176}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x0, 0x0, 0x0, 0x0,




Backend type: CPU
after_pre_tile_position_embeddings[0] = 0.242798 (isnan=0)
after_pre_tile_position_embeddings[1] = -0.385010 (isnan=0)
after_pre_tile_position_embeddings[2] = 0.044067 (isnan=0)
after_pre_tile_position_embeddings[3] = -0.288818 (isnan=0)
after_pre_tile_position_embeddings[4] = -0.059143 (isnan=0)
after_pre_tile_position_embeddings[5] = -0.113281 (isnan=0)
after_pre_tile_position_embeddings[6] = -0.445801 (isnan=0)
after_pre_tile_position_embeddings[7] = 0.079895 (isnan=0)
after_pre_tile_position_embeddings[8] = -0.218384 (isnan=0)
after_pre_tile_position_embeddings[9] = -0.167725 (isnan=0)
Backend type: CPU
embeddings[0] = 0.242798 (isnan=0)
embeddings[1] = -0.385010 (isnan=0)
embeddings[2] = 0.044067 (isnan=0)
embeddings[3] = -0.288818 (isnan=0)
embeddings[4] = -0.059143 (isnan=0)
embeddings[5] = -0.113281 (isnan=0)
embeddings[6] = -0.445801 (isnan=0)
embeddings[7] = 0.079895 (isnan=0)
embeddings[8] = -0.218384 (isnan=0)
embeddings[9] = -0.167725 (isnan=0)
Backend type: CPU
after_class_embeddings[0] = -0.167969 (isnan=0)
after_class_embeddings[1] = 0.072754 (isnan=0)
after_class_embeddings[2] = -0.002396 (isnan=0)
after_class_embeddings[3] = 0.021851 (isnan=0)
after_class_embeddings[4] = 0.035400 (isnan=0)
after_class_embeddings[5] = -0.068359 (isnan=0)
after_class_embeddings[6] = 0.074219 (isnan=0)
after_class_embeddings[7] = -0.016602 (isnan=0)
after_class_embeddings[8] = -0.004120 (isnan=0)
after_class_embeddings[9] = 0.045898 (isnan=0)
Backend type: CPU
positions[0] = 0
positions[1] = 1
positions[2] = 2
positions[3] = 3
positions[4] = 4
positions[5] = 5
positions[6] = 6
positions[7] = 7
positions[8] = 8
positions[9] = 9
Backend type: CUDA0
after_position_embd[0] = -0.216930 (isnan=0)
after_position_embd[1] = 0.131767 (isnan=0)
after_position_embd[2] = -0.003434 (isnan=0)
after_position_embd[3] = 0.061733 (isnan=0)
after_position_embd[4] = 0.079498 (isnan=0)
after_position_embd[5] = -0.074155 (isnan=0)
after_position_embd[6] = 0.101131 (isnan=0)
after_position_embd[7] = 0.004637 (isnan=0)
after_position_embd[8] = 0.001554 (isnan=0)
after_position_embd[9] = 0.087240 (isnan=0)


    0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ffb804a4dc0,
  name = "v.enc.patch_embd.weight", '\000' <repeats 40 times>, extra = 0x0,
  padding = "\000\000\000\000\000\000\000"}
```

```console
(venv) $ ./inspect-model.sh models/llama-3-2-11b.gguf | grep patch
INFO:gguf-dump:* Loading: models/llama-3-2-11b.gguf
     37: UINT32     |        1 | vision.patch_size = 14
    173:     752640 |    14,    14,     3,  1280 | F16     | v.enc.patch_embd.weight
```
I should fix the inconsistency with the naming here, this shouuld be `v.enc.embd.patch`.

```console
(gdb) p *inp_raw
$6 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x5555567393b0, ne = {560, 560, 3, 4}, nb = {
    4, 2240, 1254400, 3763200}, op = GGML_OP_NONE, op_params = {0 <repeats 16 times>}, flags = 1, src = {0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x7ff8a1f45000,
  name = "inp_raw", '\000' <repeats 56 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}
```


llama.cpp:
```console
(venv) $ python read-tensor.py models/llama-3-2-11b.gguf v.enc.embd.pos_gate

Tensor Information:
Name: v.enc.embd.pos_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = -1.328125
```
Ollama:

(venv) $ python read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.position_embd.gate

Tensor Information:
Name: v.position_embd.gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 1.8687903881072998
``
And the orignal model has this value:
```console
Tensor Information:
Name: vision_model.gated_positional_embedding.gate
Shape: torch.Size([1])
Type: torch.bfloat16
First 10 values:
[0] = -1.328125
```


```console
Tensor Information:
Name: vision_model.gated_positional_embedding.embedding
Shape: torch.Size([1601, 1280])
Type: torch.bfloat16
First 10 values:
[0] = 0.036865234375
[1] = -0.04443359375
[2] = 0.000782012939453125
[3] = -0.030029296875
[4] = -0.033203125
[5] = 0.004364013671875
[6] = -0.020263671875
[7] = -0.0159912109375
[8] = -0.0042724609375
[9] = -0.0311279296875
```


(gdb) p *KQ_mul
$1 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {6432, 6432, 16, 1}, nb = {4,
    25728, 165482496, 2647719936}, op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>}, flags = 0, src = {
    0x55555a4ac420, 0x55555a4abe60, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0,
  data = 0x0, name = "KQ_mul-0", '\000' <repeats 55 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}

(gdb) p *KQV_mul
$5 = {type = GGML_TYPE_F32, backend = GGML_BACKEND_TYPE_CPU, buffer = 0x0, ne = {80, 6432, 16, 1}, nb = {4, 320,
    2058240, 32931840}, op = GGML_OP_MUL_MAT, op_params = {0 <repeats 16 times>}, flags = 0, src = {0x55555a4ac9e0,
    0x55555a4ace30, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0}, view_src = 0x0, view_offs = 0, data = 0x0,
  name = '\000' <repeats 63 times>, extra = 0x0, padding = "\000\000\000\000\000\000\000"}


### Image processing issue
Using the same image this is the output from ollama:
```console
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=0 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=1 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=2 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=3 data=63
Bytes [0]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=4 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=5 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=6 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=7 data=63
Bytes [1]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=8 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=9 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=10 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=11 data=63
Bytes [2]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=12 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=13 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=14 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=15 data=63
Bytes [3]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=16 data=28
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=17 data=12
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=18 data=148
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:70 msg="[danbev] Image.NewEmbed" i=19 data=63
Bytes [4]: 28, 12, 148, 63

time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=0 data=1.1566195487976074
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=1 data=1.1566195487976074
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=2 data=1.1566195487976074
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=3 data=1.1566195487976074
Bytes [5]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=4 data=1.1566195487976074
Bytes [6]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=5 data=1.1712180376052856
Bytes [7]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=6 data=1.1712180376052856
Bytes [8]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=7 data=1.1566195487976074
Bytes [9]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=8 data=1.1712180376052856
Bytes [10]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=9 data=1.1712180376052856
Bytes [11]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=10 data=1.1712180376052856
Bytes [12]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=11 data=1.1712180376052856
Bytes [13]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=12 data=1.1712180376052856
Bytes [14]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=13 data=1.1566195487976074
Bytes [15]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=14 data=1.1566195487976074
Bytes [16]: 121, 234, 149, 63
Bytes [17]: 121, 234, 149, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=15 data=1.1712180376052856
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=16 data=1.1712180376052856
Bytes [18]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=17 data=1.1712180376052856
Bytes [19]: 28, 12, 148, 63
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=18 data=1.1566195487976074
time=2024-11-28T08:15:38.804+01:00 level=INFO source=image.go:80 msg="As float32 " i=19 data=1.1566195487976074
And this is the output I get after processing:
```console
Normalized[0] = 1.156620
Normalized[1] = 1.156620
Normalized[2] = 1.156620
Normalized[3] = 1.156620
Normalized[4] = 1.156620
Normalized[5] = 1.156620
Normalized[6] = 1.171218
Normalized[7] = 1.171218
Normalized[8] = 1.156620
Normalized[9] = 1.171218
Normalized[10] = 1.171218
Normalized[11] = 1.171218
Normalized[12] = 1.171218
Normalized[13] = 1.171218
Normalized[14] = 1.171218
Normalized[15] = 1.156620
Normalized[16] = 1.156620
Normalized[17] = 1.156620
Normalized[18] = 1.171218
Normalized[19] = 1.171218
[simple-vision-mllama] loaded image data[0] = 28
[simple-vision-mllama] loaded image data[1] = 12
[simple-vision-mllama] loaded image data[2] = 148
[simple-vision-mllama] loaded image data[3] = 63
[simple-vision-mllama] loaded image data[4] = 28
[simple-vision-mllama] loaded image data[5] = 12
[simple-vision-mllama] loaded image data[6] = 148
[simple-vision-mllama] loaded image data[7] = 63
[simple-vision-mllama] loaded image data[8] = 28
[simple-vision-mllama] loaded image data[9] = 12
[simple-vision-mllama] loaded image data[10] = 148
[simple-vision-mllama] loaded image data[11] = 63
[simple-vision-mllama] loaded image data[12] = 28
[simple-vision-mllama] loaded image data[13] = 12
[simple-vision-mllama] loaded image data[14] = 148
[simple-vision-mllama] loaded image data[15] = 63
[simple-vision-mllama] loaded image data[16] = 28
[simple-vision-mllama] loaded image data[17] = 12
[simple-vision-mllama] loaded image data[18] = 148
[simple-vision-mllama] loaded image data[19] = 63
```
So these look pretty similar. Now, let print out the values before
we set them as inputs before the graph computation.

This is from llama.cpp (vision.cpp):
```console
Before encode_image_with_ca_vision:
Input image size: 560x560
Input data[0] = 1.156620
Input data[1] = 1.156620
Input data[2] = 1.156620
Input data[3] = 1.156620
Input data[4] = 1.156620
Input data[5] = 1.156620
Input data[6] = 1.171218
Input data[7] = 1.171218
Input data[8] = 1.156620
Input data[9] = 1.171218
Input data[10] = 1.171218
Input data[11] = 1.171218
Input data[12] = 1.171218
Input data[13] = 1.171218
Input data[14] = 1.171218
Input data[15] = 1.156620
Input data[16] = 1.156620
Input data[17] = 1.156620
Input data[18] = 1.171218
Input data[19] = 1.171218


inp_raw backend type: CPU
First values of inp_raw:
inp_raw[0] = 1.156620
inp_raw[1] = 1.156620
inp_raw[2] = 1.171218
inp_raw[3] = 1.171218
inp_raw[4] = 1.171218
inp_raw[5] = 1.156620
inp_raw[6] = 1.171218
inp_raw[7] = 1.156620
inp_raw[8] = 1.112824
inp_raw[9] = 1.098226
inp_raw[10] = 1.083627
inp_raw[11] = 0.996037
inp_raw[12] = 0.981439
inp_raw[13] = 0.908446
inp_raw[14] = 0.850053
inp_raw[15] = 0.820856
inp_raw[16] = 0.820856
inp_raw[17] = 0.820856
inp_raw[18] = 0.791659
inp_raw[19] = 0.791659
inp_raw[20] = 0.791659
inp_raw[21] = 0.806257
inp_raw[22] = 0.937643
inp_raw[23] = 1.142021
inp_raw[24] = 1.127423
inp_raw[25] = 0.966840
inp_raw[26] = 1.419391
inp_raw[27] = 1.638368
inp_raw[28] = 1.725958
inp_raw[29] = 1.448588
```

```console
First values after inp_raw:
inp_raw[0] = 1.156620 (isnan=0)
inp_raw[1] = 1.156620 (isnan=0)
inp_raw[2] = 1.156620 (isnan=0)
inp_raw[3] = 1.156620 (isnan=0)
inp_raw[4] = 1.156620 (isnan=0)
inp_raw[5] = 1.171218 (isnan=0)
inp_raw[6] = 1.171218 (isnan=0)
inp_raw[7] = 1.156620 (isnan=0)
inp_raw[8] = 1.171218 (isnan=0)
inp_raw[9] = 1.171218 (isnan=0)
inp_raw[10] = 1.171218 (isnan=0)
inp_raw[11] = 1.171218 (isnan=0)
inp_raw[12] = 1.171218 (isnan=0)
inp_raw[13] = 1.156620 (isnan=0)
inp_raw[14] = 1.156620 (isnan=0)
inp_raw[15] = 1.171218 (isnan=0)
inp_raw[16] = 1.171218 (isnan=0)
inp_raw[17] = 1.171218 (isnan=0)
inp_raw[18] = 1.156620 (isnan=0)
inp_raw[19] = 1.156620 (isnan=0)
inp_raw[20] = 1.142021 (isnan=0)
inp_raw[21] = 1.127423 (isnan=0)
inp_raw[22] = 1.112824 (isnan=0)
inp_raw[23] = 1.112824 (isnan=0)
inp_raw[24] = 1.112824 (isnan=0)
inp_raw[25] = 1.112824 (isnan=0)
inp_raw[26] = 1.098226 (isnan=0)
inp_raw[27] = 1.098226 (isnan=0)
inp_raw[28] = 1.025234 (isnan=0)
inp_raw[29] = 0.996037 (isnan=0)
```

```console
[danbev] num_positions: 1601
[danbev] image data length: 15052800
[danbev] image width: 560
[danbev] image height: 560
```
So inp_raw is image witdh * image height * channels * tiles:
```console
(gdb) p ggml_nelements(inp_raw)
$4 = 3763200
(gdb) p ggml_nelements(inp_raw) * 4
$5 = 15052800
(gdb) p ggml_nbytes(inp_raw)
$6 = 15052800

(gdb) p 560 * 560 * 3 * 4
$7 = 3763200
(gdb) p 560 * 560 * 3 * 4 * 4
$8 = 15052800
```
And we have 4 bytes per entry which is the last 4. Now, in ollama they can
just to the following:
```c++
    {
        struct ggml_tensor *inp_raw = ggml_graph_get_tensor(gf, "inp_raw");
        ggml_backend_tensor_set(inp_raw, imgs->data[0].data.data(), 0, ggml_nbytes(inp_raw));
    }
```
What is the size of the image data that is passed in here?
In my example the size is Copying 13102080 bytes of data


### Prediction issue
```console
inp_after_conv2d Backend type: CUDA0
inp_after_conv2d[0] = -0.002579 (isnan=0)
inp_after_conv2d[1] = 0.000140 (isnan=0)
inp_after_conv2d[2] = 0.001678 (isnan=0)
inp_after_conv2d[3] = -0.001488 (isnan=0)
inp_after_conv2d[4] = 0.000246 (isnan=0)
inp_after_conv2d[5] = 0.000538 (isnan=0)
inp_after_conv2d[6] = -0.003235 (isnan=0)
inp_after_conv2d[7] = 0.000212 (isnan=0)
inp_after_conv2d[8] = -0.000094 (isnan=0)
inp_after_conv2d[9] = 0.001617 (isnan=0)
```
```console
inp_after_conv2d[0] = -4.334547 (isnan=0)
inp_after_conv2d[1] = -0.271873 (isnan=0)
inp_after_conv2d[2] = -0.259376 (isnan=0)
inp_after_conv2d[3] = 0.546166 (isnan=0)
inp_after_conv2d[4] = -0.929878 (isnan=0)
inp_after_conv2d[5] = -3.958198 (isnan=0)
inp_after_conv2d[6] = 3.839755 (isnan=0)
inp_after_conv2d[7] = 9.393863 (isnan=0)
inp_after_conv2d[8] = 6.277499 (isnan=0)
inp_after_conv2d[9] = -2.536955 (isnan=0)
```


pre_tile_position_embeddings_gate[0] = 0.635149 (isnan=0)





inp_after_conv2d Backend type: CUDA0
inp_after_conv2d[0] = -0.002579 (isnan=0)
inp_after_conv2d[1] = 0.000140 (isnan=0)
inp_after_conv2d[2] = 0.001678 (isnan=0)
inp_after_conv2d[3] = -0.001488 (isnan=0)
inp_after_conv2d[4] = 0.000246 (isnan=0)
inp_after_conv2d[5] = 0.000538 (isnan=0)
inp_after_conv2d[6] = -0.003235 (isnan=0)
inp_after_conv2d[7] = 0.000212 (isnan=0)
inp_after_conv2d[8] = -0.000094 (isnan=0)
inp_after_conv2d[9] = 0.001617 (isnan=0)


Tensor type: f32
Tensor backend buffer type: CUDA0
inp_after_conv2d[0] = -4.334547 (isnan=0)
inp_after_conv2d[1] = -0.271873 (isnan=0)
inp_after_conv2d[2] = -0.259376 (isnan=0)
inp_after_conv2d[3] = 0.546166 (isnan=0)
inp_after_conv2d[4] = -0.929878 (isnan=0)
inp_after_conv2d[5] = -3.958198 (isnan=0)
inp_after_conv2d[6] = 3.839755 (isnan=0)
inp_after_conv2d[7] = 9.393863 (isnan=0)
inp_after_conv2d[8] = 6.277499 (isnan=0)
inp_after_conv2d[9] = -2.536955 (isnan=0)



Tensor type: f32
inp_after_conv2d Backend type: CUDA0
inp_after_conv2d[0] = 4.817005 (isnan=0)
inp_after_conv2d[1] = -1.839836 (isnan=0)
inp_after_conv2d[2] = 0.149719 (isnan=0)
inp_after_conv2d[3] = -1.052750 (isnan=0)
inp_after_conv2d[4] = 0.045052 (isnan=0)
inp_after_conv2d[5] = 5.482178 (isnan=0)
inp_after_conv2d[6] = -20.486389 (isnan=0)
inp_after_conv2d[7] = -20.188850 (isnan=0)
inp_after_conv2d[8] = -5.085636 (isnan=0)
inp_after_conv2d[9] = -12.865143 (isnan=0)

index 6:
model.pre_tile_positional_embeddings[6][0] = 0.008728 (isnan=0)
model.pre_tile_positional_embeddings[6][1] = -0.194336 (isnan=0)
model.pre_tile_positional_embeddings[6][2] = -0.074707 (isnan=0)
model.pre_tile_positional_embeddings[6][3] = -0.125000 (isnan=0)
model.pre_tile_positional_embeddings[6][4] = -0.125977 (isnan=0)
model.pre_tile_positional_embeddings[6][5] = -0.157227 (isnan=0)
model.pre_tile_positional_embeddings[6][6] = 0.049316 (isnan=0)
model.pre_tile_positional_embeddings[6][7] = -0.098145 (isnan=0)
model.pre_tile_positional_embeddings[6][8] = 0.080078 (isnan=0)
model.pre_tile_positional_embeddings[6][9] = -0.134766 (isnan=0)
model.pre_tile_positional_embeddings[6][10] = 0.010620 (isnan=0)
model.pre_tile_positional_embeddings[6][11] = -0.159180 (isnan=0)
model.pre_tile_positional_embeddings[6][12] = -0.187500 (isnan=0)
model.pre_tile_positional_embeddings[6][13] = -0.119141 (isnan=0)
model.pre_tile_positional_embeddings[6][14] = 0.025269 (isnan=0)
model.pre_tile_positional_embeddings[6][15] = -0.283203 (isnan=0)
model.pre_tile_positional_embeddings[6][16] = -0.166016 (isnan=0)
model.pre_tile_positional_embeddings[6][17] = -0.114258 (isnan=0)
model.pre_tile_positional_embeddings[6][18] = -0.213867 (isnan=0)
model.pre_tile_positional_embeddings[6][19] = -0.097168 (isnan=0)

The ones from get_rows:
pre_tile_position_embeddings[0] = 0.151923 (isnan=0)
pre_tile_position_embeddings[1] = 1.211459 (isnan=0)
pre_tile_position_embeddings[2] = 4.014451 (isnan=0)
pre_tile_position_embeddings[3] = 0.946178 (isnan=0)
pre_tile_position_embeddings[4] = -5.457583 (isnan=0)
pre_tile_position_embeddings[5] = -0.536786 (isnan=0)
pre_tile_position_embeddings[6] = -12.600449 (isnan=0)
pre_tile_position_embeddings[7] = -4.407394 (isnan=0)
pre_tile_position_embeddings[8] = -6.397855 (isnan=0)
pre_tile_position_embeddings[9] = -1.396894 (isnan=0)
pre_tile_position_embeddings[10] = -2.710128 (isnan=0)
pre_tile_position_embeddings[11] = 5.050854 (isnan=0)
pre_tile_position_embeddings[12] = 4.036346 (isnan=0)
pre_tile_position_embeddings[13] = -1.571802 (isnan=0)
pre_tile_position_embeddings[14] = -7.525723 (isnan=0)
pre_tile_position_embeddings[15] = -0.567268 (isnan=0)
pre_tile_position_embeddings[16] = -3.055859 (isnan=0)
pre_tile_position_embeddings[17] = 5.793323 (isnan=0)
pre_tile_position_embeddings[18] = 3.643304 (isnan=0)
pre_tile_position_embeddings[19] = -5.682245 (isnan=0)
pre_tile_position_embeddings[20] = -4.672400 (isnan=0)
pre_tile_position_embeddings[21] = -0.146351 (isnan=0)
pre_tile_position_embeddings[22] = 6.713018 (isnan=0)
pre_tile_position_embeddings[23] = -2.168883 (isnan=0)
pre_tile_position_embeddings[24] = -0.882346 (isnan=0)
pre_tile_position_embeddings[25] = -1.765321 (isnan=0)
pre_tile_position_embeddings[26] = 0.616241 (isnan=0)
pre_tile_position_embeddings[27] = -2.195093 (isnan=0)
pre_tile_position_embeddings[28] = -0.563926 (isnan=0)
pre_tile_position_embeddings[29] = -1.999931 (isnan=0)

```console
(venv) $ python read-tensor.py models/llama-3-2-11b.gguf v.enc.embd.patch.weight

Tensor Information:
Name: v.enc.embd.patch.weight
Shape: 14 x 14 x 3 x 1280
Type: F32
Total elements: 752640

First 10 values:
[0] = 0.0064697265625
[1] = 0.00543212890625
[2] = 4.4345855712890625e-05
[3] = -0.00933837890625
[4] = -0.00579833984375
[5] = 0.00836181640625
[6] = 0.0037994384765625
[7] = 0.0054931640625
[8] = 0.0157470703125
[9] = 0.00299072265625
```
```console
(venv) $ python read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.patch_embd.weight

Tensor Information:
Name: v.patch_embd.weight
Shape: 14 x 14 x 3 x 1280
Type: F16
Total elements: 752640

First 10 values:
[0] = 0.0064697265625
[1] = 0.00543212890625
[2] = 4.4345855712890625e-05
[3] = -0.00933837890625
[4] = -0.00579833984375
[5] = 0.00836181640625
[6] = 0.0037994384765625
[7] = 0.0054931640625
[8] = 0.0157470703125
[9] = 0.00299072265625
```
But when I read this tensor in llama.cpp I get this:
```console
Tensor type: f32
model.patch_embeddings[0] = 0.000000 (isnan=0)
model.patch_embeddings[1] = 0.978516 (isnan=0)
model.patch_embeddings[2] = 0.000000 (isnan=0)
model.patch_embeddings[3] = 0.961914 (isnan=0)
model.patch_embeddings[4] = 0.000000 (isnan=0)
model.patch_embeddings[5] = 0.528320 (isnan=0)
model.patch_embeddings[6] = 0.000000 (isnan=0)
model.patch_embeddings[7] = -1.024414 (isnan=0)
model.patch_embeddings[8] = 0.000000 (isnan=0)
model.patch_embeddings[9] = -0.967773 (isnan=0)
```
Ah this was because I was reading/converting to float16 because I've gone back
and forth on the type of tensors when testing.
```console
Backend type: CPU
Tensor type: f32
model.patch_embeddings[0] = 0.006470
model.patch_embeddings[1] = 0.005432
model.patch_embeddings[2] = 0.000044
model.patch_embeddings[3] = -0.009338
model.patch_embeddings[4] = -0.005798
model.patch_embeddings[5] = 0.008362
model.patch_embeddings[6] = 0.003799
model.patch_embeddings[7] = 0.005493
model.patch_embeddings[8] = 0.015747
model.patch_embeddings[9] = 0.002991
```
And in ollama they are:
```console
Tensor backend buffer type: CUDA0
Tensor type: f16
model.patch_embeddings[0] = 0.006470
model.patch_embeddings[1] = 0.005432
model.patch_embeddings[2] = 0.000044
model.patch_embeddings[3] = -0.009338
model.patch_embeddings[4] = -0.005798
model.patch_embeddings[5] = 0.008362
model.patch_embeddings[6] = 0.003799
model.patch_embeddings[7] = 0.005493
model.patch_embeddings[8] = 0.015747
model.patch_embeddings[9] = 0.002991
```

In ollama:
```console
pre_tile_position_embeddings[0] = 2.360522 (isnan=0)
pre_tile_position_embeddings[1] = 0.542406 (isnan=0)
pre_tile_position_embeddings[2] = 1.326151 (isnan=0)
pre_tile_position_embeddings[3] = -0.383138 (isnan=0)
pre_tile_position_embeddings[4] = -0.896916 (isnan=0)
pre_tile_position_embeddings[5] = -0.538128 (isnan=0)
pre_tile_position_embeddings[6] = -0.362522 (isnan=0)
pre_tile_position_embeddings[7] = 0.237987 (isnan=0)
pre_tile_position_embeddings[8] = 0.090309 (isnan=0)
pre_tile_position_embeddings[9] = -1.181206 (isnan=0)
```

### pre_tile_position_embeddings
```console
Backend type: CPU
Tensor type: f32
pre_tile_position_embeddings[0] = 0.172945 (isnan=0)
pre_tile_position_embeddings[1] = 1.207378 (isnan=0)
pre_tile_position_embeddings[2] = 3.902387 (isnan=0)
pre_tile_position_embeddings[3] = 1.060965 (isnan=0)
pre_tile_position_embeddings[4] = -5.529974 (isnan=0)
pre_tile_position_embeddings[5] = -0.492022 (isnan=0)
pre_tile_position_embeddings[6] = -12.454433 (isnan=0)
pre_tile_position_embeddings[7] = -4.333441 (isnan=0)
pre_tile_position_embeddings[8] = -6.424330 (isnan=0)
pre_tile_position_embeddings[9] = -1.306060 (isnan=0)
```
Now these values are gotten by using a get rows operation, so there is not
math operation or anything that should mess with the values. So we should see
the same values I believe.
```c++
        struct ggml_tensor * tile_position_embeddings = ggml_get_rows(ctx0, model.tile_position_embeddings, aspect_ratios);
        ggml_set_name(tile_position_embeddings, "tile_position_embd");
```
The aspect ratio tensor is 
```console
Backend type: CPU
Tensor type: f32
Values from row 6:
```
This is from llama.cpp:
```console
model.pre_tile_positional_embeddings[6][0] = 0.008728
model.pre_tile_positional_embeddings[6][1] = -0.194336
model.pre_tile_positional_embeddings[6][2] = -0.074707
model.pre_tile_positional_embeddings[6][3] = -0.125000
model.pre_tile_positional_embeddings[6][4] = -0.125977
model.pre_tile_positional_embeddings[6][5] = -0.157227
model.pre_tile_positional_embeddings[6][6] = 0.049316
model.pre_tile_positional_embeddings[6][7] = -0.098145
model.pre_tile_positional_embeddings[6][8] = 0.080078
model.pre_tile_positional_embeddings[6][9] = -0.134766
```
And this is from ollama:
```console
```

```console
(venv) $ ./read-tensor2.py models/llama-3-2-11b.gguf v.enc.pre_tile_pos_embd.weight 6

Tensor Information:
Name: v.enc.pre_tile_pos_embd.weight
Shape: 5120 x 9
Type: F32
Total elements: 46080

Values for row 6 (up to 10):
[6, 0] = 0.00872802734375
[6, 1] = -0.1943359375
[6, 2] = -0.07470703125
[6, 3] = -0.125
[6, 4] = -0.1259765625
[6, 5] = -0.1572265625
[6, 6] = 0.04931640625
[6, 7] = -0.09814453125
[6, 8] = 0.080078125
[6, 9] = -0.134765625
```

```console
(venv) $ ./read-tensor2.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.pre_tile_position_embd.weight 6

Tensor Information:
Name: v.pre_tile_position_embd.weight
Shape: 5120 x 9
Type: F32
Total elements: 46080

Values for row 6 (up to 10):
[6, 0] = 0.00872802734375
[6, 1] = -0.1943359375
[6, 2] = -0.07470703125
[6, 3] = -0.125
[6, 4] = -0.1259765625
[6, 5] = -0.1572265625
[6, 6] = 0.04931640625
[6, 7] = -0.09814453125
[6, 8] = 0.080078125
[6, 9] = -0.134765625
```

But this is what actually get selected:
```console
Backend type: CPU
Tensor type: f32
pre_tile_position_embeddings[0] = -0.633367 (isnan=0)
pre_tile_position_embeddings[1] = 6.685422 (isnan=0)
pre_tile_position_embeddings[2] = -1.039765 (isnan=0)
pre_tile_position_embeddings[3] = 0.576086 (isnan=0)
pre_tile_position_embeddings[4] = 4.381856 (isnan=0)
pre_tile_position_embeddings[5] = -0.598374 (isnan=0)
pre_tile_position_embeddings[6] = -7.988229 (isnan=0)
pre_tile_position_embeddings[7] = -0.992554 (isnan=0)
pre_tile_position_embeddings[8] = -2.655503 (isnan=0)
pre_tile_position_embeddings[9] = -5.103714 (isnan=0)
```

### Difference in models
In llama.cpp the value of `position_embeddings_gate` is:
```console
Backend type: CPU
Tensor type: f32
    model.position_embeddings_gate = -1.328125            MISMATCH

Backend type: CPU
Tensor type: f32
    model.pre_tile_position_embeddings_gate = 0.750000    MISMATCH

Backend type: CPU
Tensor type: f32
    model.post_tile_position_embeddings_gate = -0.197266  same value
```

But in ollama it is:
```console
Tensor type: f32
Tensor backend buffer type: CUDA0
tile_position_embeddings_gate?[0] = -0.868790              MISSMATCH

Tensor type: f32
Tensor backend buffer type: CUDA0
pre_tile_position_embeddings_gate = 0.635149               MISSMATCH

Tensor type: f32
Tensor backend buffer type: CUDA0
post_tile_position_embeddings_gate = -0.194746             same value
```

If I read this value from our model I get:
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf v.enc.embd.pos_gate

Tensor Information:
Name: v.enc.embd.pos_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = -1.328125
```
And if we read the same value from the safetensor we get:
```console
(venv) $ python read-safetensor.py

Tensor Information:
Name: vision_model.gated_positional_embedding.gate
Shape: torch.Size([1])
Type: torch.bfloat16
First 10 values:
[0] = -1.328125

Tensor Information:
Name: vision_model.post_tile_positional_embedding.gate
Shape: torch.Size([1])
Type: torch.bfloat16
First 10 values:
[0] = -0.197265625

Tensor Information:
Name: vision_model.pre_tile_positional_embedding.gate
Shape: torch.Size([1])
Type: torch.bfloat16
First 10 values:
[0] = 0.75
```

v.enc.embd.pos_gate      : -0.8687899708747864
v.enc.pre_tile_pos_gate  : 0.6351490020751953
v.enc.post_tile_pos_gate : -0.197265625



```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.position_embd.weight

Tensor Information:
Name: v.position_embd.weight
Shape: 1280 x 1601
Type: F16
Total elements: 2049280

First 10 values:
[0] = 0.036865234375
[1] = -0.04443359375
[2] = 0.000782012939453125
[3] = -0.030029296875
[4] = -0.033203125
[5] = 0.004364013671875
[6] = -0.020263671875
[7] = -0.0159912109375
[8] = -0.0042724609375
[9] = -0.0311279296875

(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf v.enc.embd.pos

Tensor Information:
Name: v.enc.embd.pos
Shape: 1280 x 1601
Type: F32
Total elements: 2049280

First 10 values:
[0] = 0.036865234375
[1] = -0.04443359375
[2] = 0.000782012939453125
[3] = -0.030029296875
[4] = -0.033203125
[5] = 0.004364013671875
[6] = -0.020263671875
[7] = -0.0159912109375
[8] = -0.0042724609375
[9] = -0.0311279296875

```



Tensor type: f32
model.post_norm_w[0] = 1.289062
model.post_norm_w[1] = 1.304688
model.post_norm_w[2] = 1.296875
model.post_norm_w[3] = 1.250000
model.post_norm_w[4] = 1.218750
model.post_norm_w[5] = 1.250000
model.post_norm_w[6] = 1.296875
model.post_norm_w[7] = 1.250000
model.post_norm_w[8] = 1.289062
model.post_norm_w[9] = 1.335938


(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.post_ln.weight

Tensor Information:
Name: v.post_ln.weight
Shape: 1280
Type: F32
Total elements: 1280

First 10 values:
[0] = 1.2890625
[1] = 1.3046875
[2] = 1.296875
[3] = 1.25
[4] = 1.21875
[5] = 1.25
[6] = 1.296875
[7] = 1.25
[8] = 1.2890625
[9] = 1.3359375

(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-622429e8d31810962dd984bc98559e706db2fb1d40e99cb073beb7148d909d73 v.pre_ln.weight

Tensor Information:
Name: v.pre_ln.weight
Shape: 1280
Type: F32
Total elements: 1280

First 10 values:
[0] = 0.005157470703125
[1] = -0.0086669921875
[2] = 0.984375
[3] = 0.00031280517578125
[4] = -0.00139617919921875
[5] = 0.006195068359375
[6] = 0.427734375
[7] = 0.00153350830078125
[8] = 1.265625
[9] = -0.004974365234375


31457280 |  7680,  4096,     1,     1 | F32     | v.enc.mmproj.weight
    4096 |  4096,     1,     1,     1 | F32     | v.enc.mmproj.bias

31457280 |  7680,  4096,     1,     1 | F16     | mm.0.weight
    4096 |  4096,     1,     1,     1 | F32     | mm.0.bias



### Tokenizer
```console
(venv) $ ./inspect-model.sh /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9
INFO:gguf-dump:* Loading: /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 29 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 396
      3: UINT64     |        1 | GGUF.kv_count = 26
      4: STRING     |        1 | general.architecture = 'mllama'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Model'
      7: STRING     |        1 | general.size_label = '10B'
      8: UINT32     |        1 | mllama.block_count = 40
      9: UINT32     |        1 | mllama.context_length = 131072
     10: UINT32     |        1 | mllama.embedding_length = 4096
     11: UINT32     |        1 | mllama.feed_forward_length = 14336
     12: UINT32     |        1 | mllama.attention.head_count = 32
     13: UINT32     |        1 | mllama.attention.head_count_kv = 8
     14: FLOAT32    |        1 | mllama.rope.freq_base = 500000.0
     15: FLOAT32    |        1 | mllama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     16: UINT32     |        1 | general.file_type = 15
     17: UINT32     |        1 | mllama.vocab_size = 128256
     18: UINT32     |        1 | mllama.rope.dimension_count = 128
     19: [INT32]    |        8 | mllama.attention.cross_attention_layers
     20: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     21: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     22: [STRING]   |   128257 | tokenizer.ggml.tokens
     23: [INT32]    |   128257 | tokenizer.ggml.token_type
     24: [STRING]   |   280147 | tokenizer.ggml.merges
     25: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     26: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     27: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     28: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     29: UINT32     |        1 | general.quantization_version = 2

```
```console




(venv) $ ./inspect-model.sh models/llama-3-2-11b.gguf
INFO:gguf-dump:* Loading: models/llama-3-2-11b.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 56 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 907
      3: UINT64     |        1 | GGUF.kv_count = 53
      4: STRING     |        1 | general.architecture = 'mllama'
      5: STRING     |        1 | general.type = 'model'
      6: STRING     |        1 | general.name = 'Llama 3.2 11B Vision Instruct'
      7: STRING     |        1 | general.finetune = 'Vision-Instruct'
      8: STRING     |        1 | general.basename = 'Llama-3.2'
      9: STRING     |        1 | general.size_label = '11B'
     10: STRING     |        1 | general.license = 'llama3.2'
     11: [STRING]   |        6 | general.tags
     12: [STRING]   |        8 | general.languages
     13: UINT32     |        1 | mllama.image_token_index = 128256
     14: UINT32     |        1 | mllama.context_length = 131072
     15: UINT32     |        1 | mllama.block_count = 40
     16: UINT32     |        1 | mllama.embedding_length = 4096
     17: UINT32     |        1 | mllama.feed_forward_length = 14336
     18: UINT32     |        1 | mllama.attention.head_count = 32
     19: UINT32     |        1 | mllama.attention.head_count_kv = 8
     20: FLOAT32    |        1 | mllama.rope.freq_base = 500000.0
     21: FLOAT32    |        1 | mllama.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     22: UINT32     |        1 | general.file_type = 1
     23: [INT32]    |        8 | mllama.cross_attention_layers
     24: UINT32     |        1 | mllama.vocab_size = 128257
     25: UINT32     |        1 | mllama.rope.dimension_count = 128
     26: STRING     |        1 | vision.type = 'cross-vit'
     27: STRING     |        1 | vision.architecture = 'mllama_vision_model'
     28: UINT32     |        1 | vision.image_size = 560
     29: UINT32     |        1 | vision.block_count = 32
     30: FLOAT32    |        1 | vision.attention.layer_norm_rms_epsilon = 9.999999747378752e-06
     31: UINT32     |        1 | vision.embedding_length = 1280
     32: STRING     |        1 | vision.cross.mllama.activation_function = 'gelu'
     33: UINT32     |        1 | vision.feed_forward_length = 5120
     34: UINT32     |        1 | vision.cross.mllama.global_block_count = 8
     35: UINT32     |        1 | vision.cross.mllama.max_num_tiles = 4
     36: UINT32     |        1 | vision.cross.mllama.channels_count = 3
     37: UINT32     |        1 | vision.patch_size = 14
     38: [INT32]    |        5 | vision.cross.mllama.intermediate_layers_indices
     39: UINT32     |        1 | vision.attention.head_count = 16
     40: UINT32     |        1 | vision.cross.mllama.output_dim = 7680
     41: STRING     |        1 | vision.cross.mllama.model_type = 'mllama_vision_model'
     42: UINT32     |        1 | vision.clip.max_position_embeddings = 1601
     43: [INT32]    |       16 | vision.cross.mllama.supported_aspect_ratios
     44: [FLOAT32]  |        3 | vision.image_mean
     45: [FLOAT32]  |        3 | vision.image_std
     46: UINT32     |        1 | vision.clip.projection_dim = 7680
     47: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     48: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     49: [STRING]   |   128257 | tokenizer.ggml.tokens
     50: [INT32]    |   128257 | tokenizer.ggml.token_type
     51: [STRING]   |   280147 | tokenizer.ggml.merges
     52: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     53: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     54: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     55: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     56: UINT32     |        1 | general.quantization_version = 2
 ```

### Vocab
llama.cpp:
```console
  47: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     48: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     49: [STRING]   |   128257 | tokenizer.ggml.tokens
     50: [INT32]    |   128257 | tokenizer.ggml.token_type
     51: [STRING]   |   280147 | tokenizer.ggml.merges
     52: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     53: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     54: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     55: UINT32     |        1 | tokenizer.ggml.eot_token_id = 128000
     56: UINT32     |        1 | tokenizer.ggml.start_header_token_id = 128006
     57: UINT32     |        1 | tokenizer.ggml.end_header_token_id = 128007
     58: UINT32     |        1 | tokenizer.ggml.eom_token_id = 128008
     59: UINT32     |        1 | tokenizer.ggml.python_tag_token_id = 128010
     60: UINT32     |        1 | tokenizer.ggml.image_token_id = 128256
     61: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     62: UINT32     |        1 | general.quantization_version = 2
 ```
 ollama:
 ```console
     20: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     21: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     22: [STRING]   |   128257 | tokenizer.ggml.tokens
     23: [INT32]    |   128257 | tokenizer.ggml.token_type
     24: [STRING]   |   280147 | tokenizer.ggml.merges
     25: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     26: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     27: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     28: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     29: UINT32     |        1 | general.quantization_version = 2


```
Now, if I run the ollama language model with the same program (no image) I get:
```console
prompt = <|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the Eiffel Tower?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


token = 128000
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
[New Thread 0x7fffabe00000 (LWP 1840205)]
[New Thread 0x7fffab400000 (LWP 1840206)]
[New Thread 0x7fffaaa00000 (LWP 1840207)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 18 1 1]
The Eiffel Tower is an iconic iron lattice tower located in Paris,ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 France. It was built for the 1889 World's Fair, held to celebrate
main: decoded 32 tokens in 13.38 s, speed: 2.39 t/s


llama_perf_context_print:        load time =   14388.28 ms
llama_perf_context_print: prompt eval time =    6141.74 ms /    18 tokens (  341.21 ms per token,     2.93 tokens per second)
llama_perf_context_print:        eval time =   12827.71 ms /    31 runs   (  413.80 ms per token,     2.42 tokens per second)
llama_perf_context_print:       total time =   27771.62 ms /    49 tokens
[Thread 0x7fffc5e00000 (LWP 1840196) exited]
[Thread 0x7fffaaa00000 (LWP 1840207) exited]
[Thread 0x7fffab400000 (LWP 1840206) exited]
[Thread 0x7fffabe00000 (LWP 1840205) exited]
[Thread 0x7fffc5400000 (LWP 1840197) exited]
[Thread 0x7fffc7a00000 (LWP 1840189) exited]
[Thread 0x7ffff339b000 (LWP 1840184) exited]
[Thread 0x7fffc4a00000 (LWP 1840198) exited]
[New process 1840184]
[Inferior 1 (process 1840184) exited normally]
```
But when I run this with my converted model I don't get as good of an answer
as this.
```console
prompt = <|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the Eiffel Tower?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


token = 128000
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
[New Thread 0x7fffabe00000 (LWP 1843202)]
[New Thread 0x7fffab400000 (LWP 1843203)]
[New Thread 0x7fffaaa00000 (LWP 1843204)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 18 1 1]
I think you meant to type "Eiffel Tower" doesn't seemggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 to be a well-known landmark.

main: decoded 22 tokens in 19.26 s, speed: 1.14 t/s


llama_perf_context_print:        load time =   46051.65 ms
llama_perf_context_print: prompt eval time =   18818.83 ms /    18 tokens ( 1045.49 ms per token,     0.96 tokens per second)
llama_perf_context_print:        eval time =   19149.48 ms /    22 runs   (  870.43 ms per token,     1.15 tokens per second)
llama_perf_context_print:       total time =   65314.36 ms /    40 tokens
[Thread 0x7fffc5e00000 (LWP 1843158) exited]
[Thread 0x7fffaaa00000 (LWP 1843204) exited]
[Thread 0x7fffab400000 (LWP 1843203) exited]
[Thread 0x7fffabe00000 (LWP 1843202) exited]
[Thread 0x7fffc4a00000 (LWP 1843160) exited]
[Thread 0x7fffc5400000 (LWP 1843159) exited]
[Thread 0x7ffff339b000 (LWP 1843146) exited]
[Thread 0x7fffc7a00000 (LWP 1843149) exited]
[New process 1843146]
```
SO the tokens are the exact same. And the model graphs seems to work fine as
it can respond with a good response for the ollama model.
Could there be difference in the actual model.

```console
prompt = What is the Eiffel Tower?
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
[New Thread 0x7fffabe00000 (LWP 1846433)]
[New Thread 0x7fffab400000 (LWP 1846434)]
[New Thread 0x7fffaaa00000 (LWP 1846435)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 8 1 1]
**
 : 1035
The : 791
 E : 469
iff : 3168
el : 301
 Tower : 22703
 is : 374
 a : 264
  : 220
1 : 16
, : 11
000 : 931
-ton : 75735
ne : 818
 steel : 9699
 structure : 6070
 that : 430
 stands : 13656
  : 220
1 : 16
, : 11
000 : 931
 feet : 7693
 above : 3485
 the : 279
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 ground : 5015
 and : 323
 is : 374
 a : 264
  : 220
1 : 16
, : 11
```

```console
token = 128000
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
[New Thread 0x7fffabe00000 (LWP 1847036)]
[New Thread 0x7fffab400000 (LWP 1847037)]
[New Thread 0x7fffaaa00000 (LWP 1847038)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 18 1 1]
I : 40
 think : 1781
 you : 499
 meant : 8967
 to : 311
 type : 955
 " : 330
E : 36
iff : 3168
el : 301
 Tower : 22703
" : 1
 doesn : 3250
't : 956
 seem : 2873
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 0)
 to : 311
 be : 387
 a : 264
 well : 1664
-known : 22015
 landmark : 38350
. : 13
```

### Troubleshooting
I've tried using my converted model with only the language model and this did
not initially work when using template formatting. But it worked well without
and could give a good anwser to the question: "What is the Eiffel Tower?".

So I managed to find an issue with one of the weights which has the output 
norm which I had mistakenly set to be the same as the attention norm.


### Computation graph language model
We first have the token embeddings matrix which is what is used to get the
token embeddings from the token indices:
```c++
model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab + 7}, 0);
```
This from the tensor named `token_embd`
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf token_embd.weight

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: F32
Total elements: 525369344

First 10 values:
[0] = 0.001007080078125
[1] = 0.005584716796875
[2] = -0.0034027099609375
[3] = -0.0012359619140625
[4] = -0.003570556640625
[5] = 0.0006256103515625
[6] = -0.001495361328125
[7] = -0.002166748046875
[8] = -0.0036163330078125
[9] = -0.00433349609375
```
Ollama's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 token_embd.weight

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: Q4_K
Total elements: 525369344

First 10 values:
[0] = 111
[1] = 3
[2] = 37
[3] = 13
[4] = 161
[5] = 191
[6] = 171
[7] = 161
[8] = 229
[9] = 250
```


Notice that this is larger then the language vocabulary by 8 and I think this
is because there are 8 special tokens that are added to the vocabulary.

How this done is by `llm_build_inp_embd` which uses ggml_get_rows:
```c++
        lctx.inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch.n_tokens);
        cb(lctx.inp_tokens, "inp_tokens", -1);
        ggml_set_input(lctx.inp_tokens);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);
```

The first thing in a layer is the attention normalization:
```c++
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
```
```c++
layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
```
This is from the tensor named `attn_norm.weight`
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.0.attn_norm.weight

Tensor Information:
Name: blk.0.attn_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 0.047119140625
[1] = 0.1875
[2] = 0.41796875
[3] = 0.01708984375
[4] = 0.43359375
[5] = 0.021484375
[6] = -0.00020599365234375
[7] = 0.004547119140625
[8] = 0.0341796875
[9] = 0.024658203125
```
Ollam's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.0.attn_norm.weight

Tensor Information:
Name: blk.0.attn_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 0.047119140625
[1] = 0.1875
[2] = 0.41796875
[3] = 0.01708984375
[4] = 0.43359375
[5] = 0.021484375
[6] = -0.00020599365234375
[7] = 0.004547119140625
[8] = 0.0341796875
[9] = 0.024658203125
```

After this we have a condition for if the layer we are iterating over is one
of the cross attention layers:
```console
  23: [INT32]    |        8 | mllama.cross_attention_layers
```
These are layers 3, 8, 13, 18, 23, 28, 33, 38.

After the attention computation we then have:
```c++
      // feed-forward network
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);
```
And this is using the tensor named `blk.%d.post_attention_norm`:
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.0.post_attention_norm.weight

Tensor Information:
Name: blk.0.post_attention_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 0.134765625
[1] = 0.125
[2] = 0.1376953125
[3] = 0.1357421875
[4] = 0.1259765625
[5] = 0.134765625
[6] = 0.134765625
[7] = 0.134765625
[8] = 0.134765625
[9] = 0.134765625
```
Ollama's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.0.ffn_norm.weight

Tensor Information:
Name: blk.0.ffn_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 0.134765625
[1] = 0.125
[2] = 0.1376953125
[3] = 0.1357421875
[4] = 0.1259765625
[5] = 0.134765625
[6] = 0.134765625
[7] = 0.134765625
[8] = 0.134765625
[9] = 0.134765625
```

After the layers we then have:
```c++
        cur = llm_build_norm(ctx0, cur, hparams,
                model.output_norm, NULL,
                LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);
```
This is using the tensor named `output_norm.weight`:
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf output_norm.weight

Tensor Information:
Name: output_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 2.46875
[1] = 2.390625
[2] = 2.53125
[3] = 2.421875
[4] = 2.390625
[5] = 2.46875
[6] = 2.265625
[7] = 2.4375
[8] = 2.296875
[9] = 2.328125
```
Ollama's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 output_norm.weight

Tensor Information:
Name: output_norm.weight
Shape: 4096
Type: F32
Total elements: 4096

First 10 values:
[0] = 2.46875
[1] = 2.390625
[2] = 2.53125
[3] = 2.421875
[4] = 2.390625
[5] = 2.46875
[6] = 2.265625
[7] = 2.4375
[8] = 2.296875
[9] = 2.328125

```

And this normalized tensor is then multiplied by the tensor named `output.weight`:
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf output.weight

Tensor Information:
Name: output.weight
Shape: 4096 x 128256
Type: F32
Total elements: 525336576

First 10 values:
[0] = 0.0081787109375
[1] = 0.007171630859375
[2] = 0.012451171875
[3] = 0.023681640625
[4] = -0.017578125
[5] = 0.01275634765625
[6] = -0.02001953125
[7] = -0.005279541015625
[8] = -0.0015411376953125
[9] = 0.01556396484375
```
Ollama's:
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 output.weight

Tensor Information:
Name: output.weight
Shape: 4096 x 128256
Type: Q6_K
Total elements: 525336576

First 10 values:
[0] = 5
[1] = 182
[2] = 191
[3] = 160
[4] = 184
[5] = 159
[6] = 27
[7] = 247
[8] = 98
[9] = 11
```

The espilon value are in ollama:
```console
9.999999747378752e-06
9.999999747378752e-06
```
And in llama.cpp:
```console
9.999999747378752e-06
```

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.0.ffn_gate.weight

Tensor Information:
Name: blk.0.ffn_gate.weight
Shape: 4096 x 14336
Type: F16
Total elements: 58720256

First 10 values:
[0] = -0.0166015625
[1] = -0.0062255859375
[2] = -0.0013885498046875
[3] = -0.000461578369140625
[4] = 0.007293701171875
[5] = 0.0038604736328125
[6] = -0.0037994384765625
[7] = -0.0274658203125
[8] = -0.021728515625
[9] = 0.00131988525390625
```

```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.0.ffn_gate.weight

Tensor Information:
Name: blk.0.ffn_gate.weight
Shape: 4096 x 14336
Type: Q4_K
Total elements: 58720256

First 10 values:
[0] = 232
[1] = 3
[2] = 39
[3] = 15
[4] = 246
[5] = 247
[6] = 251
[7] = 248
[8] = 255
[9] = 248
```

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.0.attn_output.weight

Tensor Information:
Name: blk.0.attn_output.weight
Shape: 4096 x 4096
Type: F16
Total elements: 16777216

First 10 values:
[0] = 0.00592041015625
[1] = -0.001983642578125
[2] = -0.0101318359375
[3] = -0.00110626220703125
[4] = 0.003387451171875
[5] = -0.00994873046875
[6] = -0.00848388671875
[7] = -0.000186920166015625
[8] = -0.00015735626220703125
[9] = -0.00130462646484375
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.0.attn_output.weight

Tensor Information:
Name: blk.0.attn_output.weight
Shape: 4096 x 4096
Type: Q4_K
Total elements: 16777216

First 10 values:
[0] = 16
[1] = 3
[2] = 48
[3] = 13
[4] = 230
[5] = 227
[6] = 229
[7] = 229
[8] = 225
[9] = 229
```

```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_mlp_gate

Tensor Information:
Name: blk.3.cross_attn_mlp_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 0.006256103515625

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_ffn_gate

Tensor Information:
Name: blk.3.cross_attn_ffn_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 0.006256103515625
```


```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_gate

Tensor Information:
Name: blk.3.cross_attn_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 0.000545501708984375
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_attn_gate

Tensor Information:
Name: blk.3.cross_attn_attn_gate
Shape: 1
Type: F32
Total elements: 1

First 10 values:
[0] = 0.000545501708984375
```

```
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_q.weight

Tensor Information:
Name: blk.3.cross_attn_q.weight
Shape: 4096 x 4096
Type: F16
Total elements: 16777216

First 10 values:
[0] = -0.01263427734375
[1] = 0.0615234375
[2] = 0.02392578125
[3] = -0.0023193359375
[4] = -0.004852294921875
[5] = 0.017333984375
[6] = 0.0576171875
[7] = 0.000858306884765625
[8] = 0.0198974609375
[9] = -0.033203125
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_q_proj.weight

Tensor Information:
Name: blk.3.cross_attn_q_proj.weight
Shape: 4096 x 4096
Type: Q4_K
Total elements: 16777216

First 10 values:
[0] = 69
[1] = 10
[2] = 1
[3] = 22
[4] = 163
[5] = 95
[6] = 175
[7] = 191
[8] = 215
[9] = 93
```
```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_k_norm.weight

Tensor Information:
Name: blk.3.cross_attn_k_norm.weight
Shape: 128
Type: F32
Total elements: 128

First 10 values:
[0] = 1.3359375
[1] = 1.359375
[2] = 1.28125
[3] = 1.2890625
[4] = 1.2578125
[5] = 1.2890625
[6] = 1.3125
[7] = 1.296875
[8] = 1.3359375
[9] = 1.1484375
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_k_norm.weight

Tensor Information:
Name: blk.3.cross_attn_k_norm.weight
Shape: 128
Type: F32
Total elements: 128

First 10 values:
[0] = 1.3359375
[1] = 1.359375
[2] = 1.28125
[3] = 1.2890625
[4] = 1.2578125
[5] = 1.2890625
[6] = 1.3125
[7] = 1.296875
[8] = 1.3359375
[9] = 1.1484375
```

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf blk.3.cross_attn_q_norm.weight

Tensor Information:
Name: blk.3.cross_attn_q_norm.weight
Shape: 128
Type: F32
Total elements: 128

First 10 values:
[0] = 1.3359375
[1] = 1.359375
[2] = 1.28125
[3] = 1.2890625
[4] = 1.2578125
[5] = 1.2890625
[6] = 1.3125
[7] = 1.296875
[8] = 1.3359375
[9] = 1.1484375
```
```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 blk.3.cross_attn_q_norm.weight

Tensor Information:
Name: blk.3.cross_attn_q_norm.weight
Shape: 128
Type: F32
Total elements: 128

First 10 values:
[0] = 1.3359375
[1] = 1.359375
[2] = 1.28125
[3] = 1.2890625
[4] = 1.2578125
[5] = 1.2890625
[6] = 1.3125
[7] = 1.296875
[8] = 1.3359375
[9] = 1.1484375
```

```console
(venv) $ ./read-tensor.py models/llama-3-2-11b.gguf rope_freqs.weight

Tensor Information:
Name: rope_freqs.weight
Shape: 64
Type: F32
Total elements: 64

First 10 values:
[0] = 1.0
[1] = 1.0
[2] = 1.0
[3] = 1.0
[4] = 1.0
[5] = 1.0
[6] = 1.0
[7] = 1.0
[8] = 1.0
[9] = 1.0

```

```console
(venv) $ ./read-tensor.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9 rope_freqs.weight

Tensor Information:
Name: rope_freqs.weight
Shape: 64
Type: F32
Total elements: 64

First 10 values:
[0] = 1.0
[1] = 1.0
[2] = 1.0
[3] = 1.0
[4] = 1.0
[5] = 1.0
[6] = 1.0
[7] = 1.0
[8] = 1.0
[9] = 1.0
```

Now if I run the llama.cpp code but use the ollama model I get the following
for a non-image prompt:
```console
llama_new_context_with_model: graph splits = 225 (with bs=512), 3 (with bs=1)
prompt = <|start_header_id|>user<|end_header_id|>

What is the Eiffel Tower?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 16.01 MiB to 426.13 MiB
[New Thread 0x7fffabe00000 (LWP 110381)]
[New Thread 0x7fffab400000 (LWP 110382)]
[New Thread 0x7fffaaa00000 (LWP 110383)]
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [l_out-24] [4096 17 1 1]
ggml_backend_cuda_graph_compute: CUDA graph update failed
The : 791
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 E : 469
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
iff : 3168
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
el : 301
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 Tower : 22703
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 is : 374
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 a : 264
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
 famous : 11495
```

With my version of llama.cpp:
```console
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
```
So the tokens generated for the input are identical. This will then be used
to lookup the embeddings from the token embeddings matrix.

And it seems to try to predict something from an image even if there is not
image tag nore is there a question about an image.

So the this is using the exact same graph, but the model is not the same. I've
inspected the weights above and they seem to match up, at least the ones that
are not quentized.

Could it be that one of the weights have been mixed up in the case of llama.cpp
and they are not used in the correct place. Perhaps I can change

llama.cpp startup:
```console
llm_load_print_meta: general.name     = Llama 3.2 11B Vision Instruct New
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
```


ollama startup:
```console
llm_load_vocab: control token: 128256 '<|image|>' is not marked as EOG
llm_load_vocab: control token: 128255 '<|reserved_special_token_246|>' is not marked as EOG
llm_load_vocab: control token: 128250 '<|reserved_special_token_241|>' is not marked as EOG
llm_load_vocab: control token: 128247 '<|reserved_special_token_238|>' is not marked as EOG
llm_load_vocab: control token: 128244 '<|reserved_special_token_235|>' is not marked as EOG
llm_load_vocab: control token: 128243 '<|reserved_special_token_234|>' is not marked as EOG
llm_load_vocab: control token: 128242 '<|reserved_special_token_233|>' is not marked as EOG
llm_load_vocab: control token: 128241 '<|reserved_special_token_232|>' is not marked as EOG
llm_load_vocab: control token: 128236 '<|reserved_special_token_227|>' is not marked as EOG
llm_load_vocab: control token: 128232 '<|reserved_special_token_223|>' is not marked as EOG
llm_load_vocab: control token: 128231 '<|reserved_special_token_222|>' is not marked as EOG
llm_load_vocab: control token: 128229 '<|reserved_special_token_220|>' is not marked as EOG
llm_load_vocab: control token: 128226 '<|reserved_special_token_217|>' is not marked as EOG
llm_load_vocab: control token: 128219 '<|reserved_special_token_210|>' is not marked as EOG
llm_load_vocab: control token: 128215 '<|reserved_special_token_206|>' is not marked as EOG
llm_load_vocab: control token: 128214 '<|reserved_special_token_205|>' is not marked as EOG
llm_load_vocab: control token: 128208 '<|reserved_special_token_199|>' is not marked as EOG
llm_load_vocab: control token: 128207 '<|reserved_special_token_198|>' is not marked as EOG
llm_load_vocab: control token: 128205 '<|reserved_special_token_196|>' is not marked as EOG
llm_load_vocab: control token: 128201 '<|reserved_special_token_192|>' is not marked as EOG
llm_load_vocab: control token: 128200 '<|reserved_special_token_191|>' is not marked as EOG
llm_load_vocab: control token: 128199 '<|reserved_special_token_190|>' is not marked as EOG
llm_load_vocab: control token: 128197 '<|reserved_special_token_188|>' is not marked as EOG
llm_load_vocab: control token: 128195 '<|reserved_special_token_186|>' is not marked as EOG
llm_load_vocab: control token: 128194 '<|reserved_special_token_185|>' is not marked as EOG
llm_load_vocab: control token: 128189 '<|reserved_special_token_180|>' is not marked as EOG
llm_load_vocab: control token: 128188 '<|reserved_special_token_179|>' is not marked as EOG
llm_load_vocab: control token: 128186 '<|reserved_special_token_177|>' is not marked as EOG
llm_load_vocab: control token: 128185 '<|reserved_special_token_176|>' is not marked as EOG
llm_load_vocab: control token: 128181 '<|reserved_special_token_172|>' is not marked as EOG
llm_load_vocab: control token: 128180 '<|reserved_special_token_171|>' is not marked as EOG
llm_load_vocab: control token: 128179 '<|reserved_special_token_170|>' is not marked as EOG
llm_load_vocab: control token: 128178 '<|reserved_special_token_169|>' is not marked as EOG
llm_load_vocab: control token: 128177 '<|reserved_special_token_168|>' is not marked as EOG
llm_load_vocab: control token: 128176 '<|reserved_special_token_167|>' is not marked as EOG
llm_load_vocab: control token: 128172 '<|reserved_special_token_163|>' is not marked as EOG
llm_load_vocab: control token: 128171 '<|reserved_special_token_162|>' is not marked as EOG
llm_load_vocab: control token: 128170 '<|reserved_special_token_161|>' is not marked as EOG
llm_load_vocab: control token: 128169 '<|reserved_special_token_160|>' is not marked as EOG
llm_load_vocab: control token: 128166 '<|reserved_special_token_157|>' is not marked as EOG
llm_load_vocab: control token: 128163 '<|reserved_special_token_154|>' is not marked as EOG
llm_load_vocab: control token: 128159 '<|reserved_special_token_150|>' is not marked as EOG
llm_load_vocab: control token: 128157 '<|reserved_special_token_148|>' is not marked as EOG
llm_load_vocab: control token: 128156 '<|reserved_special_token_147|>' is not marked as EOG
llm_load_vocab: control token: 128155 '<|reserved_special_token_146|>' is not marked as EOG
llm_load_vocab: control token: 128152 '<|reserved_special_token_143|>' is not marked as EOG
llm_load_vocab: control token: 128150 '<|reserved_special_token_141|>' is not marked as EOG
llm_load_vocab: control token: 128148 '<|reserved_special_token_139|>' is not marked as EOG
llm_load_vocab: control token: 128147 '<|reserved_special_token_138|>' is not marked as EOG
llm_load_vocab: control token: 128145 '<|reserved_special_token_136|>' is not marked as EOG
llm_load_vocab: control token: 128143 '<|reserved_special_token_134|>' is not marked as EOG
llm_load_vocab: control token: 128142 '<|reserved_special_token_133|>' is not marked as EOG
llm_load_vocab: control token: 128139 '<|reserved_special_token_130|>' is not marked as EOG
llm_load_vocab: control token: 128137 '<|reserved_special_token_128|>' is not marked as EOG
llm_load_vocab: control token: 128136 '<|reserved_special_token_127|>' is not marked as EOG
llm_load_vocab: control token: 128135 '<|reserved_special_token_126|>' is not marked as EOG
llm_load_vocab: control token: 128134 '<|reserved_special_token_125|>' is not marked as EOG
llm_load_vocab: control token: 128132 '<|reserved_special_token_123|>' is not marked as EOG
llm_load_vocab: control token: 128129 '<|reserved_special_token_120|>' is not marked as EOG
llm_load_vocab: control token: 128125 '<|reserved_special_token_116|>' is not marked as EOG
llm_load_vocab: control token: 128124 '<|reserved_special_token_115|>' is not marked as EOG
llm_load_vocab: control token: 128123 '<|reserved_special_token_114|>' is not marked as EOG
llm_load_vocab: control token: 128120 '<|reserved_special_token_111|>' is not marked as EOG
llm_load_vocab: control token: 128116 '<|reserved_special_token_107|>' is not marked as EOG
llm_load_vocab: control token: 128113 '<|reserved_special_token_104|>' is not marked as EOG
llm_load_vocab: control token: 128111 '<|reserved_special_token_102|>' is not marked as EOG
llm_load_vocab: control token: 128110 '<|reserved_special_token_101|>' is not marked as EOG
llm_load_vocab: control token: 128109 '<|reserved_special_token_100|>' is not marked as EOG
llm_load_vocab: control token: 128107 '<|reserved_special_token_98|>' is not marked as EOG
llm_load_vocab: control token: 128104 '<|reserved_special_token_95|>' is not marked as EOG
llm_load_vocab: control token: 128103 '<|reserved_special_token_94|>' is not marked as EOG
llm_load_vocab: control token: 128102 '<|reserved_special_token_93|>' is not marked as EOG
llm_load_vocab: control token: 128098 '<|reserved_special_token_89|>' is not marked as EOG
llm_load_vocab: control token: 128092 '<|reserved_special_token_83|>' is not marked as EOG
llm_load_vocab: control token: 128091 '<|reserved_special_token_82|>' is not marked as EOG
llm_load_vocab: control token: 128090 '<|reserved_special_token_81|>' is not marked as EOG
llm_load_vocab: control token: 128088 '<|reserved_special_token_79|>' is not marked as EOG
llm_load_vocab: control token: 128086 '<|reserved_special_token_77|>' is not marked as EOG
llm_load_vocab: control token: 128082 '<|reserved_special_token_73|>' is not marked as EOG
llm_load_vocab: control token: 128079 '<|reserved_special_token_70|>' is not marked as EOG
llm_load_vocab: control token: 128077 '<|reserved_special_token_68|>' is not marked as EOG
llm_load_vocab: control token: 128076 '<|reserved_special_token_67|>' is not marked as EOG
llm_load_vocab: control token: 128074 '<|reserved_special_token_65|>' is not marked as EOG
llm_load_vocab: control token: 128069 '<|reserved_special_token_60|>' is not marked as EOG
llm_load_vocab: control token: 128068 '<|reserved_special_token_59|>' is not marked as EOG
llm_load_vocab: control token: 128066 '<|reserved_special_token_57|>' is not marked as EOG
llm_load_vocab: control token: 128064 '<|reserved_special_token_55|>' is not marked as EOG
llm_load_vocab: control token: 128063 '<|reserved_special_token_54|>' is not marked as EOG
llm_load_vocab: control token: 128061 '<|reserved_special_token_52|>' is not marked as EOG
llm_load_vocab: control token: 128060 '<|reserved_special_token_51|>' is not marked as EOG
llm_load_vocab: control token: 128058 '<|reserved_special_token_49|>' is not marked as EOG
llm_load_vocab: control token: 128055 '<|reserved_special_token_46|>' is not marked as EOG
llm_load_vocab: control token: 128047 '<|reserved_special_token_38|>' is not marked as EOG
llm_load_vocab: control token: 128046 '<|reserved_special_token_37|>' is not marked as EOG
llm_load_vocab: control token: 128045 '<|reserved_special_token_36|>' is not marked as EOG
llm_load_vocab: control token: 128044 '<|reserved_special_token_35|>' is not marked as EOG
llm_load_vocab: control token: 128039 '<|reserved_special_token_30|>' is not marked as EOG
llm_load_vocab: control token: 128037 '<|reserved_special_token_28|>' is not marked as EOG
llm_load_vocab: control token: 128036 '<|reserved_special_token_27|>' is not marked as EOG
llm_load_vocab: control token: 128033 '<|reserved_special_token_24|>' is not marked as EOG
llm_load_vocab: control token: 128029 '<|reserved_special_token_20|>' is not marked as EOG
llm_load_vocab: control token: 128028 '<|reserved_special_token_19|>' is not marked as EOG
llm_load_vocab: control token: 128025 '<|reserved_special_token_16|>' is not marked as EOG
llm_load_vocab: control token: 128024 '<|reserved_special_token_15|>' is not marked as EOG
llm_load_vocab: control token: 128023 '<|reserved_special_token_14|>' is not marked as EOG
llm_load_vocab: control token: 128022 '<|reserved_special_token_13|>' is not marked as EOG
llm_load_vocab: control token: 128019 '<|reserved_special_token_10|>' is not marked as EOG
llm_load_vocab: control token: 128017 '<|reserved_special_token_8|>' is not marked as EOG
llm_load_vocab: control token: 128016 '<|reserved_special_token_7|>' is not marked as EOG
llm_load_vocab: control token: 128014 '<|reserved_special_token_5|>' is not marked as EOG
llm_load_vocab: control token: 128012 '<|reserved_special_token_3|>' is not marked as EOG
llm_load_vocab: control token: 128011 '<|reserved_special_token_2|>' is not marked as EOG
llm_load_vocab: control token: 128004 '<|finetune_right_pad_id|>' is not marked as EOG
llm_load_vocab: control token: 128002 '<|reserved_special_token_0|>' is not marked as EOG
llm_load_vocab: control token: 128253 '<|reserved_special_token_244|>' is not marked as EOG
llm_load_vocab: control token: 128191 '<|reserved_special_token_182|>' is not marked as EOG
llm_load_vocab: control token: 128184 '<|reserved_special_token_175|>' is not marked as EOG
llm_load_vocab: control token: 128138 '<|reserved_special_token_129|>' is not marked as EOG
llm_load_vocab: control token: 128183 '<|reserved_special_token_174|>' is not marked as EOG
llm_load_vocab: control token: 128041 '<|reserved_special_token_32|>' is not marked as EOG
llm_load_vocab: control token: 128049 '<|reserved_special_token_40|>' is not marked as EOG
llm_load_vocab: control token: 128093 '<|reserved_special_token_84|>' is not marked as EOG
llm_load_vocab: control token: 128216 '<|reserved_special_token_207|>' is not marked as EOG
llm_load_vocab: control token: 128108 '<|reserved_special_token_99|>' is not marked as EOG
llm_load_vocab: control token: 128209 '<|reserved_special_token_200|>' is not marked as EOG
llm_load_vocab: control token: 128146 '<|reserved_special_token_137|>' is not marked as EOG
llm_load_vocab: control token: 128032 '<|reserved_special_token_23|>' is not marked as EOG
llm_load_vocab: control token: 128130 '<|reserved_special_token_121|>' is not marked as EOG
llm_load_vocab: control token: 128202 '<|reserved_special_token_193|>' is not marked as EOG
llm_load_vocab: control token: 128075 '<|reserved_special_token_66|>' is not marked as EOG
llm_load_vocab: control token: 128096 '<|reserved_special_token_87|>' is not marked as EOG
llm_load_vocab: control token: 128187 '<|reserved_special_token_178|>' is not marked as EOG
llm_load_vocab: control token: 128144 '<|reserved_special_token_135|>' is not marked as EOG
llm_load_vocab: control token: 128230 '<|reserved_special_token_221|>' is not marked as EOG
llm_load_vocab: control token: 128007 '<|end_header_id|>' is not marked as EOG
llm_load_vocab: control token: 128056 '<|reserved_special_token_47|>' is not marked as EOG
llm_load_vocab: control token: 128057 '<|reserved_special_token_48|>' is not marked as EOG
llm_load_vocab: control token: 128062 '<|reserved_special_token_53|>' is not marked as EOG
llm_load_vocab: control token: 128154 '<|reserved_special_token_145|>' is not marked as EOG
llm_load_vocab: control token: 128153 '<|reserved_special_token_144|>' is not marked as EOG
llm_load_vocab: control token: 128213 '<|reserved_special_token_204|>' is not marked as EOG
llm_load_vocab: control token: 128173 '<|reserved_special_token_164|>' is not marked as EOG
llm_load_vocab: control token: 128161 '<|reserved_special_token_152|>' is not marked as EOG
llm_load_vocab: control token: 128042 '<|reserved_special_token_33|>' is not marked as EOG
llm_load_vocab: control token: 128182 '<|reserved_special_token_173|>' is not marked as EOG
llm_load_vocab: control token: 128095 '<|reserved_special_token_86|>' is not marked as EOG
llm_load_vocab: control token: 128119 '<|reserved_special_token_110|>' is not marked as EOG
llm_load_vocab: control token: 128237 '<|reserved_special_token_228|>' is not marked as EOG
llm_load_vocab: control token: 128149 '<|reserved_special_token_140|>' is not marked as EOG
llm_load_vocab: control token: 128043 '<|reserved_special_token_34|>' is not marked as EOG
llm_load_vocab: control token: 128140 '<|reserved_special_token_131|>' is not marked as EOG
llm_load_vocab: control token: 128174 '<|reserved_special_token_165|>' is not marked as EOG
llm_load_vocab: control token: 128240 '<|reserved_special_token_231|>' is not marked as EOG
llm_load_vocab: control token: 128158 '<|reserved_special_token_149|>' is not marked as EOG
llm_load_vocab: control token: 128053 '<|reserved_special_token_44|>' is not marked as EOG
llm_load_vocab: control token: 128027 '<|reserved_special_token_18|>' is not marked as EOG
llm_load_vocab: control token: 128003 '<|reserved_special_token_1|>' is not marked as EOG
llm_load_vocab: control token: 128020 '<|reserved_special_token_11|>' is not marked as EOG
llm_load_vocab: control token: 128117 '<|reserved_special_token_108|>' is not marked as EOG
llm_load_vocab: control token: 128162 '<|reserved_special_token_153|>' is not marked as EOG
llm_load_vocab: control token: 128227 '<|reserved_special_token_218|>' is not marked as EOG
llm_load_vocab: control token: 128160 '<|reserved_special_token_151|>' is not marked as EOG
llm_load_vocab: control token: 128013 '<|reserved_special_token_4|>' is not marked as EOG
llm_load_vocab: control token: 128089 '<|reserved_special_token_80|>' is not marked as EOG
llm_load_vocab: control token: 128164 '<|reserved_special_token_155|>' is not marked as EOG
llm_load_vocab: control token: 128001 '<|end_of_text|>' is not marked as EOG
llm_load_vocab: control token: 128114 '<|reserved_special_token_105|>' is not marked as EOG
llm_load_vocab: control token: 128251 '<|reserved_special_token_242|>' is not marked as EOG
llm_load_vocab: control token: 128126 '<|reserved_special_token_117|>' is not marked as EOG
llm_load_vocab: control token: 128054 '<|reserved_special_token_45|>' is not marked as EOG
llm_load_vocab: control token: 128225 '<|reserved_special_token_216|>' is not marked as EOG
llm_load_vocab: control token: 128248 '<|reserved_special_token_239|>' is not marked as EOG
llm_load_vocab: control token: 128252 '<|reserved_special_token_243|>' is not marked as EOG
llm_load_vocab: control token: 128217 '<|reserved_special_token_208|>' is not marked as EOG
llm_load_vocab: control token: 128006 '<|start_header_id|>' is not marked as EOG
llm_load_vocab: control token: 128212 '<|reserved_special_token_203|>' is not marked as EOG
llm_load_vocab: control token: 128078 '<|reserved_special_token_69|>' is not marked as EOG
llm_load_vocab: control token: 128238 '<|reserved_special_token_229|>' is not marked as EOG
llm_load_vocab: control token: 128087 '<|reserved_special_token_78|>' is not marked as EOG
llm_load_vocab: control token: 128228 '<|reserved_special_token_219|>' is not marked as EOG
llm_load_vocab: control token: 128059 '<|reserved_special_token_50|>' is not marked as EOG
llm_load_vocab: control token: 128101 '<|reserved_special_token_92|>' is not marked as EOG
llm_load_vocab: control token: 128210 '<|reserved_special_token_201|>' is not marked as EOG
llm_load_vocab: control token: 128085 '<|reserved_special_token_76|>' is not marked as EOG
llm_load_vocab: control token: 128072 '<|reserved_special_token_63|>' is not marked as EOG
llm_load_vocab: control token: 128071 '<|reserved_special_token_62|>' is not marked as EOG
llm_load_vocab: control token: 128050 '<|reserved_special_token_41|>' is not marked as EOG
llm_load_vocab: control token: 128198 '<|reserved_special_token_189|>' is not marked as EOG
llm_load_vocab: control token: 128073 '<|reserved_special_token_64|>' is not marked as EOG
llm_load_vocab: control token: 128000 '<|begin_of_text|>' is not marked as EOG
llm_load_vocab: control token: 128224 '<|reserved_special_token_215|>' is not marked as EOG
llm_load_vocab: control token: 128218 '<|reserved_special_token_209|>' is not marked as EOG
llm_load_vocab: control token: 128112 '<|reserved_special_token_103|>' is not marked as EOG
llm_load_vocab: control token: 128204 '<|reserved_special_token_195|>' is not marked as EOG
llm_load_vocab: control token: 128052 '<|reserved_special_token_43|>' is not marked as EOG
llm_load_vocab: control token: 128031 '<|reserved_special_token_22|>' is not marked as EOG
llm_load_vocab: control token: 128118 '<|reserved_special_token_109|>' is not marked as EOG
llm_load_vocab: control token: 128010 '<|python_tag|>' is not marked as EOG
llm_load_vocab: control token: 128239 '<|reserved_special_token_230|>' is not marked as EOG
llm_load_vocab: control token: 128203 '<|reserved_special_token_194|>' is not marked as EOG
llm_load_vocab: control token: 128133 '<|reserved_special_token_124|>' is not marked as EOG
llm_load_vocab: control token: 128249 '<|reserved_special_token_240|>' is not marked as EOG
llm_load_vocab: control token: 128168 '<|reserved_special_token_159|>' is not marked as EOG
llm_load_vocab: control token: 128128 '<|reserved_special_token_119|>' is not marked as EOG
llm_load_vocab: control token: 128106 '<|reserved_special_token_97|>' is not marked as EOG
llm_load_vocab: control token: 128040 '<|reserved_special_token_31|>' is not marked as EOG
llm_load_vocab: control token: 128233 '<|reserved_special_token_224|>' is not marked as EOG
llm_load_vocab: control token: 128167 '<|reserved_special_token_158|>' is not marked as EOG
llm_load_vocab: control token: 128131 '<|reserved_special_token_122|>' is not marked as EOG
llm_load_vocab: control token: 128115 '<|reserved_special_token_106|>' is not marked as EOG
llm_load_vocab: control token: 128235 '<|reserved_special_token_226|>' is not marked as EOG
llm_load_vocab: control token: 128192 '<|reserved_special_token_183|>' is not marked as EOG
llm_load_vocab: control token: 128065 '<|reserved_special_token_56|>' is not marked as EOG
llm_load_vocab: control token: 128141 '<|reserved_special_token_132|>' is not marked as EOG
llm_load_vocab: control token: 128097 '<|reserved_special_token_88|>' is not marked as EOG
llm_load_vocab: control token: 128099 '<|reserved_special_token_90|>' is not marked as EOG
llm_load_vocab: control token: 128193 '<|reserved_special_token_184|>' is not marked as EOG
llm_load_vocab: control token: 128094 '<|reserved_special_token_85|>' is not marked as EOG
llm_load_vocab: control token: 128151 '<|reserved_special_token_142|>' is not marked as EOG
llm_load_vocab: control token: 128223 '<|reserved_special_token_214|>' is not marked as EOG
llm_load_vocab: control token: 128234 '<|reserved_special_token_225|>' is not marked as EOG
llm_load_vocab: control token: 128221 '<|reserved_special_token_212|>' is not marked as EOG
llm_load_vocab: control token: 128035 '<|reserved_special_token_26|>' is not marked as EOG
llm_load_vocab: control token: 128034 '<|reserved_special_token_25|>' is not marked as EOG
llm_load_vocab: control token: 128254 '<|reserved_special_token_245|>' is not marked as EOG
llm_load_vocab: control token: 128196 '<|reserved_special_token_187|>' is not marked as EOG
llm_load_vocab: control token: 128100 '<|reserved_special_token_91|>' is not marked as EOG
llm_load_vocab: control token: 128190 '<|reserved_special_token_181|>' is not marked as EOG
llm_load_vocab: control token: 128211 '<|reserved_special_token_202|>' is not marked as EOG
llm_load_vocab: control token: 128175 '<|reserved_special_token_166|>' is not marked as EOG
llm_load_vocab: control token: 128084 '<|reserved_special_token_75|>' is not marked as EOG
llm_load_vocab: control token: 128081 '<|reserved_special_token_72|>' is not marked as EOG
llm_load_vocab: control token: 128105 '<|reserved_special_token_96|>' is not marked as EOG
llm_load_vocab: control token: 128083 '<|reserved_special_token_74|>' is not marked as EOG
llm_load_vocab: control token: 128220 '<|reserved_special_token_211|>' is not marked as EOG
llm_load_vocab: control token: 128018 '<|reserved_special_token_9|>' is not marked as EOG
llm_load_vocab: control token: 128005 '<|step_id|>' is not marked as EOG
llm_load_vocab: control token: 128051 '<|reserved_special_token_42|>' is not marked as EOG
llm_load_vocab: control token: 128206 '<|reserved_special_token_197|>' is not marked as EOG
llm_load_vocab: control token: 128048 '<|reserved_special_token_39|>' is not marked as EOG
llm_load_vocab: control token: 128165 '<|reserved_special_token_156|>' is not marked as EOG
llm_load_vocab: control token: 128021 '<|reserved_special_token_12|>' is not marked as EOG
llm_load_vocab: control token: 128070 '<|reserved_special_token_61|>' is not marked as EOG
llm_load_vocab: control token: 128246 '<|reserved_special_token_237|>' is not marked as EOG
llm_load_vocab: control token: 128122 '<|reserved_special_token_113|>' is not marked as EOG
llm_load_vocab: control token: 128080 '<|reserved_special_token_71|>' is not marked as EOG
llm_load_vocab: control token: 128038 '<|reserved_special_token_29|>' is not marked as EOG
llm_load_vocab: control token: 128245 '<|reserved_special_token_236|>' is not marked as EOG
llm_load_vocab: control token: 128030 '<|reserved_special_token_21|>' is not marked as EOG
llm_load_vocab: control token: 128222 '<|reserved_special_token_213|>' is not marked as EOG
llm_load_vocab: control token: 128067 '<|reserved_special_token_58|>' is not marked as EOG
llm_load_vocab: control token: 128121 '<|reserved_special_token_112|>' is not marked as EOG
llm_load_vocab: control token: 128015 '<|reserved_special_token_6|>' is not marked as EOG
llm_load_vocab: control token: 128026 '<|reserved_special_token_17|>' is not marked as EOG
llm_load_vocab: control token: 128127 '<|reserved_special_token_118|>' is not marked as EOG
llm_load_vocab: special tokens cache size = 257
llm_load_vocab: token to piece cache size = 0.7999 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = mllama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 40
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 11B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 9.78 B
llm_load_print_meta: model size       = 5.55 GiB (4.87 BPW)
llm_load_print_meta: general.name     = Model
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256
```




arch: 50, layer: 34, cross_attention_layers: 0, n_embed_head_k: 128, n_head_kv: 8

arch: 50, layer: 38, cross_attention_layers: 1, n_embed_head_k: 128, n_head_kv: 8


     20: STRING     |        1 | tokenizer.ggml.model = 'gpt2'
     21: STRING     |        1 | tokenizer.ggml.pre = 'smaug-bpe'
     22: [STRING]   |   128257 | tokenizer.ggml.tokens
     23: [INT32]    |   128257 | tokenizer.ggml.token_type
     24: [STRING]   |   280147 | tokenizer.ggml.merges
     25: UINT32     |        1 | tokenizer.ggml.bos_token_id = 128000
     26: UINT32     |        1 | tokenizer.ggml.eos_token_id = 128009
     27: UINT32     |        1 | tokenizer.ggml.padding_token_id = 128004
     28: STRING     |        1 | tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- s'
     29: UINT32     |        1 | general.quantization_version = 2

word: ĠÐ·Ð°ÐºÐ¾Ð½Ð¾Ð´Ð°Ð² ÑģÑĤÐ²Ð°
word: ãĢĢ ãĤ¤
word: Ġëħ¸ íķĺìļ°
word: ĠD Ã¼ÅŁ
word: ĠÐ³ ÑĥÑģÑĤ
word: ĠÐĴ Ð°ÑĪ
word: ĠØ§Ùħ ØªÛĮ
word: Ġpar amet
word: Ġparam et
word: Ġpara met
word: ĠÎłÎ±Î½ ÎµÏĢ
word: à¹Į à¸ģà¸£
word: à¹Įà¸ģ à¸£
word: Î¶ Î±
word: ĠëįĶ ìļ±
word: ÙĪ ÙĦØ§Øª
word: ÙĪÙĦ Ø§Øª
word: ÙĪÙĦØ§ Øª
word: Ð² Ð°ÑĤÐ¸ÑģÑı
word: Ð²Ð° ÑĤÐ¸ÑģÑı
word: Ð²Ð°ÑĤÐ¸ ÑģÑı
word: Ġk Ã¶k
word: ĠkÃ¶ k
word: ÙĨ Ø¨
word: ĠÐ²ÑĭÑģÐ¾Ðº Ð¾Ð¹
word: ãĥ¼ ãĥ¼
word: ãĥ¼ãĥ ¼
word: éĶ ¦


word: ĠÐ·Ð°ÐºÐ¾Ð½Ð¾Ð´Ð°Ð² ÑģÑĤÐ²Ð°
word: ãĢĢ ãĤ¤
word: Ġëħ¸ íķĺìļ°
word: ĠD Ã¼ÅŁ
word: ĠÐ³ ÑĥÑģÑĤ
word: ĠÐĴ Ð°ÑĪ
word: ĠØ§Ùħ ØªÛĮ
word: Ġpar amet
word: Ġparam et
word: Ġpara met
word: ĠÎłÎ±Î½ ÎµÏĢ
word: à¹Į à¸ģà¸£
word: à¹Įà¸ģ à¸£
word: Î¶ Î±
word: ĠëįĶ ìļ±
word: ÙĪ ÙĦØ§Øª
word: ÙĪÙĦ Ø§Øª
word: ÙĪÙĦØ§ Øª
word: Ð² Ð°ÑĤÐ¸ÑģÑı
word: Ð²Ð° ÑĤÐ¸ÑģÑı
word: Ð²Ð°ÑĤÐ¸ ÑģÑı
word: Ġk Ã¶k
word: ĠkÃ¶ k
word: ÙĨ Ø¨
word: ĠÐ²ÑĭÑģÐ¾Ðº Ð¾Ð¹
word: ãĥ¼ ãĥ¼
word: ãĥ¼ãĥ ¼
word: éĶ ¦


Rope settings ollama:
rope parameters: n_rot: 128, rope_type: 0, n_ctx_orig: 131072, freq_base: 100, freq_scale: 500000.000000, ext_factor: 1.000000, attn_factor: 0.000000, beta_fast: 1.000000, beta_slow: 32.000000

Rope settings llama.cpp:
rope parameters: n_rot: 128, rope_type: 0, n_ctx_orig: 131072, freq_base: 100, freq_scale: 500000.000000, ext_factor: 1.000000, attn_factor: 0.000000, beta_fast: 1.000000, beta_slow: 32.000000

Ollama:
kq_scale: 0.088388

llama.cpp:
kq_scale: 0.088388



llama.cpp input embeddings for 128006:
```console
input_embeddings tensor type: f32
input_embeddings backend type: CPU
input_embeddings[0] = 0.250884
input_embeddings[1] = -1.903877
input_embeddings[2] = 1.126612
input_embeddings[3] = 0.874009
input_embeddings[4] = -0.151682
input_embeddings[5] = 1.005559
input_embeddings[6] = 2.459111
input_embeddings[7] = -0.477424
input_embeddings[8] = 0.324140
input_embeddings[9] = -1.908321
input_embeddings[10] = 1.061303
input_embeddings[11] = -0.503033
input_embeddings[12] = 1.330023
input_embeddings[13] = -0.804025
input_embeddings[14] = -0.962649
input_embeddings[15] = -0.704584
input_embeddings[16] = 1.547000



(venv) $ ./read-embd-token.py models/llama-3-2-11b.gguf

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: F32
Total elements: 525369344

Embedding values for token 128006:
inp_tokens[0] = 0.0108642578125
inp_tokens[1] = -0.0137939453125
inp_tokens[2] = 0.000736236572265625
inp_tokens[3] = -4.880365614062863e-23
inp_tokens[4] = -0.0147705078125
inp_tokens[5] = 3.5982356646056704e-23
inp_tokens[6] = -0.0032501220703125
inp_tokens[7] = 0.006988525390625
inp_tokens[8] = -0.0135498046875
inp_tokens[9] = -0.00482177734375
inp_tokens[10] = 0.0032806396484375
inp_tokens[11] = -0.003814697265625
inp_tokens[12] = 0.00087738037109375
inp_tokens[13] = -0.00830078125
inp_tokens[14] = 0.0034027099609375
inp_tokens[15] = 0.00701904296875
inp_tokens[16] = 0.02099609375
```

```console
(venv) $ python print-safe-embeddings.py

Tensor Information:
Name: language_model.model.embed_tokens.weight
Shape: torch.Size([128264, 4096])
Type: torch.bfloat16

Embedding values for token 128006:
inp_tokens[0] = -0.00018405914306640625
inp_tokens[1] = -0.000240325927734375
inp_tokens[2] = 0.000164031982421875
inp_tokens[3] = -0.000537872314453125
inp_tokens[4] = 0.0002651214599609375
inp_tokens[5] = -1.2814998626708984e-05
inp_tokens[6] = -0.0002899169921875
inp_tokens[7] = 0.00106048583984375
inp_tokens[8] = 3.6716461181640625e-05
inp_tokens[9] = 0.000530242919921875
inp_tokens[10] = -0.00020599365234375
inp_tokens[11] = -0.0003948211669921875
inp_tokens[12] = 0.000965118408203125
inp_tokens[13] = -0.000164031982421875
inp_tokens[14] = -0.0005645751953125
inp_tokens[15] = 0.000518798828125
inp_tokens[16] = -6.341934204101562e-05
```

ollama:
(venv) $ ./read-embd-token-q.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: Q4_K
Total elements: 525369344

Raw quantized values:
raw[0] = [111   3  37 ... 159  73 109]
raw[1] = [125   3 179 ... 127  86  89]
raw[2] = [ 52   3 254 ... 168 232 191]
raw[3] = [173   2 223 ... 166 107 138]
raw[4] = [ 41   3 152 ... 239 175 159]
raw[5] = [249   2  83 ... 168 174 122]
raw[6] = [116   2 227 ... 122 199 137]
raw[7] = [191   2  45 ... 167 156   4]
raw[8] = [114   3 146 ... 254 141 216]
raw[9] = [ 46   3 158 ... 171 236 169]
raw[10] = [240   2  92 ... 185 156 149]
raw[11] = [168   5 125 ... 191 117 116]
raw[12] = [152   2  51 ... 173 203 116]
raw[13] = [ 59   6 110 ... 175  90 103]
raw[14] = [ 65   3 126 ... 146 120 104]
raw[15] = [202   4 165 ... 216 117  97]
raw[16] = [ 13   6  51 ... 250 199 138]

Quantized data size: 128264
Bytes per value: 1


(gdb) p model.vocab.token_to_id
$14 = std::unordered_map with 128257 elements = {["<|image|>"] = 128256,
  ["<|reserved_special_token_246|>"] = 128255, ["<|reserved_special_token_241|>"] = 128250,
  ["<|reserved_special_token_238|>"] = 128247, ["<|reserved_special_token_235|>"] = 128244,
  ["<|reserved_special_token_234|>"] = 128243, ["<|reserved_special_token_233|>"] = 128242,
  ["<|reserved_special_token_232|>"] = 128241, ["<|reserved_special_token_227|>"] = 128236,
  ["<|reserved_special_token_223|>"] = 128232, ["<|reserved_special_token_222|>"] = 128231,
  ["<|reserved_special_token_220|>"] = 128229, ["<|reserved_special_token_217|>"] = 128226,
  ["<|reserved_special_token_210|>"] = 128219, ["<|reserved_special_token_206|>"] = 128215,
  ["<|reserved_special_token_205|>"] = 128214, ["<|reserved_special_token_199|>"] = 128208,
  ["<|reserved_special_token_198|>"] = 128207, ["<|reserved_special_token_196|>"] = 128205,
  ["<|reserved_special_token_192|>"] = 128201, ["<|reserved_special_token_191|>"] = 128200,
  ["<|reserved_special_token_190|>"] = 128199, ["<|reserved_special_token_188|>"] = 128197,
  ["<|reserved_special_token_186|>"] = 128195, ["<|reserved_special_token_185|>"] = 128194,
  ["<|reserved_special_token_180|>"] = 128189, ["<|reserved_special_token_179|>"] = 128188,
  ["<|reserved_special_token_177|>"] = 128186, ["<|reserved_special_token_176|>"] = 128185,
  ["<|reserved_special_token_172|>"] = 128181, ["<|reserved_special_token_171|>"] = 128180,
  ["<|reserved_special_token_170|>"] = 128179, ["<|reserved_special_token_169|>"] = 128178,
  ["<|reserved_special_token_168|>"] = 128177, ["<|reserved_special_token_167|>"] = 128176,
  ["<|reserved_special_token_163|>"] = 128172, ["<|reserved_special_token_162|>"] = 128171,
  ["<|reserved_special_token_161|>"] = 128170, ["<|reserved_special_token_160|>"] = 128169,
  ["<|reserved_special_token_157|>"] = 128166, ["<|reserved_special_token_154|>"] = 128163,
  ["<|reserved_special_token_150|>"] = 128159, ["<|reserved_special_token_148|>"] = 128157,
  ["<|reserved_special_token_147|>"] = 128156, ["<|reserved_special_token_146|>"] = 128155,
  ["<|reserved_special_token_143|>"] = 128152, ["<|reserved_special_token_141|>"] = 128150,
  ["<|reserved_special_token_139|>"] = 128148, ["<|reserved_special_token_138|>"] = 128147,
  ["<|reserved_special_token_136|>"] = 128145, ["<|reserved_special_token_134|>"] = 128143,
  ["<|reserved_special_token_133|>"] = 128142, ["<|reserved_special_token_130|>"] = 128139,
  ["<|reserved_special_token_128|>"] = 128137, ["<|reserved_special_token_127|>"] = 128136,
  ["<|reserved_special_token_126|>"] = 128135, ["<|reserved_special_token_125|>"] = 128134,
  ["<|reserved_special_token_123|>"] = 128132, ["<|reserved_special_token_120|>"] = 128129,
  ["<|reserved_special_token_116|>"] = 128125, ["<|reserved_special_token_115|>"] = 128124,
  ["<|reserved_special_token_114|>"] = 128123, ["<|reserved_special_token_111|>"] = 128120,
  ["<|reserved_special_token_107|>"] = 128116, ["<|reserved_special_token_104|>"] = 128113,
  ["<|reserved_special_token_102|>"] = 128111, ["<|reserved_special_token_101|>"] = 128110,
  ["<|reserved_special_token_100|>"] = 128109, ["<|reserved_special_token_98|>"] = 128107,
  ["<|reserved_special_token_95|>"] = 128104, ["<|reserved_special_token_94|>"] = 128103,
  ["<|reserved_special_token_93|>"] = 128102, ["<|reserved_special_token_89|>"] = 128098,
  ["<|reserved_special_token_83|>"] = 128092, ["<|reserved_special_token_82|>"] = 128091,
  ["<|reserved_special_token_81|>"] = 128090, ["<|reserved_special_token_79|>"] = 128088,
  ["<|reserved_special_token_77|>"] = 128086, ["<|reserved_special_token_73|>"] = 128082,
  ["<|reserved_special_token_70|>"] = 128079, ["<|reserved_special_token_68|>"] = 128077,
  ["<|reserved_special_token_67|>"] = 128076, ["<|reserved_special_token_65|>"] = 128074,
  ["<|reserved_special_token_60|>"] = 128069, ["<|reserved_special_token_59|>"] = 128068,
  ["<|reserved_special_token_57|>"] = 128066, ["<|reserved_special_token_55|>"] = 128064,
  ["<|reserved_special_token_54|>"] = 128063, ["<|reserved_special_token_52|>"] = 128061,
  ["<|reserved_special_token_51|>"] = 128060, ["<|reserved_special_token_49|>"] = 128058,
  ["<|reserved_special_token_46|>"] = 128055, ["<|reserved_special_token_38|>"] = 128047,
  ["<|reserved_special_token_37|>"] = 128046, ["<|reserved_special_token_36|>"] = 128045,
  ["<|reserved_special_token_35|>"] = 128044, ["<|reserved_special_token_30|>"] = 128039,
  ["<|reserved_special_token_28|>"] = 128037, ["<|reserved_special_token_27|>"] = 128036,
  ["<|reserved_special_token_24|>"] = 128033, ["<|reserved_special_token_20|>"] = 128029,
  ["<|reserved_special_token_19|>"] = 128028, ["<|reserved_special_token_16|>"] = 128025,
  ["<|reserved_special_token_15|>"] = 128024, ["<|reserved_special_token_14|>"] = 128023,
  ["<|reserved_special_token_13|>"] = 128022, ["<|reserved_special_token_10|>"] = 128019,
  ["<|reserved_special_token_8|>"] = 128017, ["<|reserved_special_token_7|>"] = 128016,
  ["<|reserved_special_token_5|>"] = 128014, ["<|reserved_special_token_3|>"] = 128012,
  ["<|reserved_special_token_2|>"] = 128011, ["<|eom_id|>"] = 128008, ["<|finetune_right_pad_id|>"] = 128004,
  ["<|reserved_special_token_0|>"] = 128002, ["ÙĨØ¨"] = 127996, ["Ð²Ð°ÑĤÐ¸ÑģÑı"] = 127994,
  ["ĠëįĶìļ±"] = 127992, ["Î¶Î±"] = 127991, ["ĠØ§ÙħØªÛĮ"] = 127987, ["ĠDÃ¼ÅŁ"] = 127984,
  ["ĠÐ·Ð°ÐºÐ¾Ð½Ð¾Ð´Ð°Ð²ÑģÑĤÐ²Ð°"] = 127981,
  ["à¸Ľà¸ģà¸Ħà¸£à¸Ńà¸ĩ"] = 127980, ["à¸´à¸§à¹Ģà¸ķà¸Ńà¸£"] = 127979,
  ["Ð»ÐµÐ½Ð½Ñĸ"] = 127978...}



(venv) $ ./read-embd-token.py models/llama-3-2-11b-f32.gguf

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: F32
Total elements: 525369344

Embedding values for token 128006:
inp_tokens[0] = 0.0108642578125
inp_tokens[1] = -0.0137939453125
inp_tokens[2] = 0.000736236572265625
inp_tokens[3] = -4.880365614062863e-23
inp_tokens[4] = -0.0147705078125
inp_tokens[5] = 3.5982356646056704e-23
inp_tokens[6] = -0.0032501220703125
inp_tokens[7] = 0.006988525390625
inp_tokens[8] = -0.0135498046875
inp_tokens[9] = -0.00482177734375
inp_tokens[10] = 0.0032806396484375
inp_tokens[11] = -0.003814697265625
inp_tokens[12] = 0.00087738037109375
inp_tokens[13] = -0.00830078125
inp_tokens[14] = 0.0034027099609375
inp_tokens[15] = 0.00701904296875
inp_tokens[16] = 0.02099609375

(venv) $ python print-safe-embeddings.py

Tensor Information:
Name: language_model.model.embed_tokens.weight
Shape: torch.Size([128264, 4096])
Type: torch.bfloat16

Embedding values for token 128006:
inp_tokens[0] = -0.00018405914306640625
inp_tokens[1] = -0.000240325927734375
inp_tokens[2] = 0.000164031982421875
inp_tokens[3] = -0.000537872314453125
inp_tokens[4] = 0.0002651214599609375
inp_tokens[5] = -1.2814998626708984e-05
inp_tokens[6] = -0.0002899169921875
inp_tokens[7] = 0.00106048583984375
inp_tokens[8] = 3.6716461181640625e-05
inp_tokens[9] = 0.000530242919921875
inp_tokens[10] = -0.00020599365234375
inp_tokens[11] = -0.0003948211669921875
inp_tokens[12] = 0.000965118408203125
inp_tokens[13] = -0.000164031982421875
inp_tokens[14] = -0.0005645751953125
inp_tokens[15] = 0.000518798828125
inp_tokens[16] = -6.341934204101562e-05

token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
kq_scale: 0.088388
[New Thread 0x7fffabe00000 (LWP 84068)]
[New Thread 0x7fffab400000 (LWP 84069)]
[New Thread 0x7fffaaa00000 (LWP 84070)]
input_embeddings tensor type: f32
input_embeddings backend type: CPU
input_embeddings[0] = -0.309861
input_embeddings[1] = -0.889324
input_embeddings[2] = 1.652823
input_embeddings[3] = 2.878123
input_embeddings[4] = 2.721913
input_embeddings[5] = 1.118232
input_embeddings[6] = -4.750814
input_embeddings[7] = 1.661370
input_embeddings[8] = 1.025107
input_embeddings[9] = -0.051233
input_embeddings[10] = 7.026322
input_embeddings[11] = 2.739259
input_embeddings[12] = 1.209301
input_embeddings[13] = -1.540354
input_embeddings[14] = -2.568749
input_embeddings[15] = -2.265143
input_embeddings[16] = 2.384460


(venv) $ ./read-embd-token-q.py /home/danbev/.ollama/models/blobs/sha256-652e85aa1e14c9087a4ccc3ab516fb794cbcf152f8b4b8d3c0b828da4ada62d9

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: Q4_K
Total elements: 525369344

Raw quantized values:
raw[0] = [111   3  37 ... 159  73 109]
raw[1] = [125   3 179 ... 127  86  89]
raw[2] = [ 52   3 254 ... 168 232 191]
raw[3] = [173   2 223 ... 166 107 138]
raw[4] = [ 41   3 152 ... 239 175 159]
raw[5] = [249   2  83 ... 168 174 122]
raw[6] = [116   2 227 ... 122 199 137]
raw[7] = [191   2  45 ... 167 156   4]
raw[8] = [114   3 146 ... 254 141 216]
raw[9] = [ 46   3 158 ... 171 236 169]
raw[10] = [240   2  92 ... 185 156 149]
raw[11] = [168   5 125 ... 191 117 116]
raw[12] = [152   2  51 ... 173 203 116]
raw[13] = [ 59   6 110 ... 175  90 103]
raw[14] = [ 65   3 126 ... 146 120 104]
raw[15] = [202   4 165 ... 216 117  97]
raw[16] = [ 13   6  51 ... 250 199 138]

Quantized data size: 128264
Bytes per value: 1


ollama:
(gdb) p lctx.model.vocab.cache_special_tokens
$3 = std::vector of length 257, capacity 512 = {128173, 128158, 128159, 128160, 128161, 128162, 128163, 128164,
  128165, 128166, 128167, 128168, 128169, 128170, 128171, 128172, 128157, 128174, 128175, 128176, 128177, 128178,
  128179, 128180, 128181, 128182, 128183, 128184, 128185, 128186, 128187, 128188, 128142, 128127, 128128, 128129,
  128130, 128131, 128132, 128133, 128134, 128135, 128136, 128137, 128138, 128139, 128140, 128141, 128189, 128143,
  128144, 128145, 128146, 128147, 128148, 128149, 128150, 128151, 128152, 128153, 128154, 128155, 128156, 128236,
  128221, 128222, 128223, 128224, 128225, 128226, 128227, 128228, 128229, 128230, 128231, 128232, 128233, 128234,
  128235, 128220, 128237, 128238, 128239, 128240, 128241, 128242, 128243, 128244, 128245, 128246, 128247, 128248,
  128249, 128250, 128251, 128205, 128190, 128191, 128192, 128193, 128194, 128195, 128196, 128197, 128198, 128199,
  128200, 128201, 128202, 128203, 128204, 128126, 128206, 128207, 128208, 128209, 128210, 128211, 128212, 128213,
  128214, 128215, 128216, 128217, 128218, 128219, 128125, 128252, 128253, 128254, 128255, 128109, 128110, 128111,
  128112, 128113, 128114, 128116, 128115, 128124, 128123, 128122, 128121, 128120, 128119, 128118, 128117, 128041,
  128042, 128043, 128044, 128045, 128046, 128047, 128048, 128049, 128050, 128051, 128052, 128053, 128054, 128055,
  128056, 128057, 128058, 128059, 128060, 128063, 128061, 128029, 128019, 128020, 128021, 128022, 128023, 128024,
  128025, 128026, 128027, 128028, 128040, 128030, 128031, 128032, 128033, 128034, 128035, 128036, 128037, 128038,
  128039, 128097, 128086, 128087, 128088, 128089, 128090, 128091, 128092, 128093, 128094, 128095, 128096, 128062,
  128098, 128099, 128100, 128101, 128102, 128103, 128104, 128105, 128106, 128107, 128108, 128084, 128064, 128065,
  128066, 128067, 128068, 128069, 128070, 128071, 128072, 128073, 128075, 128076, 128077, 128078, 128079, 128080,
  128081, 128074, 128085, 128083, 128082, 128003, 128002, 128011, 128012, 128013, 128014, 128015, 128016, 128017,
  128018, 128004, 128006, 128000, 128007, 128001, 128010, 128005, 128009, 128008, 128256}

  [{first = "!(", second = ":"}] = 127765...}, special_bos_id = 128000, special_eos_id = 128009,
  special_eot_id = 128009, special_eom_id = 128008, special_unk_id = -1, special_sep_id = -1,
  special_pad_id = 128004, special_cls_id = -1, special_mask_id = -1, linefeed_id = 128, special_fim_pre_id = -1,
  special_fim_suf_id = -1, special_fim_mid_id = -1, special_fim_pad_id = -1, special_fim_rep_id = -1,
  special_fim_sep_id = -1, special_eog_ids = std::set with 2 elements = {[0] = 128008, [1] = 128009},
  special_image_id = -1, tokenizer_add_space_prefix = false, tokenizer_add_bos = false, tokenizer_add_eos = false,
  tokenizer_ignore_merges = false, tokenizer_clean_spaces = true, tokenizer_remove_extra_whitespaces = false,
  tokenizer_escape_whitespaces = true, tokenizer_treat_whitespace_as_suffix = false,
  precompiled_charsmap = std::vector of length 0, capacity 0, tokenizer = 0x555559cc1980}


(gdb) p model.vocab.cache_special_tokens
$3 = std::vector of length 257, capacity 512 = {128173, 128158, 128159, 128160, 128161, 128162, 128163, 128164,
  128165, 128166, 128167, 128168, 128169, 128170, 128171, 128172, 128157, 128174, 128175, 128176, 128177, 128178,
  128179, 128180, 128181, 128182, 128183, 128184, 128185, 128186, 128187, 128188, 128142, 128127, 128128, 128129,
  128130, 128131, 128132, 128133, 128134, 128135, 128136, 128137, 128138, 128139, 128140, 128141, 128189, 128143,
  128144, 128145, 128146, 128147, 128148, 128149, 128150, 128151, 128152, 128153, 128154, 128155, 128156, 128236,
  128221, 128222, 128223, 128224, 128225, 128226, 128227, 128228, 128229, 128230, 128231, 128232, 128233, 128234,
  128235, 128220, 128237, 128238, 128239, 128240, 128241, 128242, 128243, 128244, 128245, 128246, 128247, 128248,
  128249, 128250, 128251, 128205, 128190, 128191, 128192, 128193, 128194, 128195, 128196, 128197, 128198, 128199,
  128200, 128201, 128202, 128203, 128204, 128126, 128206, 128207, 128208, 128209, 128210, 128211, 128212, 128213,
  128214, 128215, 128216, 128217, 128218, 128219, 128125, 128252, 128253, 128254, 128255, 128109, 128110, 128111,
  128112, 128113, 128114, 128116, 128115, 128124, 128123, 128122, 128121, 128120, 128119, 128118, 128117, 128041,
  128042, 128043, 128044, 128045, 128046, 128047, 128048, 128049, 128050, 128051, 128052, 128053, 128054, 128055,
  128056, 128057, 128058, 128059, 128060, 128063, 128061, 128029, 128019, 128020, 128021, 128022, 128023, 128024,
  128025, 128026, 128027, 128028, 128040, 128030, 128031, 128032, 128033, 128034, 128035, 128036, 128037, 128038,
  128039, 128097, 128086, 128087, 128088, 128089, 128090, 128091, 128092, 128093, 128094, 128095, 128096, 128062,
  128098, 128099, 128100, 128101, 128102, 128103, 128104, 128105, 128106, 128107, 128108, 128084, 128064, 128065,
  128066, 128067, 128068, 128069, 128070, 128071, 128072, 128073, 128075, 128076, 128077, 128078, 128079, 128080,
  128081, 128074, 128085, 128083, 128082, 128003, 128002, 128011, 128012, 128013, 128014, 128015, 128016, 128017,
  128018, 128004, 128006, 128000, 128007, 128001, 128010, 128005, 128009, 128008, 128256}

      [{first = "!(", second = ":"}] = 127765...}, special_bos_id = 128000, special_eos_id = 128009,
  special_eot_id = 128009, special_eom_id = 128008, special_unk_id = -1, special_sep_id = -1,
  special_pad_id = 128004, special_cls_id = -1, special_mask_id = -1, linefeed_id = 128, special_fim_pre_id = -1,
  special_fim_suf_id = -1, special_fim_mid_id = -1, special_fim_pad_id = -1, special_fim_rep_id = -1,
  special_fim_sep_id = -1, special_eog_ids = std::set with 2 elements = {[0] = 128008, [1] = 128009},
  special_image_id = -1, tokenizer_add_space_prefix = false, tokenizer_add_bos = false, tokenizer_add_eos = false,
  tokenizer_ignore_merges = false, tokenizer_clean_spaces = true, tokenizer_remove_extra_whitespaces = false,
  tokenizer_escape_whitespaces = true, tokenizer_treat_whitespace_as_suffix = false,
  precompiled_charsmap = std::vector of length 0, capacity 0, tokenizer = 0x555559d14c20}


input_embeddings[0] = 0.250884
input_embeddings[1] = -1.903877
input_embeddings[2] = 1.126612
input_embeddings[3] = 0.874009
input_embeddings[4] = -0.151682
input_embeddings[5] = 1.005559
input_embeddings[6] = 2.459111
input_embeddings[7] = -0.477424
input_embeddings[8] = 0.324140
input_embeddings[9] = -1.908321


prompt = What is the Eiffel Tower?
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
kq_scale: 0.088388
[New Thread 0x7fffabe00000 (LWP 97126)]
[New Thread 0x7fffab400000 (LWP 97127)]
[New Thread 0x7fffaaa00000 (LWP 97128)]
input_embeddings tensor type: f32
input_embeddings backend type: CPU
input_embeddings[0] = 0.020752
input_embeddings[1] = -0.001289
input_embeddings[2] = 0.002808
input_embeddings[3] = 0.007385
input_embeddings[4] = -0.008240
input_embeddings[5] = 0.005554
input_embeddings[6] = 0.004944
input_embeddings[7] = -0.001511
input_embeddings[8] = -0.000870
input_embeddings[9] = 0.019043

(venv) $ ./read-embd-token.py models/llama-3-2-11b.gguf

Tensor Information:
Name: token_embd.weight
Shape: 4096 x 128264
Type: F32
Total elements: 525369344

Embedding values for token 3923:
inp_tokens[0] = 0.0107421875
inp_tokens[1] = -9.47713851928711e-06
inp_tokens[2] = -0.0078125
inp_tokens[3] = 0.00156402587890625
inp_tokens[4] = -0.00494384765625
inp_tokens[5] = 0.01141357421875
inp_tokens[6] = 0.00164794921875
inp_tokens[7] = 0.000728607177734375
inp_tokens[8] = -0.00579833984375
inp_tokens[9] = -0.000736236572265625
inp_tokens[10] = -0.00732421875
inp_tokens[11] = 0.0010986328125
inp_tokens[12] = 0.0072021484375
inp_tokens[13] = 0.0152587890625
inp_tokens[14] = -0.0048828125
inp_tokens[15] = -0.00579833984375
inp_tokens[16] = -0.0037994384765625


token_embd.weight tensor type: f32
token_embd.weight backend type: CPU
token_embd.weight[0] = 0.020752
token_embd.weight[1] = -0.001289
token_embd.weight[2] = 0.002808
token_embd.weight[3] = 0.007385
token_embd.weight[4] = -0.008240
token_embd.weight[5] = 0.005554
token_embd.weight[6] = 0.004944
token_embd.weight[7] = -0.001511
token_embd.weight[8] = -0.000870
token_embd.weight[9] = 0.019043


-----------------------------------------
128006 in process:
token_embd.weight tensor type: f32
token_embd.weight backend type: CPU
token_embd.weight[0] = -0.000184
token_embd.weight[1] = -0.000240
token_embd.weight[2] = 0.000164
token_embd.weight[3] = -0.000538
token_embd.weight[4] = 0.000265
token_embd.weight[5] = -0.000013
token_embd.weight[6] = -0.000290
token_embd.weight[7] = 0.001060
token_embd.weight[8] = 0.000037
token_embd.weight[9] = 0.000530


llm_load_print_meta: general.name     = Llama 3.2 11B Vision Instruct New
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'

llm_load_print_meta: general.name     = Model
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256


special_eos_id = 128009, special_eot_id = 128009, special_eom_id = 128008, special_unk_id = -1,
  special_sep_id = -1, special_pad_id = 128004, special_cls_id = -1, special_mask_id = -1,
  special_start_header_id = -1, special_end_header_id = -1, linefeed_id = 128, special_fim_pre_id = -1,
  special_fim_suf_id = -1, special_fim_mid_id = -1, special_fim_pad_id = -1, special_fim_rep_id = -1,
  special_fim_sep_id = -1, special_eog_ids = std::set with 2 elements = {[0] = 128008, [1] = 128009},
  special_image_id = -1, tokenizer_add_space_prefix = false, tokenizer_add_bos = false, tokenizer_add_eos = false,
  tokenizer_ignore_merges = false, tokenizer_clean_spaces = true, tokenizer_remove_extra_whitespaces = false,
  tokenizer_escape_whitespaces = true, tokenizer_treat_whitespace_as_suffix = false,

special_eos_id = 128009, special_eot_id = 128009, special_eom_id = 128008, special_unk_id = -1,
  special_sep_id = -1, special_pad_id = 128004, special_cls_id = -1, special_mask_id = -1,
  special_start_header_id = 128006, special_end_header_id = 128007, linefeed_id = 128, special_fim_pre_id = -1,
  special_fim_suf_id = -1, special_fim_mid_id = -1, special_fim_pad_id = -1, special_fim_rep_id = -1,
  special_fim_sep_id = -1, special_eog_ids = std::set with 2 elements = {[0] = 128008, [1] = 128009},
  special_image_id = -1, tokenizer_add_space_prefix = false, tokenizer_add_bos = false, tokenizer_add_eos = false,
  tokenizer_ignore_merges = false, tokenizer_clean_spaces = true, tokenizer_remove_extra_whitespaces = false,
  tokenizer_escape_whitespaces = true, tokenizer_treat_whitespace_as_suffix = false,
  precompiled_charsmap = std::vector of length 0, capacity 0, tokenizer = 0x555559d14860}


llm_load_print_meta: general.name     = Llama 3.2 11B Vision Instruct New
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>


ollama:
    364:    4194304 |  4096,  1024,     1,     1 | Q4_K    | blk.37.attn_k.weight
    365:   16777216 |  4096,  4096,     1,     1 | Q4_K    | blk.37.attn_output.weight
    366:   16777216 |  4096,  4096,     1,     1 | Q4_K    | blk.37.attn_q.weight
    367:    4194304 |  4096,  1024,     1,     1 | Q6_K    | blk.37.attn_v.weight
    369:       4096 |  4096,     1,     1,     1 | F32     | blk.37.attn_norm.weight
    370:   58720256 | 14336,  4096,     1,     1 | Q6_K    | blk.37.ffn_down.weight
    371:   58720256 |  4096, 14336,     1,     1 | Q4_K    | blk.37.ffn_gate.weight
    372:   58720256 |  4096, 14336,     1,     1 | Q4_K    | blk.37.ffn_up.weight
    373:       4096 |  4096,     1,     1,     1 | F32     | blk.37.ffn_norm.weight

llama.cpp:
    878:    4194304 |  4096,  1024,     1,     1 | F16     | blk.37.attn_k.weight
    879:   16777216 |  4096,  4096,     1,     1 | F16     | blk.37.attn_output.weight
    880:   16777216 |  4096,  4096,     1,     1 | F16     | blk.37.attn_q.weight
    881:    4194304 |  4096,  1024,     1,     1 | F16     | blk.37.attn_v_proj.weight
    873:       4096 |  4096,     1,     1,     1 | F32     | blk.37.attn_norm.weight
    874:   58720256 | 14336,  4096,     1,     1 | F16     | blk.37.ffn_down.weight
    875:   58720256 |  4096, 14336,     1,     1 | F16     | blk.37.ffn_gate.weight
    876:   58720256 |  4096, 14336,     1,     1 | F16     | blk.37.ffn_up.weight
    877:       4096 |  4096,     1,     1,     1 | F32     | blk.37.post_attention_norm.weight


The tensors match excatly to the original model:
token = 128000
token = 128006
token = 9125
token = 128007
token = 271
token = 2675
token = 527
token = 264
token = 11190
token = 18328
token = 128009
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
blk.0.attn_norm.weight tensor type: f32
blk.0.attn_norm.weight backend type: CPU
blk.0.attn_norm.weight[0] = 0.047119
blk.0.attn_norm.weight[1] = 0.187500
blk.0.attn_norm.weight[2] = 0.417969
blk.0.attn_norm.weight[3] = 0.017090
blk.0.attn_norm.weight[4] = 0.433594
blk.0.attn_norm.weight[5] = 0.021484
blk.0.attn_norm.weight[6] = -0.000206
blk.0.attn_norm.weight[7] = 0.004547
blk.0.attn_norm.weight[8] = 0.034180
blk.0.attn_norm.weight[9] = 0.024658
blk.0.ffn_down.weight tensor type: f16
blk.0.ffn_down.weight backend type: CPU
blk.0.ffn_down.weight[0] = 0.012085
blk.0.ffn_down.weight[1] = -0.014771
blk.0.ffn_down.weight[2] = -0.008606
blk.0.ffn_down.weight[3] = 0.000786
blk.0.ffn_down.weight[4] = -0.022095
blk.0.ffn_down.weight[5] = -0.005341
blk.0.ffn_down.weight[6] = 0.000843
blk.0.ffn_down.weight[7] = 0.001816
blk.0.ffn_down.weight[8] = 0.010986
blk.0.ffn_down.weight[9] = -0.001320
blk.0.ffn_gate.weight tensor type: f16
blk.0.ffn_gate.weight backend type: CPU
blk.0.ffn_gate.weight[0] = -0.016602
blk.0.ffn_gate.weight[1] = -0.006226
blk.0.ffn_gate.weight[2] = -0.001389
blk.0.ffn_gate.weight[3] = -0.000462
blk.0.ffn_gate.weight[4] = 0.007294
blk.0.ffn_gate.weight[5] = 0.003860
blk.0.ffn_gate.weight[6] = -0.003799
blk.0.ffn_gate.weight[7] = -0.027466
blk.0.ffn_gate.weight[8] = -0.021729
blk.0.ffn_gate.weight[9] = 0.001320
blk.0.ffn_up.weight tensor type: f16
blk.0.ffn_up.weight backend type: CPU
blk.0.ffn_up.weight[0] = -0.011658
blk.0.ffn_up.weight[1] = 0.006042
blk.0.ffn_up.weight[2] = 0.010437
blk.0.ffn_up.weight[3] = 0.005493
blk.0.ffn_up.weight[4] = 0.019409
blk.0.ffn_up.weight[5] = 0.003906
blk.0.ffn_up.weight[6] = 0.004578
blk.0.ffn_up.weight[7] = -0.006165
blk.0.ffn_up.weight[8] = -0.007111
blk.0.ffn_up.weight[9] = 0.015869
blk.0.post_attention_norm.weight tensor type: f32
blk.0.post_attention_norm.weight backend type: CPU
blk.0.post_attention_norm.weight[0] = 0.134766
blk.0.post_attention_norm.weight[1] = 0.125000
blk.0.post_attention_norm.weight[2] = 0.137695
blk.0.post_attention_norm.weight[3] = 0.135742
blk.0.post_attention_norm.weight[4] = 0.125977
blk.0.post_attention_norm.weight[5] = 0.134766
blk.0.post_attention_norm.weight[6] = 0.134766
blk.0.post_attention_norm.weight[7] = 0.134766
blk.0.post_attention_norm.weight[8] = 0.134766
blk.0.post_attention_norm.weight[9] = 0.134766
blk.0.attn_k.weight tensor type: f16
blk.0.attn_k.weight backend type: CPU
blk.0.attn_k.weight[0] = -0.104492
blk.0.attn_k.weight[1] = -0.150391
blk.0.attn_k.weight[2] = 0.082520
blk.0.attn_k.weight[3] = -0.053711
blk.0.attn_k.weight[4] = -0.101562
blk.0.attn_k.weight[5] = -0.003510
blk.0.attn_k.weight[6] = -0.000261
blk.0.attn_k.weight[7] = -0.002975
blk.0.attn_k.weight[8] = -0.011353
blk.0.attn_k.weight[9] = 0.038574
blk.0.attn_output.weight tensor type: f16
blk.0.attn_output.weight backend type: CPU
blk.0.attn_output.weight[0] = 0.005920
blk.0.attn_output.weight[1] = -0.001984
blk.0.attn_output.weight[2] = -0.010132
blk.0.attn_output.weight[3] = -0.001106
blk.0.attn_output.weight[4] = 0.003387
blk.0.attn_output.weight[5] = -0.009949
blk.0.attn_output.weight[6] = -0.008484
blk.0.attn_output.weight[7] = -0.000187
blk.0.attn_output.weight[8] = -0.000157
blk.0.attn_output.weight[9] = -0.001305
blk.0.attn_q.weight tensor type: f16
blk.0.attn_q.weight backend type: CPU
blk.0.attn_q.weight[0] = 0.005188
blk.0.attn_q.weight[1] = -0.029297
blk.0.attn_q.weight[2] = -0.006439
blk.0.attn_q.weight[3] = -0.018188
blk.0.attn_q.weight[4] = -0.003265
blk.0.attn_q.weight[5] = 0.026367
blk.0.attn_q.weight[6] = -0.000717
blk.0.attn_q.weight[7] = -0.002289
blk.0.attn_q.weight[8] = -0.025146
blk.0.attn_q.weight[9] = 0.011902
blk.0.attn_v_proj.weight tensor type: f16
blk.0.attn_v_proj.weight backend type: CPU
blk.0.attn_v_proj.weight[0] = 0.011475
blk.0.attn_v_proj.weight[1] = -0.001747
blk.0.attn_v_proj.weight[2] = -0.000641
blk.0.attn_v_proj.weight[3] = 0.003876
blk.0.attn_v_proj.weight[4] = 0.002533
blk.0.attn_v_proj.weight[5] = -0.006256
blk.0.attn_v_proj.weight[6] = 0.002045
blk.0.attn_v_proj.weight[7] = -0.000163
blk.0.attn_v_proj.weight[8] = -0.002396
blk.0.attn_v_proj.weight[9] = -0.012451
token_embd.weight tensor type: f32
token_embd.weight backend type: CPU
token_embd.weight[0] = 0.001007
token_embd.weight[1] = 0.005585
token_embd.weight[2] = -0.003403
token_embd.weight[3] = -0.001236
token_embd.weight[4] = -0.003571
token_embd.weight[5] = 0.000626
token_embd.weight[6] = -0.001495
token_embd.weight[7] = -0.002167
token_embd.weight[8] = -0.003616
token_embd.weight[9] = -0.004333
output.weight tensor type: f32
output.weight backend type: CPU
output.weight[0] = 0.008179
output.weight[1] = 0.007172
output.weight[2] = 0.012451
output.weight[3] = 0.023682
output.weight[4] = -0.017578
output.weight[5] = 0.012756
output.weight[6] = -0.020020
output.weight[7] = -0.005280
output.weight[8] = -0.001541
output.weight[9] = 0.015564
output_norm.weight tensor type: f32
output_norm.weight backend type: CPU
output_norm.weight[0] = 2.468750
output_norm.weight[1] = 2.390625
output_norm.weight[2] = 2.531250
output_norm.weight[3] = 2.421875
output_norm.weight[4] = 2.390625
output_norm.weight[5] = 2.468750
output_norm.weight[6] = 2.265625
output_norm.weight[7] = 2.437500
output_norm.weight[8] = 2.296875
output_norm.weight[9] = 2.328125


But are we using them in the correct locations? If one of them has been 
placed/swapped the model might not work as expected.


(gdb) p vocab.cache_special_tokens
$8 = std::vector of length 257, capacity 512 = {128173, 128158, 128159, 128160, 128161, 128162, 128163, 128164,
  128165, 128166, 128167, 128168, 128169, 128170, 128171, 128172, 128157, 128174, 128175, 128176, 128177, 128178,
  128179, 128180, 128181, 128182, 128183, 128184, 128185, 128186, 128187, 128188, 128142, 128127, 128128, 128129,
  128130, 128131, 128132, 128133, 128134, 128135, 128136, 128137, 128138, 128139, 128140, 128141, 128189, 128143,
  128144, 128145, 128146, 128147, 128148, 128149, 128150, 128151, 128152, 128153, 128154, 128155, 128156, 128236,
  128221, 128222, 128223, 128224, 128225, 128226, 128227, 128228, 128229, 128230, 128231, 128232, 128233, 128234,
  128235, 128220, 128237, 128238, 128239, 128240, 128241, 128242, 128243, 128244, 128245, 128246, 128247, 128248,
  128249, 128250, 128251, 128205, 128190, 128191, 128192, 128193, 128194, 128195, 128196, 128197, 128198, 128199,
  128200, 128201, 128202, 128203, 128204, 128126, 128206, 128207, 128208, 128209, 128210, 128211, 128212, 128213,
  128214, 128215, 128216, 128217, 128218, 128219, 128125, 128252, 128253, 128254, 128255, 128109, 128110, 128111,
  128112, 128113, 128114, 128116, 128115, 128124, 128123, 128122, 128121, 128120, 128119, 128118, 128117, 128041,
  128042, 128043, 128044, 128045, 128046, 128047, 128048, 128049, 128050, 128051, 128052, 128053, 128054, 128055,
  128056, 128057, 128058, 128059, 128060, 128063, 128061, 128029, 128019, 128020, 128021, 128022, 128023, 128024,
  128025, 128026, 128027, 128028, 128040, 128030, 128031, 128032, 128033, 128034, 128035, 128036, 128037, 128038,
  128039, 128097, 128086, 128087, 128088, 128089, 128090, 128091, 128092, 128093, 128094, 128095, 128096, 128062,
  128098, 128099, 128100, 128101, 128102, 128103, 128104, 128105, 128106, 128107, 128108, 128084, 128064, 128065,
  128066, 128067, 128068, 128069, 128070, 128071, 128072, 128073, 128075, 128076, 128077, 128078, 128079, 128080,
  128081, 128074, 128085, 128083, 128082, 128003, 128002, 128011, 128012, 128013, 128014, 128015, 128016, 128017,
  128018, 128004, 128006, 128000, 128007, 128001, 128010, 128005, 128009, 128008, 128256}



Hmm, so when I set tokenizer.add_eos to true I get token appended:
```
token = 128000     <------- BOS token
token = 128006     <------- Start header token
token = 9125
token = 128007     <------- End of header token
token = 271
token = 2675
token = 527
token = 264
token = 11190
token = 18328
token = 128009     <------- EOS token
token = 128006     <------- Start header token
token = 882
token = 128007     <------- End of header token
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009     <------- EOS token
token = 128006     <------- Start header token
token = 78191
token = 128007     <------- End of header token
token = 271
token = 128009     <------- EOS token
```
Notice that 128009 was appended at the end. This is the EOS token. But also
notice that this is 

token = 128000
token = 128006
token = 9125
token = 128007
token = 271
token = 2675
token = 527
token = 264
token = 11190
token = 18328
token = 128009
token = 128006
token = 882
token = 128007
token = 271
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271
token = 128009


llm_load_print_meta: general.name     = Llama 3.2 11B Vision Instruct New
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256

llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256



$1 = std::vector of length 128256, capacity 128256 = {{id = 0, logit = 8.49976158, p = 0}, {id = 1,
    logit = 10.1715689, p = 0}, {id = 2, logit = 8.18358612, p = 0}, {id = 3, logit = 5.96539259, p = 0}, {id = 4,
    logit = 4.58965874, p = 0}, {id = 5, logit = 3.58176112, p = 0}, {id = 6, logit = 5.22433853, p = 0}, {id = 7,
    logit = 6.61313725, p = 0}, {id = 8, logit = 3.1309166, p = 0}, {id = 9, logit = 10.8829632, p = 0}, {id = 10,
    logit = 4.16702938, p = 0}, {id = 11, logit = -0.871033311, p = 0}, {id = 12, logit = 6.87977314, p = 0}, {
    id = 13, logit = 2.09409761, p = 0}, {id = 14, logit = 4.69480038, p = 0}, {id = 15, logit = 5.72630501,
    p = 0}, {id = 16, logit = 11.2072248, p = 0}, {id = 17, logit = 5.98049355, p = 0}, {id = 18,
    logit = 4.52976608, p = 0}, {id = 19, logit = 4.35963964, p = 0}, {id = 20, logit = 4.28019142, p = 0}, {
    id = 21, logit = 2.90609908, p = 0}, {id = 22, logit = 5.58546638, p = 0}, {id = 23, logit = 6.61845541,
    p = 0}, {id = 24, logit = 4.68056488, p = 0}, {id = 25, logit = 3.8785944, p = 0}, {id = 26,
    logit = 3.23900819, p = 0}, {id = 27, logit = 5.60019684, p = 0}, {id = 28, logit = 3.70628548, p = 0}, {
    id = 29, logit = 6.76123238, p = 0}, {id = 30, logit = 6.10255241, p = 0}, {id = 31, logit = 3.93015885,
    p = 0}, {id = 32, logit = 23.4324837, p = 0}, {id = 33, logit = 13.0228834, p = 0}, {id = 34,
    logit = 13.4126663, p = 0}, {id = 35, logit = 12.2931318, p = 0}, {id = 36, logit = 15.1049614, p = 0}, {
    id = 37, logit = 11.1683235, p = 0}, {id = 38, logit = 13.2216568, p = 0}, {id = 39, logit = 9.54018593,
    p = 0}, {id = 40, logit = 14.3643408, p = 0}, {id = 41, logit = 9.79798126, p = 0}, {id = 42,
    logit = 8.29475594, p = 0}, {id = 43, logit = 15.5155182, p = 0}, {id = 44, logit = 11.6688404, p = 0}, {
    id = 45, logit = 13.8286457, p = 0}, {id = 46, logit = 11.0427971, p = 0}, {id = 47, logit = 11.2783184,
    p = 0}, {id = 48, logit = 6.400424, p = 0}, {id = 49, logit = 8.81002235, p = 0}, {id = 50, logit = 10.4180832,
    p = 0}, {id = 51, logit = 12.4973354, p = 0}, {id = 52, logit = 7.55433369, p = 0}, {id = 53,
    logit = 10.2087955, p = 0}, {id = 54, logit = 10.9162216, p = 0}, {id = 55, logit = 7.13089657, p = 0}, {
    id = 56, logit = 9.26593208, p = 0}, {id = 57, logit = 8.93574429, p = 0}, {id = 58, logit = 7.99292278,
    p = 0}, {id = 59, logit = 5.16484356, p = 0}, {id = 60, logit = 2.26537371, p = 0}, {id = 61,
    logit = 4.27812099, p = 0}, {id = 62, logit = 6.97065258, p = 0}, {id = 63, logit = 5.55177784, p = 0}, {
    id = 64, logit = 10.826395, p = 0}, {id = 65, logit = 4.96431303, p = 0}, {id = 66, logit = 5.6362381, p = 0}, {
    id = 67, logit = 3.24965167, p = 0}, {id = 68, logit = 5.17834473, p = 0}, {id = 69, logit = 2.763237, p = 0}, {
    id = 70, logit = 3.31013441, p = 0}, {id = 71, logit = 1.78753734, p = 0}, {id = 72, logit = 2.03962135,
    p = 0}, {id = 73, logit = 1.93985009, p = 0}, {id = 74, logit = 0.535915256, p = 0}, {id = 75,
    logit = 8.8295517, p = 0}, {id = 76, logit = 3.75333977, p = 0}, {id = 77, logit = 4.26086664, p = 0}, {
    id = 78, logit = 3.47949839, p = 0}, {id = 79, logit = 2.88825035, p = 0}, {id = 80, logit = -0.654387832,
    p = 0}, {id = 81, logit = 2.62319422, p = 0}, {id = 82, logit = 3.51516438, p = 0}, {id = 83,
    logit = 4.42897701, p = 0}, {id = 84, logit = 1.67860031, p = 0}, {id = 85, logit = 1.64415562, p = 0}, {
    id = 86, logit = 3.85886574, p = 0}, {id = 87, logit = 1.91346169, p = 0}, {id = 88, logit = 0.478820562,
    p = 0}, {id = 89, logit = 1.77757692, p = 0}, {id = 90, logit = 4.44467449, p = 0}, {id = 91,
    logit = 5.3191452, p = 0}, {id = 92, logit = 2.52070379, p = 0}, {id = 93, logit = 5.86438942, p = 0}, {
    id = 94, logit = 1.15119612, p = 0}, {id = 95, logit = 4.47134972, p = 0}, {id = 96, logit = -2.37457752,
    p = 0}, {id = 97, logit = 2.88380814, p = 0}, {id = 98, logit = 2.30934954, p = 0}, {id = 99,
    logit = 3.00934339, p = 0}, {id = 100, logit = 0.234121978, p = 0}, {id = 101, logit = 3.32834339, p = 0}, {
    id = 102, logit = 2.49167013, p = 0}, {id = 103, logit = 0.215313435, p = 0}, {id = 104,
    logit = -0.00874829292, p = 0}, {id = 105, logit = 1.43826699, p = 0}, {id = 106, logit = 5.06977272, p = 0}, {
    id = 107, logit = -1.9828943, p = 0}, {id = 108, logit = 3.64318919, p = 0}, {id = 109, logit = 3.10319901,
    p = 0}, {id = 110, logit = 4.4609375, p = 0}, {id = 111, logit = 1.50990367, p = 0}, {id = 112,
    logit = -2.13374853, p = 0}, {id = 113, logit = -2.24656177, p = 0}, {id = 114, logit = 0.171405077, p = 0}, {
    id = 115, logit = -1.33581853, p = 0}, {id = 116, logit = 3.22373414, p = 0}, {id = 117, logit = 1.47635686,
    p = 0}, {id = 118, logit = 4.7530303, p = 0}, {id = 119, logit = -6.05642128, p = 0}, {id = 120,
    logit = 1.43492329, p = 0}, {id = 121, logit = 2.19013405, p = 0}, {id = 122, logit = 2.17576599, p = 0}, {
    id = 123, logit = 1.24602056, p = 0}, {id = 124, logit = 0.492516935, p = 0}, {id = 125, logit = 0.492717743,
    p = 0}, {id = 126, logit = 0.27341938, p = 0}, {id = 127, logit = 0.217201293, p = 0}, {id = 128,
    logit = 3.61102986, p = 0}, {id = 129, logit = 5.21890926, p = 0}, {id = 130, logit = 1.95245504, p = 0}, {
    id = 131, logit = 3.46553183, p = 0}, {id = 132, logit = 1.33044052, p = 0}, {id = 133, logit = 2.78988004,
    p = 0}, {id = 134, logit = -0.19154492, p = 0}, {id = 135, logit = 4.97160912, p = 0}, {id = 136,
    logit = 0.34388864, p = 0}, {id = 137, logit = 1.03900313, p = 0}, {id = 138, logit = 2.83288765, p = 0}, {
    id = 139, logit = 1.81119275, p = 0}, {id = 140, logit = -0.34994936, p = 0}, {id = 141, logit = 2.57403493,
    p = 0}, {id = 142, logit = 0.664959431, p = 0}, {id = 143, logit = 2.16751337, p = 0}, {id = 144,
    logit = 3.08757401, p = 0}, {id = 145, logit = 0.732305884, p = 0}, {id = 146, logit = 1.598665, p = 0}, {
    id = 147, logit = 1.4109385, p = 0}, {id = 148, logit = 0.902055979, p = 0}, {id = 149, logit = 1.13629985,
    p = 0}, {id = 150, logit = 3.41794729, p = 0}, {id = 151, logit = 3.25173187, p = 0}, {id = 152,
    logit = 0.352133751, p = 0}, {id = 153, logit = -1.94435334, p = 0}, {id = 154, logit = -0.198191166, p = 0}, {
    id = 155, logit = 3.83232141, p = 0}, {id = 156, logit = -0.658055067, p = 0}, {id = 157, logit = 3.4361074,
    p = 0}, {id = 158, logit = 8.25267601, p = 0}, {id = 159, logit = -0.783573866, p = 0}, {id = 160,
    logit = -2.25503516, p = 0}, {id = 161, logit = -0.778335512, p = 0}, {id = 162, logit = -1.46536994, p = 0}, {
    id = 163, logit = -1.76495659, p = 0}, {id = 164, logit = 4.74475336, p = 0}, {id = 165, logit = -0.999192834,
    p = 0}, {id = 166, logit = 1.05844581, p = 0}, {id = 167, logit = 4.33199692, p = 0}, {id = 168,
    logit = 3.14366794, p = 0}, {id = 169, logit = 2.34770203, p = 0}, {id = 170, logit = 5.12091351, p = 0}, {
    id = 171, logit = 3.45590782, p = 0}, {id = 172, logit = -1.46026778, p = 0}, {id = 173, logit = 2.2244525,
    p = 0}, {id = 174, logit = 1.32021654, p = 0}, {id = 175, logit = 4.76530409, p = 0}, {id = 176,


n_rot: 128, n_ctx_orig: 131072, freq_base: 0, freq_scale: 0, ext_factor: 500000.000000, attn_factor: 1.000000, beta_fast: 0.000000, beta_slow: 1.000000

n_rot: 128, n_ctx_orig: 131072, freq_base: 0, freq_scale: 0, ext_factor: 500000.000000, attn_factor: 1.000000, beta_fast: 0.000000, beta_slow: 1.000000


llm_load_print_meta: arch             = mllama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 40
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336

llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown

llm_load_print_meta: model type       = 11B
llm_load_print_meta: model ftype      = F16
llm_load_print_meta: model params     = 10.67 B
llm_load_print_meta: model size       = 22.03 GiB (17.74 BPW)
llm_load_print_meta: general.name     = Llama 3.2 11B Vision Instruct New
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'
llm_load_print_meta: max token length = 256

llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = mllama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 40
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 11B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 9.78 B
llm_load_print_meta: model size       = 5.55 GiB (4.87 BPW)
llm_load_print_meta: general.name     = Model
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_print_meta: EOM token        = 128008 '<|eom_id|>'
llm_load_print_meta: PAD token        = 128004 '<|finetune_right_pad_id|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOG token        = 128008 '<|eom_id|>'
llm_load_print_meta: EOG token        = 128009 '<|eot_id|>'


gdb) p vocab.special_eog_ids
$4 = std::set with 2 elements = {[0] = 128008, [1] = 128009}


llama_model_loader: - kv  21:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  22:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  23:            tokenizer.ggml.padding_token_id u32              = 128004
llama_model_loader: - kv  24:                    tokenizer.chat_template str              = {{- bos_token }}\n{%- if custom_tools ...
llama_model_loader: - kv  25:               general.quantization_version u32              = 2

(gdb) p vocab.special_eos_id
$1 = 128009
(gdb) p vocab.special_eog_ids
$2 = std::set with 2 elements = {[0] = 128008, [1] = 128009}


ollama:
(gdb) p vocab.id_to_token[3]
$5 = {text = "$", score = 0, attr = LLAMA_TOKEN_ATTR_NORMAL}
(gdb) p vocab.id_to_token[128000]
$6 = {text = "<|begin_of_text|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128001]
$7 = {text = "<|end_of_text|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128002]
$8 = {text = "<|reserved_special_token_0|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128003]
$9 = {text = "<|reserved_special_token_1|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128004]
$10 = {text = "<|finetune_right_pad_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128005]
$11 = {text = "<|step_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128006]
$12 = {text = "<|start_header_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128007]
$13 = {text = "<|end_header_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128008]
$14 = {text = "<|eom_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128009]
$15 = {text = "<|eot_id|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128010]
$16 = {text = "<|python_tag|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128011]
$17 = {text = "<|reserved_special_token_2|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128012]
$18 = {text = "<|reserved_special_token_3|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128013]
$19 = {text = "<|reserved_special_token_4|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}
(gdb) p vocab.id_to_token[128256]
$20 = {text = "<|image|>", score = 0, attr = LLAMA_TOKEN_ATTR_CONTROL}

rope_factors-24 tensor type: f32
rope_factors-24 backend type: CPU
rope_factors-24[0] = 1.000000
rope_factors-24[1] = 1.000000
rope_factors-24[2] = 1.000000
rope_factors-24[3] = 1.000000
rope_factors-24[4] = 1.000000
rope_factors-24[5] = 1.000000
rope_factors-24[6] = 1.000000
rope_factors-24[7] = 1.000000
rope_factors-24[8] = 1.000000
rope_factors-24[9] = 1.000000
rope_factors-24[10] = 1.000000
rope_factors-24[11] = 1.000000
rope_factors-24[12] = 1.000000
rope_factors-24[13] = 1.000000
rope_factors-24[14] = 1.000000
rope_factors-24[15] = 1.000000
rope_factors-24[16] = 1.000000
rope_factors-24[17] = 1.000000
rope_factors-24[18] = 1.000000
rope_factors-24[19] = 1.000000
rope_factors-24[20] = 1.000000
rope_factors-24[21] = 1.000000
rope_factors-24[22] = 1.000000
rope_factors-24[23] = 1.000000
rope_factors-24[24] = 1.000000
rope_factors-24[25] = 1.000000
rope_factors-24[26] = 1.000000
rope_factors-24[27] = 1.000000
rope_factors-24[28] = 1.000000
rope_factors-24[29] = 1.207484
rope_factors-24[30] = 1.553415
rope_factors-24[31] = 2.026313
rope_factors-24[32] = 2.694530
rope_factors-24[33] = 3.684253
rope_factors-24[34] = 5.257327
rope_factors-24[35] = 8.000000
rope_factors-24[36] = 8.000000
rope_factors-24[37] = 8.000000
rope_factors-24[38] = 8.000000
rope_factors-24[39] = 8.000000
rope_factors-24[40] = 8.000000
rope_factors-24[41] = 8.000000
rope_factors-24[42] = 8.000000
rope_factors-24[43] = 8.000000
rope_factors-24[44] = 8.000000
rope_factors-24[45] = 8.000000
rope_factors-24[46] = 8.000000
rope_factors-24[47] = 8.000000
rope_factors-24[48] = 8.000000
rope_factors-24[49] = 8.000000

rope_factors-24 backend type: CPU
rope_factors-24[0] = 1.000000
rope_factors-24[1] = 1.000000
rope_factors-24[2] = 1.000000
rope_factors-24[3] = 1.000000
rope_factors-24[4] = 1.000000
rope_factors-24[5] = 1.000000
rope_factors-24[6] = 1.000000
rope_factors-24[7] = 1.000000
rope_factors-24[8] = 1.000000
rope_factors-24[9] = 1.000000
rope_factors-24[10] = 1.000000
rope_factors-24[11] = 1.000000
rope_factors-24[12] = 1.000000
rope_factors-24[13] = 1.000000
rope_factors-24[14] = 1.000000
rope_factors-24[15] = 1.000000
rope_factors-24[16] = 1.000000
rope_factors-24[17] = 1.000000
rope_factors-24[18] = 1.000000
rope_factors-24[19] = 1.000000
rope_factors-24[20] = 1.000000
rope_factors-24[21] = 1.000000
rope_factors-24[22] = 1.000000
rope_factors-24[23] = 1.000000
rope_factors-24[24] = 1.000000
rope_factors-24[25] = 1.000000
rope_factors-24[26] = 1.000000
rope_factors-24[27] = 1.000000
rope_factors-24[28] = 1.000000
rope_factors-24[29] = 1.207484
rope_factors-24[30] = 1.553415
rope_factors-24[31] = 2.026313
rope_factors-24[32] = 2.694530
rope_factors-24[33] = 3.684253
rope_factors-24[34] = 5.257327
rope_factors-24[35] = 8.000000
rope_factors-24[36] = 8.000000
rope_factors-24[37] = 8.000000
rope_factors-24[38] = 8.000000
rope_factors-24[39] = 8.000000
rope_factors-24[40] = 8.000000
rope_factors-24[41] = 8.000000
rope_factors-24[42] = 8.000000
rope_factors-24[43] = 8.000000
rope_factors-24[44] = 8.000000
rope_factors-24[45] = 8.000000
rope_factors-24[46] = 8.000000
rope_factors-24[47] = 8.000000
rope_factors-24[48] = 8.000000
rope_factors-24[49] = 8.000000



load_vocab: n_vocab: 128257
llm_tokenizer_bpe: using default regex for BPE tokenization pre-processing
load_vocab: found eom token (128008): <|eom_id|>
load_vocab: found eot token (128009): <|eot_id|>
load_vocab: found eog token (128008): <|eom_id|>
load_vocab: found eog token (128009): <|eot_id|>

load_vocab: n_vocab: 128257
llm_tokenizer_bpe: using default regex for BPE tokenization pre-processing
load_vocab: found eom token (128008): <|eom_id|>
load_vocab: found eot token (128009): <|eot_id|>
load_vocab: found eog token (128008): <|eom_id|>
load_vocab: found eog token (128009): <|eot_id|>


special token: 128000
special token: 128001
special token: 128002
special token: 128003
special token: 128004
special token: 128005
special token: 128006
special token: 128007
special token: 128008
special token: 128009
special token: 128010
special token: 128011
special token: 128012
special token: 128013
special token: 128014
special token: 128015
special token: 128016
special token: 128017
special token: 128018
special token: 128019
special token: 128020
special token: 128021
special token: 128022
special token: 128023
special token: 128024
special token: 128025
special token: 128026
special token: 128027
special token: 128028
special token: 128029
special token: 128030
special token: 128031
special token: 128032
special token: 128033
special token: 128034
special token: 128035
special token: 128036
special token: 128037
special token: 128038
special token: 128039
special token: 128040
special token: 128041
special token: 128042
special token: 128043
special token: 128044
special token: 128045
special token: 128046
special token: 128047
special token: 128048
special token: 128049
special token: 128050
special token: 128051
special token: 128052
special token: 128053
special token: 128054
special token: 128055
special token: 128056
special token: 128057
special token: 128058
special token: 128059
special token: 128060
special token: 128061
special token: 128062
special token: 128063
special token: 128064
special token: 128065
special token: 128066
special token: 128067
special token: 128068
special token: 128069
special token: 128070
special token: 128071
special token: 128072
special token: 128073
special token: 128074
special token: 128075
special token: 128076
special token: 128077
special token: 128078
special token: 128079
special token: 128080
special token: 128081
special token: 128082
special token: 128083
special token: 128084
special token: 128085
special token: 128086
special token: 128087
special token: 128088
special token: 128089
special token: 128090
special token: 128091
special token: 128092
special token: 128093
special token: 128094
special token: 128095
special token: 128096
special token: 128097
special token: 128098
special token: 128099
special token: 128100
special token: 128101
special token: 128102
special token: 128103
special token: 128104
special token: 128105
special token: 128106
special token: 128107
special token: 128108
special token: 128109
special token: 128110
special token: 128111
special token: 128112
special token: 128113
special token: 128114
special token: 128115
special token: 128116
special token: 128117
special token: 128118
special token: 128119
special token: 128120
special token: 128121
special token: 128122
special token: 128123
special token: 128124
special token: 128125
special token: 128126
special token: 128127
special token: 128128
special token: 128129
special token: 128130
special token: 128131
special token: 128132
special token: 128133
special token: 128134
special token: 128135
special token: 128136
special token: 128137
special token: 128138
special token: 128139
special token: 128140
special token: 128141
special token: 128142
special token: 128143
special token: 128144
special token: 128145
special token: 128146
special token: 128147
special token: 128148
special token: 128149
special token: 128150
special token: 128151
special token: 128152
special token: 128153
special token: 128154
special token: 128155
special token: 128156
special token: 128157
special token: 128158
special token: 128159
special token: 128160
special token: 128161
special token: 128162
special token: 128163
special token: 128164
special token: 128165
special token: 128166
special token: 128167
special token: 128168
special token: 128169
special token: 128170
special token: 128171
special token: 128172
special token: 128173
special token: 128174
special token: 128175
special token: 128176
special token: 128177
special token: 128178
special token: 128179
special token: 128180
special token: 128181
special token: 128182
special token: 128183
special token: 128184
special token: 128185
special token: 128186
special token: 128187
special token: 128188
special token: 128189
special token: 128190
special token: 128191
special token: 128192
special token: 128193
special token: 128194
special token: 128195
special token: 128196
special token: 128197
special token: 128198
special token: 128199
special token: 128200
special token: 128201
special token: 128202
special token: 128203
special token: 128204
special token: 128205
special token: 128206
special token: 128207
special token: 128208
special token: 128209
special token: 128210
special token: 128211
special token: 128212
special token: 128213
special token: 128214
special token: 128215
special token: 128216
special token: 128217
special token: 128218
special token: 128219
special token: 128220
special token: 128221
special token: 128222
special token: 128223
special token: 128224
special token: 128225
special token: 128226
special token: 128227
special token: 128228
special token: 128229
special token: 128230
special token: 128231
special token: 128232
special token: 128233
special token: 128234
special token: 128235
special token: 128236
special token: 128237
special token: 128238
special token: 128239
special token: 128240
special token: 128241
special token: 128242
special token: 128243
special token: 128244
special token: 128245
special token: 128246
special token: 128247
special token: 128248
special token: 128249
special token: 128250
special token: 128251
special token: 128252
special token: 128253
special token: 128254
special token: 128255
special token: 128256


token = 128006
token = 9125
token = 128007
token = 271
token = 2675
token = 527
token = 264
token = 11190
token = 18328
token = 128009
token = 128006
token = 882
token = 128007
token = 198
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271


token = 128006
token = 9125
token = 128007
token = 271
token = 2675
token = 527
token = 264
token = 11190
token = 18328
token = 128009
token = 128006
token = 882
token = 128007
token = 198
token = 3923
token = 374
token = 279
token = 469
token = 3168
token = 301
token = 22703
token = 30
token = 128009
token = 128006
token = 78191
token = 128007
token = 271

build_inp_KQ_mask n_kv: 32, n_tokens: 28

Ibuild_inp_KQ_mask n_kv: 32, n_tokens: 1
 thinkbuild_inp_KQ_mask n_kv: 32, n_tokens: 1
 youbuild_inp_KQ_mask n_kv: 32, n_tokens: 1
 meantbuild_inp_KQ_mask n_kv: 32, n_tokens: 1
 tobuild_inp_KQ_mask n_kv: 64, n_tokens: 1
 askbuild_inp_KQ_mask n_kv: 64, n_tokens: 1
 "build_inp_KQ_mask n_kv: 64, n_tokens: 1
Whatbuild_inp_KQ_mask n_kv: 64, n_tokens: 1
 isbuild_inp_KQ_mask n_kv: 64, n_tokens: 1
 thebuild_inp_KQ_mask n_kv: 64, n_tokens: 1
 Ebuild_inp_KQ_mask n_kv: 64, n_tokens: 1
iffbuild_inp_KQ_mask n_kv: 64, n_tokens: 1
elbuild_inp_KQ_mask n_kv: 64, n_tokens: 1
 Towerbuild_inp_KQ_mask n_kv: 64, n_tokens: 1
"?build_inp_KQ_mask n_kv: 64, n_tokens: 1


build_inp_KQ_mask n_kv: 32, n_tokens: 28


Ollama without merges:
token = 128000
token = 128006
token = 82
token = 88
token = 82
token = 83
token = 68
token = 76
token = 128007
token = 198
token = 198
token = 56
token = 78
token = 84
token = 220
token = 64
token = 81
token = 68
token = 220
token = 64
token = 220
token = 71
token = 68
token = 75
token = 79
token = 69
token = 84
token = 75
token = 220
token = 64
token = 82
token = 82
token = 72
token = 82
token = 83
token = 64
token = 77
token = 83
token = 128009
token = 128006
token = 84
token = 82
token = 68
token = 81
token = 128007
token = 198
token = 198
token = 54
token = 71
token = 64
token = 83
token = 220
token = 72
token = 82
token = 220
token = 83
token = 71
token = 68
token = 220
token = 36
token = 72
token = 69
token = 69
token = 68
token = 75
token = 220
token = 51
token = 78
token = 86
token = 68
token = 81
token = 30
token = 128009
token = 128006
token = 64
token = 82
token = 82
token = 72
token = 82
token = 83
token = 64
token = 77
token = 83
token = 128007
token = 198
token = 198
The Eiffel Tower is an iconic iron lattice tower located in Paris, France. It was built for the 1889 World's Fair and was^C

token = 128000
token = 128006
token = 82
token = 88
token = 82
token = 83
token = 68
token = 76
token = 128007
token = 198
token = 198
token = 56
token = 78
token = 84
token = 220
token = 64
token = 81
token = 68
token = 220
token = 64
token = 220
token = 71
token = 68
token = 75
token = 79
token = 69
token = 84
token = 75
token = 220
token = 64
token = 82
token = 82
token = 72
token = 82
token = 83
token = 64
token = 77
token = 83
token = 128009
token = 128006
token = 84
token = 82
token = 68
token = 81
token = 128007
token = 198
token = 198
token = 54
token = 71
token = 64
token = 83
token = 220
token = 72
token = 82
token = 220
token = 83
token = 71
token = 68
token = 220
token = 36
token = 72
token = 69
token = 69
token = 68
token = 75
token = 220
token = 51
token = 78
token = 86
token = 68
token = 81
token = 30
token = 128009
token = 128006
token = 64
token = 82
token = 82
token = 72
token = 82
token = 83
token = 64
token = 77
token = 83
token = 128007
token = 198
token = 198
[New Thread 0x7fffa7000000 (LWP 409516)]
[New Thread 0x7fffa3e00000 (LWP 409517)]
[New Thread 0x7fffa3400000 (LWP 409518)]
# Import the necessary libraries
import numpy as np
import numpy as np


128000 <|begin_of_text|>: (llama.cpp)
token_embd.weight tensor type: f32
token_embd.weight backend type: CPU
token_embd.weight[0] = 0.000277
token_embd.weight[1] = -0.000519
token_embd.weight[2] = -0.000565
token_embd.weight[3] = 0.001350
token_embd.weight[4] = 0.000102
token_embd.weight[5] = -0.000874
token_embd.weight[6] = 0.007507
token_embd.weight[7] = -0.002716
token_embd.weight[8] = -0.000568
token_embd.weight[9] = -0.000437


token_embd.weight tensor type: q4_K
token_embd.weight backend type: CPU
token_embd.weight[0] = 0.000000
token_embd.weight[1] = 28544006904348672.000000
token_embd.weight[2] = 7029865031139328.000000
token_embd.weight[3] = 0.000000
token_embd.weight[4] = -0.000000
token_embd.weight[5] = -0.000000
token_embd.weight[6] = -0.000000
token_embd.weight[7] = 0.000000
token_embd.weight[8] = -397538935281706006608267444224.000000
token_embd.weight[9] = -0.000000


Embedding for token 128006:
Embedding shape: (4096,)
First 10 values:
[0] = -0.000220716
[1] = -0.000220716
[2] = 9.92417e-05
[3] = -0.000540674
[4] = 0.000259221
[5] = -6.07371e-05
[6] = -0.000220716
[7] = 0.00105911
[8] = 9.92417e-05
[9] = 0.000579178

token_embd.weight tensor type: f32
token_embd.weight backend type: CPU
token_embd.weight[0] = -0.000184
token_embd.weight[1] = -0.000240
token_embd.weight[2] = 0.000164
token_embd.weight[3] = -0.000538
token_embd.weight[4] = 0.000265
token_embd.weight[5] = -0.000013
token_embd.weight[6] = -0.000290
token_embd.weight[7] = 0.001060
token_embd.weight[8] = 0.000037
token_embd.weight[9] = 0.000530


Embedding for token 128000:
Embedding shape: (4096,)
First 10 values:
[0] = 0
[1] = 0
[2] = 0
[3] = 0
[4] = 0
[5] = 0
[6] = 0.00894463
[7] = 0
[8] = 0
[9] = 0

token_embd.weight tensor type: f32
token_embd.weight backend type: CPU
token_embd.weight[0] = 0.000277
token_embd.weight[1] = -0.000519
token_embd.weight[2] = -0.000565
token_embd.weight[3] = 0.001350
token_embd.weight[4] = 0.000102
token_embd.weight[5] = -0.000874
token_embd.weight[6] = 0.007507
token_embd.weight[7] = -0.002716
token_embd.weight[8] = -0.000568
token_embd.weight[9] = -0.000437


Q4_K:
token_embd.weight backend type: CPU
token_embd.weight[0] = 0.000277
token_embd.weight[1] = -0.000519
token_embd.weight[2] = -0.000565
token_embd.weight[3] = 0.001350
token_embd.weight[4] = 0.000102
token_embd.weight[5] = -0.000874
token_embd.weight[6] = 0.007507
token_embd.weight[7] = -0.002716
token_embd.weight[8] = -0.000568
token_embd.weight[9] = -0.000437


token_embd.weight backend type: CPU
token_embd.weight[0] = 0.000000
token_embd.weight[1] = 28544006904348672.000000
token_embd.weight[2] = 7029865031139328.000000
token_embd.weight[3] = 0.000000
token_embd.weight[4] = -0.000000
token_embd.weight[5] = -0.000000
token_embd.weight[6] = -0.000000
token_embd.weight[7] = 0.000000
token_embd.weight[8] = -397538935281706006608267444224.000000
token_embd.weight[9] = -0.000000


### Self-Attention
Now in LLama 3.2 Vision Instruct there are two models and the both have
different configuration settings. For example, the vision model has an 
attention head count of 16:
```console
39: UINT32     |        1 | vision.attention.head_count = 16
```
While the language model has an attention head count of 12:
```console
18: UINT32     |        1 | mllama.attention.head_count = 32
19: UINT32     |        1 | mllama.attention.head_count_kv = 8
```
So lets understand what the query matrix for the language model looks like:
```console
520:   16777216 |  4096,  4096,     1,     1 | F16     | blk.0.attn_q.weight
517:    4194304 |  4096,  1024,     1,     1 | F16     | blk.0.attn_k.weight


                 Q                                    K
     0  [0              4095]          0  [0             4095]
        ...                               ...
        ...                        1023   [0             4095]
        ...
  4095  [0              4095]
```
So have 32 heads with 128 dimensions in each. And we have 8 KV heads so groups
of 4 heads share the same KV matrix.
```
Q Heads:  [Q1 Q2 Q3 Q4] [Q5 Q6 Q7 Q8]  ... [Q29 Q30 Q31 Q32]
K/V Heads:    [K1/V1]       [K2/V2]    ...      [K8/V8]
```

```console
(gdb) p hparams.n_head(0)
$20 = 32
(gdb) p hparams.n_head_kv(0)
$21 = 8
(gdb) p hparams.n_gqa(0)
$22 = 4
(gdb) p hparams.n_embd_k_gqa(0)
$23 = 1024
(gdb) p hparams.n_gqa(0)
$24 = 4
```

So in the conversion we have the following
```console
n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        print("n_head:", n_head)
        print("n_kv_head:", n_kv_head)

        # Permute the cross-attention Q and K tensors
        if name.startswith('blk.') and name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        if name.startswith('blk.') and name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
```
```console
517:    4194304 |  4096,  1024,     1,     1 | F16     | blk.0.attn_k.weight

n_head: 32
n_kv_head: 8
                K
     [0                     4095
     ...
     ...
1023 [0                     4095

data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
data_torch = LlamaModel.permute(data_torch,     32, 8) 

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))
```
So in this case both n_head and n_head_kv will be set to 8:
```console
n_head: 32, n_head_kv: 8
n_head: 8, n_head_kv: 8
weights.shape: torch.Size([1024, 4096])

        return (weights.reshape(8, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

weights.shape[0] = 1024
n_head = 8
weights.shape[0] // n_head // 2: 64
*weights.shape[1:] = 4096

n_head: 32, n_head_kv: 8
n_head: 8, n_head_kv: 8
weights.shape: torch.Size([1024, 4096])
4096
weights.shape[0] // n_head // 2: 64
temp_tensor.shape: torch.Size([8, 2, 64, 4096])
temp_tensor.shape: torch.Size([8, 64, 2, 4096])
temp_tensor.shape: torch.Size([1024, 4096])
```
So rehape are just view operations that update the strides of a new tensor
but the underlying tensors data is not changed. swapaxis (or ggml_cont in ggml)
creates a new tensor with rearranged data.

```console
5:    1638400 |  1280,  1280,     1,     1 | F16     | v.enc.blk.0.attn_k.weight
6:    1638400 |  1280,  1280,     1,     1 | F16     | v.enc.blk.0.attn_out.weight
7:       1280 |  1280,     1,     1,     1 | F32     | v.enc.blk.0.attn_out_norm.bias
8:       1280 |  1280,     1,     1,     1 | F32     | v.enc.blk.0.attn_out_norm.weight
9:    1638400 |  1280,  1280,     1,     1 | F16     | v.enc.blk.0.attn_q.weight
10:    1638400|  1280,  1280,     1,     1 | F16     | v.enc.blk.0.attn_v.weight
11:      1280 |  1280,     1,     1,     1 | F32     | v.enc.blk.0.ffn_down.bias
12:    6553600 |  5120,  1280,     1,     1 | F16     | v.enc.blk.0.ffn_down.weight
13:       5120 |  5120,     1,     1,     1 | F32     | v.enc.blk.0.ffn_up.bias
14:    6553600 |  1280,  5120,     1,     1 | F16     | v.enc.blk.0.ffn_up.weight
15:       1280 |  1280,     1,     1,     1 | F32     | v.enc.blk.0.input_norm.bias
16:       1280 |  1280,     1,     1,     1 | F32     | v.enc.blk.0.input_norm.weight

393:          1 |     1,     1,     1,     1 | F32     | v.enc.global.blk.0.attn_gate
394:    1638400 |  1280,  1280,     1,     1 | F16     | v.enc.global.blk.0.attn_k.weight
395:       1280 |  1280,     1,     1,     1 | F32     | v.enc.global.blk.0.attn_norm.bias
396:       1280 |  1280,     1,     1,     1 | F32     | v.enc.global.blk.0.attn_norm.weight
397:    1638400 |  1280,  1280,     1,     1 | F16     | v.enc.global.blk.0.attn_out.weight
398:    1638400 |  1280,  1280,     1,     1 | F16     | v.enc.global.blk.0.attn_q.weight
399:    1638400 |  1280,  1280,     1,     1 | F16     | v.enc.global.blk.0.attn_v.weight
400:       1280 |  1280,     1,     1,     1 | F32     | v.enc.global.blk.0.ffn_down.bias
401:    6553600 |  5120,  1280,     1,     1 | F16     | v.enc.global.blk.0.ffn_down.weight
402:          1 |     1,     1,     1,     1 | F32     | v.enc.global.blk.0.ffn_gate
403:       5120 |  5120,     1,     1,     1 | F32     | v.enc.global.blk.0.ffn_up.bias
404:    6553600 |  1280,  5120,     1,     1 | F16     | v.enc.global.blk.0.ffn_up.weight
405:       1280 |  1280,     1,     1,     1 | F32     | v.enc.global.blk.0.post_attn_norm.bias
406:       1280 |  1280,     1,     1,     1 | F32     | v.enc.global.blk.0.post_attn_norm.weight

39: UINT32     |        1 | vision.attention.head_count = 16
```
```console
n_embd: 1280,
n_layer: 32,
n_global_layer: 8,
max_pos_embd: 1601,
n_channel: 3,
patch_size: 14,
n_proj_dim: 7680,
n_intermediate: 5120,
v_n_embd: 1280

vision mllama_build_image_graph: n_head = 16, d_head = 80, hidden_size = 1280, n_positions = 1601, n_patches = 1600
llama_encode_vision_internal: ctx.out_embd: 104923136
```



ubatch.embd[0] = 9.647850
ubatch.embd[1] = 12.615902
ubatch.embd[2] = -4.295709
ubatch.embd[3] = 6.917406
ubatch.embd[4] = -1.574495
ubatch.embd[5] = -13.777966
ubatch.embd[6] = -1.115831
ubatch.embd[7] = 2.098805
ubatch.embd[8] = -7.775772
ubatch.embd[9] = -1.934765
ca_patch_embd tensor type: f32
ca_patch_embd backend type: CPU
ca_patch_embd[0] = 9.647850
ca_patch_embd[1] = 12.615902
ca_patch_embd[2] = -4.295709
ca_patch_embd[3] = 6.917406
ca_patch_embd[4] = -1.574495
ca_patch_embd[5] = -13.777966
ca_patch_embd[6] = -1.115831
ca_patch_embd[7] = 2.098805
ca_patch_embd[8] = -7.775772
ca_patch_embd[9] = -1.934765


ubatch.embd[0] = 9.647850
ubatch.embd[1] = 12.615902
ubatch.embd[2] = -4.295709
ubatch.embd[3] = 6.917406
ubatch.embd[4] = -1.574495
ubatch.embd[5] = -13.777966
ubatch.embd[6] = -1.115831
ubatch.embd[7] = 2.098805
ubatch.embd[8] = -7.775772
ubatch.embd[9] = -1.934765
ca_patch_embd tensor type: f32
ca_patch_embd backend type: CPU
ca_patch_embd[0] = 9.647850
ca_patch_embd[1] = 12.615902
ca_patch_embd[2] = -4.295709
ca_patch_embd[3] = 6.917406
ca_patch_embd[4] = -1.574495
ca_patch_embd[5] = -13.777966
ca_patch_embd[6] = -1.115831
ca_patch_embd[7] = 2.098805
ca_patch_embd[8] = -7.775772
ca_patch_embd[9] = -1.934765


ubatch.embd[0] = 9.647850
ubatch.embd[1] = 12.615902
ubatch.embd[2] = -4.295709
ubatch.embd[3] = 6.917406
ubatch.embd[4] = -1.574495
ubatch.embd[5] = -13.777966
ubatch.embd[6] = -1.115831
ubatch.embd[7] = 2.098805
ubatch.embd[8] = -7.775772
ubatch.embd[9] = -1.934765
ca_patch_embd tensor type: f32
ca_patch_embd backend type: CPU
ca_patch_embd[0] = 9.647850
ca_patch_embd[1] = 12.615902
ca_patch_embd[2] = -4.295709
ca_patch_embd[3] = 6.917406
ca_patch_embd[4] = -1.574495
ca_patch_embd[5] = -13.777966
ca_patch_embd[6] = -1.115831
ca_patch_embd[7] = 2.098805
ca_patch_embd[8] = -7.775772
ca_patch_embd[9] = -1.93476


(venv) $ sha256sum image_patch_embeddings.bin
319cc0572866e3b165d5f59dc3e5709b87ec503ff6af10e3cd487d21e2ad17ab  image_patch_embeddings.bin

(venv) $ sha256sum image_patch_embeddings.bin 
319cc0572866e3b165d5f59dc3e5709b87ec503ff6af10e3cd487d21e2ad17ab  image_patch_embeddings.bin

Before vision encoder:
(gdb) p ctx.sched.get().graph
$7 = {size = 2010, n_nodes = 1148, n_leafs = 361, nodes = 0x555556a0b360, grads = 0x0, grad_accs = 0x0,
  leafs = 0x555556723ce0, visited_hash_set = {size = 0, used = 0x0, keys = 0x0},
  order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT}


After vision encoder:
(gdb) p ctx.sched.get().graph
$8 = {size = 4353, n_nodes = 1699, n_leafs = 517, nodes = 0x555556ab75b0, grads = 0x0, grad_accs = 0x0,
  leafs = 0x55555676d440, visited_hash_set = {size = 0, used = 0x0, keys = 0x0},
  order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT}

After next decode:
  (gdb) p ctx.sched.get().graph
$9 = {size = 4353, n_nodes = 1312, n_leafs = 457, nodes = 0x555556ab75b0, grads = 0x0, grad_accs = 0x0,
  leafs = 0x55555676d440, visited_hash_set = {size = 0, used = 0x0, keys = 0x0},
  order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT}


### Sampling
```
source=runner.go:128 msg="[danbev] --------------> sampling params: "
sparams="&{
TopK:40
TopP:0.9
MinP:0
TfsZ:1
TypicalP:1
Temp:0 RepeatLastN:64
PenaltyRepeat:1.1
PenaltyFreq:0
PenaltyPresent:0
Mirostat:0
MirostatTau:5
MirostatEta:0.1
PenalizeNl:true
Seed:4294967295 Grammar:}"
```

