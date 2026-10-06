## Diarization in whisper.cpp
This document contains notes investigating porting
https://huggingface.co/blog/nvidia/nemotron-diarization to whisper.cpp


### Running original Nemotron diarization

```console
$ cd audio/diarization
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

```console
(venv) $ python src/nemotron-diarization.py 
Loading weights: 100%|██████████████████████████████████████████| 417/417 [00:00<00:00, 10514.69it/s]
Input features shape: torch.Size([1, 521, 128])
Output logits shape: torch.Size([1, 521, 8])
 Frame |    Time | Speaker Probabilities (Channels 0 to 7)
---------------------------------------------------------------------------------------------------------
    25 |   0.25s | 0.00361585  0.00000386  0.00000072  0.00000023  0.00000027  0.00000002  0.00000000  0.00000000
    26 |   0.26s | 0.00564652  0.00000526  0.00000103  0.00000034  0.00000043  0.00000004  0.00000000  0.00000000
    27 |   0.27s | 0.00857463  0.00000786  0.00000167  0.00000054  0.00000072  0.00000007  0.00000000  0.00000000
    28 |   0.28s | 0.01498830  0.00001288  0.00000299  0.00000098  0.00000126  0.00000016  0.00000000  0.00000000
    29 |   0.29s | 0.03030495  0.00002153  0.00000629  0.00000187  0.00000238  0.00000036  0.00000000  0.00000000
    30 |   0.30s | 0.06344025  0.00003482  0.00001442  0.00000470  0.00000578  0.00000091  0.00000001  0.00000000
    31 |   0.31s | 0.27481452  0.00005802  0.00003354  0.00001167  0.00001581  0.00000187  0.00000003  0.00000000
    32 |   0.32s | 0.70629668  0.00007214  0.00003553  0.00001548  0.00003191  0.00001723  0.00000148  0.00000007
    33 |   0.33s | 0.95494950  0.00006036  0.00004219  0.00002230  0.00004054  0.00002776  0.00000526  0.00000038
    34 |   0.34s | 0.98821682  0.00005999  0.00005002  0.00002573  0.00004131  0.00003218  0.00001350  0.00000134
    35 |   0.35s | 0.99554729  0.00005256  0.00005111  0.00003040  0.00004307  0.00003399  0.00002172  0.00000295
    36 |   0.36s | 0.99800223  0.00004736  0.00005098  0.00003456  0.00004037  0.00003122  0.00002493  0.00000419
    37 |   0.37s | 0.99892443  0.00004242  0.00004945  0.00003566  0.00003733  0.00002842  0.00002756  0.00000555
    38 |   0.38s | 0.99932444  0.00003927  0.00004804  0.00003728  0.00003439  0.00002567  0.00002845  0.00000669
    39 |   0.39s | 0.99950349  0.00003772  0.00004546  0.00003595  0.00002910  0.00002181  0.00002588  0.00000697
    40 |   0.40s | 0.99950981  0.00001408  0.00002555  0.00003388  0.00004031  0.00003250  0.00006345  0.00002751
    41 |   0.41s | 0.99956101  0.00001405  0.00002517  0.00003192  0.00003557  0.00003037  0.00006277  0.00002944
    42 |   0.42s | 0.99958378  0.00001340  0.00002416  0.00002957  0.00003105  0.00002905  0.00005959  0.00002930
    43 |   0.43s | 0.99959558  0.00001216  0.00002295  0.00002708  0.00002744  0.00002679  0.00005679  0.00002911
    44 |   0.44s | 0.99961472  0.00001096  0.00002232  0.00002599  0.00002483  0.00002537  0.00005386  0.00002846
Speaker 0: 0.32s - 2.75s
Speaker 1: 3.18s - 5.19s
```
So each frame here is 10ms, and we have 8 probabilities, one for each speaker.


### Model
The transformers model used in the above example looks like this:
```console
Nemotron3DiarizationForAudioFrameClassification(
  (model): Nemotron3DiarizationModel(
    (audio_tower): Nemotron3DiarizationAudioModel(
      (embedder): Nemotron3DiarizationFeatureStacking(
        (projection): Linear(in_features=1024, out_features=512, bias=False)
      )
      (input_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True, bias=True)
      (layers): ModuleList(
        (0-30): 31 x Nemotron3DiarizationAudioLayer(
          (self_attn): Nemotron3DiarizationAttention(
            (q_proj): Linear(in_features=512, out_features=512, bias=False)
            (k_proj): Linear(in_features=512, out_features=512, bias=False)
            (v_proj): Linear(in_features=512, out_features=512, bias=False)
            (o_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (layer_norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True, bias=True)
          (mlp): Nemotron3DiarizationMLP(
            (activation_fn): GELUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
          )
          (layer_norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True, bias=True)
        )
      )
      (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True, bias=True)
      (rotary_emb): Nemotron3DiarizationRotaryEmbedding()
    )
    (proj): Linear(in_features=512, out_features=192, bias=True)
    (upsampler): Nemotron3DiarizationSubpixelUpsampler(
      (conv): Conv1d(192, 1536, kernel_size=(3,), stride=(1,), padding=(1,))
    )
  )
  (classifier): Nemotron3DiarizationClassificationHead(
    (dense): Linear(in_features=192, out_features=192, bias=True)
    (out_proj): Linear(in_features=192, out_features=8, bias=True)
    (act_fn): ReLU()
  )
)
```
The sources for this model can be found in:
```console
(venv) $ ls venv/lib/python3.12/site-packages/transformers/models/nemotron3_diarization/
__init__.py
configuration_nemotron3_diarization.py
modeling_nemotron3_diarization.py
convert_nemotron3_diarization_to_hf.py
modular_nemotron3_diarization.py
processing_nemotron3_diarization.py
```
The model can be downloaded using:
```console
(venv) $ hf download nvidia/Nemotron-3-Diarization --local-dir Nemotron-3-Diarization
```
```console
(venv) $ ls Nemotron-3-Diarization/
ASR_INTEGRATION_GUIDE.md   model.safetensors                 processor_config.json
bias.md                    Nemotron-3-Diarization.nemo       README.md
config.json                Nemotron-3-Diarization.q8_0.gguf  safety.md
diarization_evaluation.md  nemotron3_tts_8_open_voices.mp4   streaming_diarization_demo.gif
explainability.md          privacy.md
```
So this have a safetensors file, a .nemo file (just like parakeet) and also
a gguf which is intresting.

So is seems like the .gguf file is intented to be used with
[Nemo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) which states that
Speaker diarization: Streaming Sortformer 4-speaker v2 and Nemotron 3 Diarization,
standalone or combined with ASR.

```console
(venv) $ gguf-dump Nemotron-3-Diarization/Nemotron-3-Diarization.q8_0.gguf
INFO:gguf-dump:* Loading: Nemotron-3-Diarization/Nemotron-3-Diarization.q8_0.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
* Dumping 60 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 360
      3: UINT64     |        1 | GGUF.kv_count = 57
      4: STRING     |        1 | general.architecture = 'sortformer'
      5: STRING     |        1 | general.name = 'Nemotron-3-Diarization.q8_0'
      6: STRING     |        1 | sortformer.version = 'v3'
      7: UINT32     |        1 | general.file_type = 7
      8: UINT32     |        1 | sortformer.encoder.d_model = 512
      9: UINT32     |        1 | sortformer.encoder.n_layers = 31
     10: UINT32     |        1 | sortformer.encoder.n_heads = 8
     11: UINT32     |        1 | sortformer.encoder.d_ff = 2048
     12: UINT32     |        1 | sortformer.encoder.conv_kernel_size = 0
     13: UINT32     |        1 | sortformer.encoder.subsampling_factor = 8
     14: UINT32     |        1 | sortformer.encoder.subsampling_conv_channels = 256
     15: UINT32     |        1 | sortformer.encoder.feat_in = 128
     16: BOOL       |        1 | sortformer.encoder.xscaling = False
     17: BOOL       |        1 | sortformer.encoder.use_bias = True
     18: UINT32     |        1 | sortformer.encoder.pos_emb_max_len = 5000
     19: STRING     |        1 | sortformer.encoder.conv_norm = 'batch_norm'
     20: STRING     |        1 | sortformer.encoder.conv_context = 'symmetric'
     21: STRING     |        1 | sortformer.encoder.att_context_style = 'regular'
     22: STRING     |        1 | sortformer.encoder.type = 'transformer_rope'
     23: STRING     |        1 | sortformer.encoder.subsampling_type = 'feature_stacking'
     24: BOOL       |        1 | sortformer.encoder.qkv_bias = False
     25: BOOL       |        1 | sortformer.encoder.qk_norm = False
     26: BOOL       |        1 | sortformer.encoder.pre_block_norm = True
     27: FLOAT32    |        1 | sortformer.encoder.rope_base = 10000.0
     28: FLOAT32    |        1 | sortformer.encoder.rotary_fraction = 1.0
     29: UINT32     |        1 | sortformer.transformer.n_layers = 0
     30: UINT32     |        1 | sortformer.transformer.hidden_size = 192
     31: UINT32     |        1 | sortformer.transformer.inner_size = 0
     32: UINT32     |        1 | sortformer.transformer.n_heads = 0
     33: BOOL       |        1 | sortformer.transformer.pre_ln = True
     34: UINT32     |        1 | sortformer.num_speakers = 8
     35: BOOL       |        1 | sortformer.high_resolution = True
     36: UINT32     |        1 | sortformer.output_subsampling_factor = 1
     37: UINT32     |        1 | sortformer.upsample_factor = 8
     38: BOOL       |        1 | sortformer.learnable_silence = True
     39: UINT32     |        1 | sortformer.preprocessor.sample_rate = 16000
     40: FLOAT32    |        1 | sortformer.preprocessor.window_size = 0.02500000037252903
     41: FLOAT32    |        1 | sortformer.preprocessor.window_stride = 0.009999999776482582
     42: UINT32     |        1 | sortformer.preprocessor.n_fft = 512
     43: UINT32     |        1 | sortformer.preprocessor.features = 128
     44: STRING     |        1 | sortformer.preprocessor.normalize = 'NA'
     45: FLOAT32    |        1 | sortformer.preprocessor.preemph = 0.9700000286102295
     46: FLOAT32    |        1 | sortformer.preprocessor.dither = 9.999999747378752e-06
     47: FLOAT32    |        1 | sortformer.preprocessor.log_zero_guard = 5.960464477539063e-08
     48: UINT32     |        1 | sortformer.scoring.spkcache_sil_frames_per_spk = 1
     49: FLOAT32    |        1 | sortformer.scoring.pred_score_threshold = 0.25
     50: FLOAT32    |        1 | sortformer.scoring.scores_boost_latest = 0.05000000074505806
     51: FLOAT32    |        1 | sortformer.scoring.sil_threshold = 0.20000000298023224
     52: FLOAT32    |        1 | sortformer.scoring.strong_boost_rate = 0.75
     53: FLOAT32    |        1 | sortformer.scoring.weak_boost_rate = 1.5
     54: FLOAT32    |        1 | sortformer.scoring.min_pos_scores_rate = 0.5
     55: UINT32     |        1 | sortformer.streaming.spkcache_len = 264
     56: UINT32     |        1 | sortformer.streaming.fifo_len = 0
     57: UINT32     |        1 | sortformer.streaming.chunk_len = 264
     58: UINT32     |        1 | sortformer.streaming.spkcache_update_period = 264
     59: UINT32     |        1 | sortformer.streaming.chunk_left_context = 0
     60: UINT32     |        1 | sortformer.streaming.chunk_right_context = 0
...
```
So there is a conversion script that is provided in
NeMo-Speech.cpp/conversion/diarization.py which I believe is what produced
the above .gguf.


### SortFormer (Sorting Transformer)
TODO:


### Background information
So lets say we have a sample audio which is of length 83200 samples which which
contains two speakers.

This will be divided into 160 frames, because the audio is in 16kHz and we use
a 10ms stride so 16000*0.01 = 160 samples per frame.

```console
83200 / 160 + 1 = 521
```
So we will have 521 frames of mel frames for this specific audio file/sample.

So the "token/patch" in a sequence will initially be 1024 dimensions.
Processing is done in chunks of frames, where we have 13 encoder frames.

So processing will begin with a chunk begin converted to mel-spectrogram and then
passed to the encoder as an input tensor:
```console
(gdb) p input->ne
$4 = {1024, 14, 1, 1}

0  [0  ... 1023]
1  [0  ... 1023]
        .
        .
        .
13 [0  ... 1023]
```
So we have 14 frames of 1024 mel-spectrogram features. This is 1024 because
we have 128 mel bins and 8 of them per frame (128*8=1024). This is in the frequency
domain still. This will projected down into the models 512 dimensional hidden
space where we will then be dealing with abstract features and no longer
frequencies.

```c++
    ggml_tensor * hidden = ggml_mul_mat(ctx, model.enc_pre_w, input);
```

```console
(gdb) p model.enc_pre_w->ne
$3 = {1024, 512, 1, 1}

(gdb) p hidden->ne
$5 = {512, 14, 1, 1}
```

If we have a prefix, that is a cache consisting of speaker cache and the content
of the fifo vector, then it will get prepended to the current tensor and become
the input to the model:
```c++
    ggml_tensor * cache = nullptr;
    if (prefix) {
        cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, prefix);
        ggml_set_input(cache);
        ggml_set_name(cache, "cache");
        cur = ggml_concat(ctx, cache, cur, 1);
    }
```
More on the cache in a separate section below.


Then we have the positions (frames/time):
```c++
    auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, time);
    ggml_set_input(positions);
    ggml_set_name(positions, "positions");
```

```console
(gdb) p positions->ne
$3 = {14, 1, 1, 1}
```
We then have a normalization before the layers of the model will be run.
This model has 31 layers:
```console
(gdb) p model.hparams.n_audio_layer
$4 = 31
```
```c++
    struct ggml_tensor * inpL = cur;

    for (int i = 0; i < model.hparams.n_audio_layer; ++i) {
        const auto & layer = model.layers[i];

```
Each layer has it own normalization that happens first before the self-attention
which is bi-directional. So at this point we have a sequence of frames where which
contain abstract hidden vectors in the models internal vector space. So we are
going to perform dot product operations between all of these to see which of
them are simliar. And recall that we are not really interested in "what" is
being said, but "who" said what. Before the attention we have the projections
of Q and K which filters out the phonemes (the actual words/vowels) and instead
amplify the other characteristics like tibre, pitch, harmnonics, and vocal-tract
geometry. So when dot product belong to the same human vocal tract their dot
product will be a large positive value, and where they are from different humans
the dot product will be a small or negative value.

And after that we multiply the dot product scores by V.

And that happens for all 31 layers in the model.

After the layers have been processed (self-attention and ffn) we have the
following:
```console
(gdb) p cur->ne
$1 = {512, 14, 1, 1}
```
We currently have 512 dimension which was required by the self-attention
but to output 8 speaker probabilities we don't need all of those dimensions.
The following down projection will project this down to 192 dimensions.
```c++
    cur = ggml_mul_mat(ctx, model.enc_proj_w, cur);
    cur = ggml_add(ctx, cur, model.enc_proj_b);
```
```console
(gdb) p cur->ne
$2 = {192, 14, 1, 1}
```

The next step is a convolution of the frames, and for that we have to transpose
the tensor:
```c++
    // transpose for convolution
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
```

```console
(gdb) p cur->ne
$3 = {14, 192, 1, 1}
```
So that is the input that the convolution will operate on and this is the the
kernel:
```console
(gdb) p model.upsample_w->ne
$4 = {3, 192, 1536, 1}
```
```c++
    cur = ggml_conv_1d(ctx, model.upsample_w, cur, 1, 1, 1);
```

So we have something like this:
```console
(gdb) p cur->ne
$3 = {14, 192, 1, 1}
      ↑    ↑
  frames   features/channels

    time->
0   [0 ... 13]
1   [0 ... 13]
        .
        .
        .
192 [0 ... 13]

(gdb) p model.upsample_w->ne     (kernel)
$1 = {3, 192, 1536, 1}

3 times steps wide
192 channels deep
1536 distinct filters

0       time->
    0   [0 1 2]
    1   [0 1 2]
           .
           .
           .
    191 [0 1 2]

...

1535    time->
    0   [0 1 2]
    1   [0 1 2]
           .
           .
           .
    191 [0 1 2]
```
So we have 1536 [3, 192] filters/kernels which we apply to the [14, 192] matrix
so each will "group" three time steps and produce one output value.
```console
cur:

0   [0 1 2 ... 13]
1   [0 1 2 ... 13]
2   [0 1 2 ... 13]
        .
192 [0 1 2 ... 13]

Apply kernel to t0:
0   [x x x ... 13]
1   [x x x ... 13]
2   [x x x ... 13]
        .
192 [x x x ... 13]

3*192 = 576 values
```
Computes a weighted sum over all 576 value to produce 1 scalar. And we have
1536 of these so we get 1536 values. So each filter is looking at 3 time frames
at a time, [t-1, t, t+1]:
```
 [ 80ms ] [ 80ms ] [ 80 ms]        total 240ms
 [  f0  ] [ f0   ] [ f0   ]
 [  f1  ] [ f1   ] [ f1   ]
 ...
 ...
 ...
 [  f191] [ f191 ] [ f191 ]
```
By performing a weighted sum we are merging these features accross these time
240ms time frames. This is important so understand/detect changes over time, like
if something is increasing or decreasing (energy raising or falling). And each
filter, each of the 1536, has its own unique set of 576 learned weights, that
measures a specific temporal trend accross that 240ms time span.

192 is the latent feature dimension for the classification head, and the input
was downsampled by 8x (8 10ms frames stacked into one 80ms step) and this is
expanding them back.

The result will have the following shape:
```console
(gdb) p cur->ne
$5 = {14, 1536, 1, 1}
```
So we still have 14 timesteps but we now have 1536 values per step.

Next we reshape a
```console
    cur = ggml_add(ctx, cur, ggml_reshape_2d(ctx, model.upsample_b, 1, 1536));
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
```
If we look at the shapes of these tensors:
```console
(gdb) p cur->ne
$2 = {14, 1536, 1, 1}

(gdb) p model.upsample_b->ne
$3 = {1536, 1, 1, 1}
```
So the bias has one value for each filter. In ggml broadcasting can work if one
of the dimensions is 1, but at the moment we have 14 and 1536 for the first
dimension which will lead to an error. But if we reshape the bias tensor to that
it has 1 as its first dimension then we can add the bias. That will produce:
```console
(gdb) p cur->ne
$10 = {14, 1536, 1, 1}
```
And then we transpose that:
```console
(gdb) p ggml_transpose(ctx, cur)->ne
$11 = {1536, 14, 1, 1}
```
And ggml_cont will make this contiguous in memory.

Next we have the following reshape, where time=14 (14*8 = 112):
```c++
    cur = ggml_reshape_2d(ctx, cur, 192, time * 8);
```
```console
(gdb) p cur->ne
$4 = {192, 112, 1, 1}

0   [0               191]
1   [0               191]
             .
             .
             .
111 [0               191]
```
So we now have 112 10ms time frames each with 192 features each. So we are back
to the original 10ms resolution.

Now, with this shape we can then perform a matrix multiplication with model.hidden_w
which will 
```c++
    cur = ggml_mul_mat(ctx, model.head_hidden_w, cur);
```
```console
(gdb) p model.head_hidden_w->ne
$6 = {192, 192, 1, 1}

(gdb) p cur->ne
$7 = {192, 112, 1, 1}

0   [0               191]    0 [0 ...   111]
1   [0               191]    1 [0 ...   111]
             .                      .
             .                      .
             .                      .
             .
191 [0               191]  191 [0 ...   111]
```
So head_hidden_w "contains a function" for each feature set (192). Each of these
"function" is performing a dot product "asking" specific questions about a 10ms
slice, like "does this frame have a lot of Feature 3, while at the same time
lacking Feature 17, and having a moderate energy for Feature 82?". So this runs
192 distinct composite questions in parallel.
This does not change the shape:
```console
(gdb) p cur->ne
$8 = {192, 112, 1, 1}
```
We then add the bias:
```console
    cur = ggml_add(ctx, cur, model.head_hidden_b);
```
Followed by a relu which will filter out negative traits and confirmed vocal
characteristics remain active:
```console
    cur = ggml_relu(ctx, cur);
```
Then we have:
```c++
    cur = ggml_mul_mat(ctx, model.head_spks_w, cur);
    cur = ggml_add(ctx, cur, model.head_spks_b);
```
This is a matrix mutliplication 
```console
(gdb) p model.head_spks_w->ne
$10 = {192, 8, 1, 1}
  

(gdb) p cur->ne
$9 = {192, 112, 1, 1}

0  [0              191]     0 [0  ...   111]        0  [0 ... 7]       
1  [0              191]     1 [0  ...   111]        1  [0 ... 7]
2  [0              191]            .                       .
3  [0              191]            .              =        .
4  [0              191]            .                       .
5  [0              191]            .
6  [0              191]            .               111 [0 ... 7]
7  [0              191]            .
                                   .
                                   .
                           191 [  ...   111]

result shape: {8, 112, 1, 1}
```
So each or the rows in head_spks_w (the speaker weights) are like functions in
my way of thinking which have been trained take 192 features and for each
determine if it belongs to speaker 0-7:
```console
Row 0: "Given the acoustic rules active right now, how strongly does this frame
         match the voice established as Speaker 0.
...
Row 7: "How strongly does this frame match Speaker 7?"
```
Just to clarify something about the order of speakers. The 31 layers and
self-attention orgainized the internal representations so that the first unique
voice encountered in the audio is routed to match the pattern expected by Row 0.
The second unique voice encountered is routed to match the pattern expected by
Row 1, and so on.

So at this point we have logits for all the values. We then use the sigmoid
function to turn them into probabilities:
```c++
    ggml_tensor * output = ggml_sigmoid(ctx, cur);

```
```console
(gdb) p output->ne
$13 = {8, 112, 1, 1}

0   [0     7]     [p_spk0, p_spk1, p_spk2, p_spk3, p_spk4, p_spk5, p_spk6, p_spk7]
1   [0     7]     [p_spk0, p_spk1, p_spk2, p_spk3, p_spk4, p_spk5, p_spk6, p_spk7]
        .
        .
        .
111 [0     7]     [p_spk0, p_spk1, p_spk2, p_spk3, p_spk4, p_spk5, p_spk6, p_spk7]
```
So what we have as the output is 112 rows with 8 columns in each. Each row
represents 10ms and we can read this as:
```
frame 0 ( 0-10ms): [p_spk0, p_spk1, p_spk2, p_spk3, p_spk4, p_spk5, p_spk6, p_spk7]
frame 0 (10-20ms): [p_spk0, p_spk1, p_spk2, p_spk3, p_spk4, p_spk5, p_spk6, p_spk7]
...
```
By using this we can there for determine which speakers are active during the
frame by frame.

### Cache
This models uses self-attention but it does not have a kv-cache. This is as far
as I understand the arch of Sortformer. It uses an Embedding-Level Prefix Cache.

So above I simply traced through the first invocation where prefix was 0 and I
did not really consider the cache at that point. But it would have helped to
actually work through this and understand the cache as after the graph has been
computed the cache is updated and without this background it might not be clear
as to what the code is doing.

So we have the `input` to the graph which is our log mel-spectrogram and this
does not change. We also have positions as input, and then we have the third
optional input which is `cache`. Now this is optional and we have a if statement
in the build graph function:
```c++
    ggml_tensor * cache = nullptr;
    if (prefix) {
        cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, prefix);
        ggml_set_input(cache);
        ggml_set_name(cache, "cache");
        cur = ggml_concat(ctx, cache, cur, 1);
    }
```
So this will not always be added as a node in the graph, but also prefix can
change which will cause a change in the graph and hence cause it to be reallocated
which is something we want to avoid. For example the first call prefix will be
zero and the second the prefix will be 13:
```console
(lldb) p cache->ne
(int64_t[4])  ([0] = 512, [1] = 13, [2] = 1, [3] = 1)

0   [0  ...      511]
        .
        .
12  [0  ...      511]
```
This tensor is split into two logical regions:
```console
◄───────────────────────── prefix (l1 + l2)   ─────────────────────────►
┌──────────────────────────────────────┬───────────────────────────────┐
│            spkcache (l1)             │           fifo (l2)           │
│       Long-Term Speaker History      │       Short-Term Context      │
└──────────────────────────────────────┴───────────────────────────────┘
```
The short-term contex is hidden state saved before the model layers execute, so
this is the log mel-spectrogram input projected to the models hidden vector space
, from the previous graph execution. And recall that the shape of hidden is:
```console
(lldb) p hidden->ne
(int64_t[4])  ([0] = 512, [1] = 14, [2] = 1, [3] = 1)
```
This is to help the model from cutoffs between chunks, so that the model has
some backward context accross the chunk boundry.

The first part of the cache is the speaker cache. Recall that we said that the
first person to speak is identified as Speaker 0 and so on. Now lets say someone
speaks for 5 seconds and then a second person speaks for 45 seconds. If we only
had a rolling buffer then Speaker0 might be pushed out of the cache. This would
mean that the next time that person speaks it will be considered a new speaker
instead of the same speaker. So the cache is how the model keeps this state and
is able to keep track of speakers and their acuoustic features.

```console
(lldb) p cache->ne
(int64_t[4])  ([0] = 512, [1] = 13, [2] = 1, [3] = 1)

(lldb) p cur->ne
(int64_t[4])  ([0] = 512, [1] = 27, [2] = 1, [3] = 1)

0   [0  ...      511]
        .                    Cache
        .
12  [0  ...      511]
13  [0  ...      511]
        .                    cur
        .
27  [0  ...      511]
```
So this prefix will grow for each call, by 13 frames/rows.


```console
Input 1: New Chunk Audio            Input 2: Host Cache Buffer
  (Log-mel Spectrogram)               (Raw float array in RAM)
          │                                      │
          ▼                                      ▼
   [ mel_bins x frames ]                   [ 512 x prefix ]
          │                                      │
          ▼                                      │
  ggml_mul_mat(enc_pre_w)                        │
          │                                      │
          ▼                                      │
    hidden [ 512 x frames ]                      │
          │                                      │
          └──────────────────┬───────────────────┘
                             ▼
              ggml_concat(cache, cur, dim=1)
                             │
                             ▼
                cur [ cache  | cur ]
                             │
                             ▼
               LayerNorm (enc_norm)
                             │
                             ▼
               Transformer Layer 0 ... 30
```


```c++
    // project down into the model hidden vector space.
    ggml_tensor * hidden = ggml_mul_mat(ctx, model.enc_pre_w, input);
```
```console
(gdb) p hidden->ne
$1 = {512, 14, 1, 1}

0   [0  ...      511]
1   [0  ...      511]
        .
        .
        .
13  [0  ...      511]
```
So initially we have 14 "tokens" which are frames but we can think of them as
tokens in a text model. These are saved before the model processes and we will
use this after the graph has been computed which is why it is marked as output.

After the graph has been computed we have the following:
```c++
    // Get the hidden state of the previous chunk, this is the log-mel
    // spectrogram after it has been projected into the models hidden vector space.
    // So this is 80ms of state saved before the model processes it.
    std::vector<float> h(frames * 512);
    ggml_backend_tensor_get(hidden, h.data(), 0, h.size() * sizeof(float));
```
```console
(gdb) p frames
$2 = 14
```
And just to be able to understand this better, probs is populated like this:
```c++
    std::vector<float> probs(time * 8 * speakers);
    ggml_backend_tensor_get(output, probs.data(), 0, probs.size() * sizeof(float));
```
```console
gdb) p time
$5 = 14
(gdb) p speakers
$6 = 8
(gdb) p time * 8 * speakers
$7 = 896

(gdb) p output->ne
$8 = {8, 112, 1, 1}
```
```console
0   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 0
1   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 1
2   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 2
3   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 3
4   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 4
5   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 5
6   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 6
7   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 7
8   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 8
       .
       .
       .
111 [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 111
     ↑
     8 speaker probs
```

Next we have the following:
```c++
    std::vector<float> coarse(time * speakers, 0);
    for (int f = 0; f < time; ++f) {
        for (int j = 0; j < 8; ++j) {
            for (int s = 0; s < speakers; ++s) {
                coarse[f * speakers + s] += probs[(f * 8 + j) * speakers + s] / 8;
            }
        }
    }
```
The inner loop j is what is grouping our 8 frames:
```
f=0 

(f * 8 + j) * speakers + s
(f * 8 + j) * speakers + 0 = 0

       s=0
       ↓
j=0   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 0
j=1   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 1
j=2   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 2
j=3   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 3
j=4   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 4
j=5   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 5
j=6   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 6
j=7   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 7

(f * 8 + j) * speakers + s
(f * 8 + j) * speakers + 1 = 9

j=0   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 0
       s=1
       ↓
j=1   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 1
j=2   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 2
j=3   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 3
j=4   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 4
j=5   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 5
j=6   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 6
j=7   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 7


f=1
       s=0
       ↓
j=0   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 8
j=1   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 9
j=2   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 10
j=3   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 11
j=4   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 12
j=5   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 13
j=6   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 14
j=7   [p0 p1 p2 p3 p4 p5 p6 p7] 10 ms frame 15
```
So this going through and averaging 8 10ms frames for each speaker which brings
the values back to the 80ms time scale which will make it easier for the
calculations that the caching has to do.

Then we have the updating of the cache:
```c++
    whisper_diar_cache_update(cache, h.data(), frames, probs_80ms.data(), (right_mel + 7) / 8);
```
This is for streaming in a Sortformer which needs two things:
1. The immediate past few seconds of audio so phonetic transitions across chunk boundaries don't glitch.
2. A persistent memory of who each speaker is across minutes of conversation, without letting self-attention explode to infinity.

* The fifo (short term buffer) is a rolling window of the recent 80ms audio frames.
* spkcache (long term buffer) is anchor memory of historical speaker acoustic signatures.


```c++
static std::vector<float> whisper_diar_cache_update(whisper_diar_cache & state,
        const float * hidden, int frames, const float * probs, int rc) {
```
So we have the hidden vector which as we mentioned is the just processed graphs
input, projected to the models internal vector space. And then we have the number
of frames that were processed (normally 14 if we are not at the end I think), 
then we have probs_80ms which are the probabilities that the model just predicted
and this is where I got things confused initially when not considering the cache
being prepended to the projected input (hidden).

```c++
    const int chunk_valid = frames - rc;
```

```c++
static void whisper_diar_cache_compress(whisper_diar_cache & state,
                                        const std::vector<float> &  cache_preds) {
    ...

    std::vector<float> scores(static_cast<size_t>(n) * state.n_speakers);
```

```c++
    for (int f = 0; f < n; ++f) {
        // sum of 1 minus p ( log(1 -p))
        float sum_log1p = 0.f;
        // for each frame f we are doing to iterate over all speakers.
        for (int s = 0; s < state.n_speakers; ++s) {
            const float p = cache_preds[static_cast<size_t>(f) * state.n_speakers + s];
            sum_log1p += std::log(std::max(1.f - p, state.scoring.pred_score_threshold));
        }
```
Recall that p which is extracted from cache_preds is a probability that speaker
s is talking. 1 - p is P(speaker is silent).

```console
            7
sum_log1p = Σ  ln(1 - pj) = ln(1 - p0) + ln(1 - p1) + ln(1 - p2) + .. + ln(1 - p7)
           j=0                  ↑            ↑
                                |        P(speaker1 is silent)
                            P(speaker0 is silent)
```
Notice that this sum includes the silence term for _every_ speaker including
speaker s.

This is then used in the next loop, where we again loop over all the speakers
```c++
        for (int s = 0; s < state.n_speakers; ++s) {
            const float p     = cache_preds[static_cast<size_t>(f) * state.n_speakers + s];
            const float logp  = std::log(std::max(p, state.scoring.pred_score_threshold));
            const float log1p = std::log(std::max(1.f - p, state.scoring.pred_score_threshold));
            scores[static_cast<size_t>(f) * state.n_speakers + s] = logp - log1p + sum_log1p - log_half;
        }
```
Recall that sum_log1p includes the current speaker s which is why we calculate
P(speaker s is silent), log1p, and then subtract that (logp - log1p) that as we
want speaker s to be active so we need remove that speaker s is non-active.
We then add the sum_log1p the silence probability of all speakers, and log_half
it to center this at zero (recall that subtracting in log space is the same as
division in linear space, just like addition is multiplication). So that last
subtraction is:
```console
                       P
ln(P) - ln(0.5) = ln (---)
                      0.5
```
```console
If P(only s talks) > 0.5, then P/0.5 > 1.0 providing a positive score.
If P(only s talks) < 0.5, then P/0.5 < 1.0 providing a negative score.
If P(only s talks) = 0.5, then the score will be exactly 0.0.
```
So a score > 0 means that this speaker was speaking with no cross
talk and no background ambiuity.

We have 280 speaker frames (80ms) (state.n_spk_frames) and we have 8 speakers,
so in total score will be a vector of 280*8=2240:
```console
score[0] = frame0 speaker0: log-odds that only speaker 0 is speaking
score[1] = frame0 speaker1: log-odds that only speaker 1 is speaking
score[2] = frame0 speaker2: log-odds that only speaker 2 is speaking
score[3] = frame0 speaker3: log-odds that only speaker 3 is speaking
score[4] = frame0 speaker4: log-odds that only speaker 4 is speaking
score[5] = frame0 speaker5: log-odds that only speaker 5 is speaking
score[6] = frame0 speaker6: log-odds that only speaker 6 is speaking
score[7] = frame0 speaker7: log-odds that only speaker 7 is speaking

score[8] = frame1 speaker8: log-odds that only speaker 8 is speaking
score[9] = frame1 speaker9: log-odds that only speaker 9 is speaking
...
score[2240] = frame279 speaker9: log-odds that only speaker 9 is speaking
```

After all the score have been calculated we will then proceed with:
```c++
    std::vector<int> pos_count(state.n_speakers, 0);
    for (int f = 0; f < n; ++f) {
        for (int s = 0; s < state.n_speakers; s++) {
            const size_t i = static_cast<size_t>(f) * state.n_speakers + s;
            const bool is_speech = cache_preds[i] > 0.5f;
            if (!is_speech) {
                // if the current speaker did not talk then set to -inf to avoid
                // negative values in later operations.
                scores[i] = WHISPER_DIAR_NEG_INF;
            }
            // If the current speaker was speaking with no cross talk and no
            // background ambiguity the we increment that speakers count.
            if (scores[i] > 0.f) {
                pos_count[s]++;
            }
        }
    }
```
If a speaker has zero as its count then we can't overwrite or change that speakers
existing cache vectors.

Then we have:
```c++
    for (int s = 0; s < state.n_speakers; s++) {
        // If the count for a speaker is less that min_pos then skip it.
        // Just one or two 80ms frames do not contain enough accoustic information
        // to define human voice so if we have less that the defined min we
        // skip this speaker. For example, the current min_pos is 16 which is
        // about 1.28 seconds (16*80ms ≈ 1.28s).
        if (pos_count[s] < min_pos) {
            continue;
        }

        for (int f = 0; f < n; ++f) {
            const size_t i = static_cast<size_t>(f) * state.n_speakers + s;
            const bool is_speech = cache_preds[i] > 0.5f;
            // If the probability of cache_pred[i] is greater than 0.5 we consider it
            // that someone is talking. But we also want to make sure that the
            // score for this index (speaker) is greater than 0, because if it
            // is not then there is some kind of cross-talk or background noice
            // and it is not a pure solo speaker.
            if (is_speech && !(scores[i] > 0.f)) {
                scores[i] = WHISPER_DIAR_NEG_INF;
            }
        }
    }
```

Next we have:
```c++
    if (state.scoring.scores_boost_latest > 0.f) {
        for (int f = cap; f < n; ++f) {
            for (int s = 0; s < state.n_speakers; s++) {
                scores[static_cast<size_t>(f) * state.n_speakers + s] += state.scoring.scores_boost_latest;
            }
        }
    }
```
```console
(gdb) p state.scoring.scores_boost_latest
$53 = 0.0500000007
(gdb) p cap
$54 = 264
(gdb) p n
$55 = 280
```
So the above is looping from cap (264) to n (280) so in this case we have 16
frames of 80ms and we have 8 speakers.
```
score[2212] = frame264 speaker0: log-odds that only speaker 0 is speaking
score[2213] = frame264 speaker1: log-odds that only speaker 1 is speaking
score[2214] = frame264 speaker2: log-odds that only speaker 2 is speaking
score[2215] = frame264 speaker3: log-odds that only speaker 3 is speaking
score[2216] = frame264 speaker4: log-odds that only speaker 4 is speaking
score[2217] = frame264 speaker5: log-odds that only speaker 5 is speaking
score[2218] = frame264 speaker6: log-odds that only speaker 6 is speaking
score[2219] = frame264 speaker6: log-odds that only speaker 6 is speaking
...
```
And we are increasing the score for these log-odds by adding a positive constant
value. These are scores of more recent frames which is a way to avoid old audio
from minutes ago. The motivation for doing this is that a persons acoustic profile
is not completely static during a session, the speaker might turn their head,
lean back in their chair, or anything else that changes their vocal tone, volume
or pitch. If the cache only retained frames from the very beginning the models
achor vectors would represent how the speaker sounded 10 mins ago under different
acoustic conditions. This is a way of rotating in the current representations
of the voice.

Next we have:
```c++
    const int strong_k = static_cast<int>(std::floor(per_spk * state.scoring.strong_boost_rate));
    for (int s = 0; s < state.n_speakers; ++s) {
        for (int f : whisper_diar_topk_column(scores, n, state.n_speakers, s, strong_k)) {
            scores[static_cast<size_t>(f) * state.n_speakers + s] -= 2.f * log_half;
        }
    }
```
So we have the following values:
```console
(gdb) p per_spk
$1 = 32
(gdb) p state.scoring.strong_boost_rate 
$2 = 0.75

(gdb) p n
$7 = 280
(gdb) p state.n_speakers
$8 = 8

(gdb) p strong_k
$10 = 24
```
The above will iterate over all the 8 speakers and call whisper_diar_topk_column
for each of them setting f to the result. 
```c++
static std::vector<int> whisper_diar_topk_column(const std::vector<float> & scores,
        int n, int n_spk, int spk, int k) {
    // create a vector with a size of 280.
    std::vector<int> idx(n);
    // fill with sequentially increasing values starting from 0.
    std::iota(idx.begin(), idx.end(), 0);

    // we can't pick top k from a collection if it contains fewer elements than
    // k so we pick the smallest. If that is the case we would return all frames.
    k = std::min(k, n);

    // Next we sort the idx vector so that our top k entries are come first.
    std::partial_sort(idx.begin(),      // start
        idx.begin() + k,                // middle
        idx.end(),                      // last
        [&](int a, int b) {
        const float sa = scores[static_cast<size_t>(a) * n_spk + spk];
        const float sb = scores[static_cast<size_t>(b) * n_spk + spk];
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });

    idx.resize(k);

    return idx;
}
```
I've updated the code to use nth_element as we don't need the result to be 
sorted.
```c++
    for (int s = 0; s < state.n_speakers; ++s) {
        for (int f : whisper_diar_topk_column(scores, n, state.n_speakers, s, strong_k)) {
            scores[static_cast<size_t>(f) * state.n_speakers + s] -= 2.f * log_half;
        }
    }
```
So this will call whisper_diar_topk_column for each speaker, and the inner for
loop it iterating over the frame indices that we get back. And updating the
scores for by subtracting (2.0 * log_half). This is in fact adding a positive
constant:
```console
(gdb) p log_half
$3 = -0.693147182

(gdb) p std::log(0.5)
$4 = -0.693147180559945309429

(gdb) p -2.0 * log_half
$5 = 1.3862943649291992
```
So this is adding a constant positive value to each or the top k scores for
each speaker. And remember that we are in log space so adding 1.386 (log(4)
is like multiplying by 4.

To clarify this lets look at a simple example:
```console
Total cache capacity: 4 frames (80ms each)
strong_k            : 2 (each speaker is guaranteed up to 2 protected slots)
```
We have 6 candidate frames in memory. 4 from speaker 0 which is a loud and sitting
close to the microphone speaking confidantly. And 2 from speaker 1 who is talking
quietly and setting a bit away from the microphone.

Recall that score = ln(P/0.5):
```console
Speaker 0:
Frame 0: P = 0.95 -> score = ln(0.95 / 0.5) = +0.64
Frame 1: P = 0.92 -> score = ln(0.92 / 0.5) = +0.61
Frame 2: P = 0.90 -> score = ln(0.90 / 0.5) = +0.59
Frame 3: P = 0.88 -> score = ln(0.88 / 0.5) = +0.56

Speaker 1:
Frame 0: P = 0.60 -> score = ln(0.60 / 0.5) = +0.18
Frame 1: P = 0.55 -> score = ln(0.55 / 0.5) = +0.10
```
If we just picked the top 4 frames by score we would get all the frames from
speaker 0 and none from speaker 1. But with the strong_k boost:
```console
boost = -2.0 * ln(0.5) = +2.0 * 0.693 = +1.39

Speaker 0:
Frame 0: 0.64 + 1.39 = +2.03 (boosted)
Frame 1: 0.61 + 1.39 = +2.00 (boosted)
Frame 2: not boosted as it exceeds strong_k
Frame 3: not boosted as it exceeds strong_k

Speaker 0:
Frame 0: 0.18 + 1.39 = +1.57 (boosted)
Frame 0: 0.10 + 1.39 = +1.49 (boosted)
```

Then we have the following which will use the boosted scores that was updated
just before this. And notice that this is using weak_k and not strong_k:
```c++
    const int weak_k = static_cast<int>(std::floor(per_spk * state.scoring.weak_boost_rate));
    for (int s = 0; s < state.n_speakers; ++s) {
        for (int f : whisper_diar_topk_column(scores, n, state.n_speakers, s, weak_k)) {
            scores[static_cast<size_t>(f) * state.n_speakers + s] -= log_half;
        }
    }
```
```console
(gdb) p state.scoring.weak_boost_rate
$1 = 1.5
(gdb) p weak_k
$2 = 48
```
So this is getting the 48 top k values from scores (boosted remember), and for
each one gets a boost of +0.69. So this is include more then the first boost
and giving runner ups a boost too.

Next we have:
```c++
    const int n_pad = n + state.scoring.sil_frames_per_spk;
```
```console
(gdb) p n + state.scoring.sil_frames_per_spk
$2 = 281
(gdb) p state.scoring.sil_frames_per_spk
$3 = 1
```

```c++
    // create a vector of 281 * 8 = 2248
    std::vector<int64_t> flat(static_cast<size_t>(state.n_speakers) * n_pad);
    // generate sequential increasing values starting from 0.
    std::iota(flat.begin(), flat.end(), 0);

    auto flat_score = [&](int64_t i) -> float {
        const int f = static_cast<int>(i % n_pad);
        if (f >= n) {
            return WHISPER_DIAR_POS_INF;
        }
        const int s = static_cast<int>(i / n_pad);
        return scores[static_cast<size_t>(f) * state.n_speakers + s];
    };

    std::partial_sort(flat.begin(),
                      flat.begin() + cap,    // cap=264
                      flat.end(),
                      [&](int64_t a, int64_t b) {
        const float sa = flat_score(a), sb = flat_score(b);
        if (sa != sb) {
            return sa > sb;
        }
        return a < b;
    });
```
Notice the indexing:
```c++
        const int f = static_cast<int>(i % n_pad);
        const int s = static_cast<int>(i / n_pad);
```
So recall that our scores are in [frames, speakers] where we would index using
```console
idx = f * n_speakers + s
idx = 1 * n_speakers + 1
idx = 9
    0  score[0] = frame0 speaker0: log-odds that only speaker 0 is speaking
    1  score[1] = frame0 speaker1: log-odds that only speaker 1 is speaking
    2  score[2] = frame0 speaker2: log-odds that only speaker 2 is speaking
    3  score[3] = frame0 speaker3: log-odds that only speaker 3 is speaking
    4  score[4] = frame0 speaker4: log-odds that only speaker 4 is speaking
    5  score[5] = frame0 speaker5: log-odds that only speaker 5 is speaking
    6  score[6] = frame0 speaker6: log-odds that only speaker 6 is speaking
    7  score[7] = frame0 speaker7: log-odds that only speaker 7 is speaking

    8  score[8] = frame1 speaker8: log-odds that only speaker 8 is speaking
--> 9  score[9] = frame1 speaker9: log-odds that only speaker 9 is speaking
    ...
    2240 score[2240] = frame279 speaker9: log-odds that only speaker 9 is speaking
```
The flat vector is instead in [n_speakers, n_pad]:
```console
(gdb) p state.n_speakers
$9 = 8

    const int n_pad = n + state.scoring.sil_frames_per_spk;
(gdb) p n_pad
$8 = 281

(gdb) p state.scoring.sil_frames_per_spk
$10 = 1


Speaker 0:  [ 0  ...  279 pad]
Speaker 1:  [ 0  ...  279 pad]
...
Speaker 7:  [ 0  ...  279 pad]
                        ↑  ↑  
                        m  state.scoring.sil_frames_per_spk
```
Recall that flat just contains indices [0, 2248) and these encode speaker/frame
pairs:
```console
(gdb) p n_pad
$18 = (const int &) @0x7fffffffce10: 281

(gdb) p flat.size()
$19 = 2248

    const int n_pad = n + state.scoring.sil_frames_per_spk;
    std::vector<int64_t> flat(static_cast<size_t>(state.n_speakers) * n_pad);

s = i/281
f = i%281
```
So every value in flat is a compact representation of speaker s at frame f.
So each speaker has 281 frames including one padding frame, so 280 real audio
frames and one silence padding frame. The index that can be passed to the lambda
is any of the 2248 depending on if the are part of the top cap selected ones.
If we get an index into the first 381 then what would one for the first speaker:
```console
(gdb) p flat[7]/281
$21 = 0  (speaker 0)
(gdb) p flat[7]%281
$22 = 7  (frame 7)

(gdb) p flat[2240]/281
$36 = 7   (speaker 7)
gdb) p flat[2240]%281
$35 = 273  (frame 273
```
```c++
    auto flat_score = [&](int64_t i) -> float {
        // mod with real audio frames plus padding (n_pad)
        const int f = static_cast<int>(i % n_pad);
        if (f >= n) {
            // padding so set to positive inf
            return WHISPER_DIAR_POS_INF;
        }
        // the index i is the index in the 
        const int s = static_cast<int>(i / n_pad);
        return scores[static_cast<size_t>(f) * state.n_speakers + s];
    };
```
In the partial sort we we flat_score to 
```c++
    std::partial_sort(flat.begin(), flat.begin() + cap, flat.end(), [&](int64_t a, int64_t b) {
        const float sa = flat_score(a);
        const float sb = flat_score(b);
        // highest score wins if the scores are not equal. Because we set
        // the padding frames as positive infinity they are greater than any
        // real score.
        if (sa != sb) {
            return sa > sb;
        }
        // if the are equal return the one with the lowest index.
        return a < b;
    });
```

```console
$10 = std::vector of length 2248, capacity 2248 = {
280,  speaker 0 silence (0 * 281 + 280)
561,  speaker 1 silence (1 * 281 + 280)
842,  speaker 2 silence (2 * 281 + 280)
1123, speaker 3 silence (3 * 281 + 280)
1404, speaker 4 silence (4 * 281 + 280)
1685, speaker 5 silence (5 * 281 + 280)
1966, speaker 6 silence (6 * 281 + 280)
2247, speaker 7 silence (7 * 281 + 280)
```
So we can see that all the silence frames indices will be first in the flat
vector. So what we have done here is to fullfill a requirement that Sortformer
has:
Sortformer requires a baseline silence anchor for every speaker channel
(even inactive ones) so self-attention has a negative reference point
and doesn't hallucinate speech during pauses. Setting virtual silence
frames (f >= n) to +INF guarantees that exactly sil_frames_per_spk slots
per speaker are locked in at the front of the cache (indices 0..7).

Next we have:
```c++
    // Create a new vector with the contents of flat up to cap (264).
    std::vector<int64_t> picked(flat.begin(), flat.begin() + cap);
    // iterate over all the indices as modifiable reference so we are updating
    // i that is.
    for (auto & i : picked) {
        if (flat_score(i) == WHISPER_DIAR_NEG_INF) {
            // update the index to a sentinel id
            i = WHISPER_DIAR_MAX_INDEX * static_cast<int64_t>(n_pad) + WHISPER_DIAR_MAX_INDEX;
        }
    }
    std::sort(picked.begin(), picked.end());
```

```console
(gdb) p picked
$2 = std::vector of length 264, capacity 264 = {280, 561, 842, 1123, 1404, 1685, 1966, 2247, 68, 103, 104, 100, 
  105, 101, 106, 156, 67, 102, 107, 111, 157, 69, 98, 99, 110, 155, 62, 159, 109, 108, 158, 66, 313, 314, 599, 315, 
  746, 312, 598, 747, 600, 316, 97, 154, 55, 59, 60, 61, 63, 58, 56, 57, 81, 54, 65, 96, 153, 64, 82, 160, 161, 50, 
  70, 112, 53, 51, 83, 52, 95, 49, 152, 84, 48, 21, 47, 162, 20, 19, 22, 17, 94, 18, 46, 151, 16, 71, 15, 45, 150, 
  1, 73, 14, 44, 72, 149, 80, 93, 113, 74, 0, 148, 217, 121, 43, 122, 120, 123, 216, 124, 125, 218, 147, 114, 119, 
  2, 13, 118, 115, 117, 42, 116, 215, 163, 12, 126, 229, 146, 23, 228, 127, 41, 11, 230, 128, 237, 238, 219, 191, 
  192, 239, 164, 129, 168, 236, 85, 130, 227, 240, 214, 131, 165, 40, 167, 166, 10, 193, 241, 132, 3, 169, 75, 235, 
  231, 242, 133, 190, 39, 134, 92, 24, 243, 220, 79, 135, 194, 4, 5, 6, 7, 8, 9, 25, 26, 27, 28, 29, 30, 31, 32, 
  33, 34, 35, 36, 37, 38, 76, 77, 78, 86, 87, 88, 89, 90, 91, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 
  170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 195, 196, 
  197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 221, 222, 223, 224, 225, 
  226, 232, 233, 234, 244, 245}

before sort:
(gdb) p picked
$7 = std::vector of length 264, capacity 264 = {280, 561, 842, 1123, 1404, 1685, 1966, 2247, 68, 103, 104, 100, 
  105, 101, 106, 156, 67, 102, 107, 111, 157, 69, 98, 99, 110, 155, 62, 159, 109, 108, 158, 66, 313, 314, 599, 315, 
  746, 312, 598, 747, 600, 316, 97, 154, 55, 59, 60, 61, 63, 58, 56, 57, 81, 54, 65, 96, 153, 64, 82, 160, 161, 50, 
  70, 112, 53, 51, 83, 52, 95, 49, 152, 84, 48, 21, 47, 162, 20, 19, 22, 17, 94, 18, 46, 151, 16, 71, 15, 45, 150, 
  1, 73, 14, 44, 72, 149, 80, 93, 113, 74, 0, 148, 217, 121, 43, 122, 120, 123, 216, 124, 125, 218, 147, 114, 119, 
  2, 13, 118, 115, 117, 42, 116, 215, 163, 12, 126, 229, 146, 23, 228, 127, 41, 11, 230, 128, 237, 238, 219, 191, 
  192, 239, 164, 129, 168, 236, 85, 130, 227, 240, 214, 131, 165, 40, 167, 166, 10, 193, 241, 132, 3, 169, 75, 235, 
  231, 242, 133, 190, 39, 134, 92, 24, 243, 220, 79, 135, 194, 28199718, 28199718, 28199718, 28199718, 28199718, 
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718}

after sort:
(gdb) p picked
$8 = std::vector of length 264, capacity 264 = {0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
  24, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
  66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 79, 80, 81, 82, 83, 84, 85, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
  102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
  124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
  156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 190, 191, 192, 193, 194, 214, 215, 216,
  217, 218, 219, 220, 227, 228, 229, 230, 231, 235, 236, 237, 238, 239, 240, 241, 242, 243, 280, 312, 313, 314,
  315, 316, 561, 598, 599, 600, 746, 747, 842, 1123, 1404, 1685, 1966, 2247, 28199718, 28199718, 28199718,
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718,
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718,
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718,
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718,
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718,
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718,
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718,
  28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718, 28199718}
```
The winning frames are grouped by speaker and sequenced in time after this.


Next we have:
```c++
    // one vector for the acuostic embeddings which is what feeds into the
    // attention layer.
    std::vector<float> new_cache;
    new_cache.reserve(static_cast<size_t>(cap) * state.n_embd);

    // Stores what models predicted for each speaker during each of those cached
    // frames.
    std::vector<float> new_preds;
    new_preds.reserve(static_cast<size_t>(cap) * state.n_speakers);

    for (int j = 0; j < cap; ++j) {
        const int64_t i = picked[j];
        const int f = static_cast<int>(i % n_pad);

        bool silence_anchor = f >= n;
        bool sentinel_id = i >= static_cast<int64_t>(state.n_speakers) * n_pad;
        if (silence_anchor || sentiel_id) {
            new_cache.insert(new_cache.end(), state.mean_sil_emb.begin(), state.mean_sil_emb.end());
            new_preds.insert(new_preds.end(), state.n_speakers, 0.f);
        } else {
            const auto emb_begin = state.spkcache.begin() + static_cast<size_t>(f) * state.n_embd;
            const auto pred_begin = cache_preds.begin() + static_cast<size_t>(f) * state.n_speakers;
            new_cache.insert(new_cache.end(), emb_begin, emb_begin + state.n_embd);
            new_preds.insert(new_preds.end(), pred_begin, pred_begin + state.n_speakers);
        }
    }
```

output from ami_en2002d_2132.wav:
```console
speaker 0: 0.00 -> 0.30
speaker 0: 0.80 -> 1.97
speaker 1: 2.48 -> 2.86
speaker 2: 2.88 -> 3.13
speaker 0: 3.11 -> 6.07
speaker 0: 6.36 -> 6.88
speaker 0: 7.39 -> 10.86
speaker 0: 11.65 -> 13.60
speaker 2: 14.68 -> 14.89
speaker 0: 15.22 -> 15.56
speaker 0: 17.08 -> 17.67
speaker 0: 18.13 -> 18.55
speaker 0: 18.81 -> 19.51
speaker 1: 26.80 -> 27.06
speaker 1: 28.01 -> 28.18
speaker 1: 28.37 -> 28.72
speaker 1: 29.23 -> 29.35
speaker 1: 30.26 -> 31.16
speaker 1: 32.32 -> 33.49
speaker 1: 33.71 -> 35.71
speaker 2: 33.88 -> 34.12
speaker 1: 35.97 -> 36.36
speaker 2: 36.93 -> 37.10
speaker 1: 37.32 -> 39.54
speaker 2: 40.45 -> 40.65
speaker 1: 40.84 -> 41.41
speaker 1: 41.59 -> 43.13
speaker 2: 44.42 -> 44.67
speaker 1: 45.39 -> 46.01
speaker 2: 46.76 -> 47.07
speaker 1: 47.37 -> 47.98
speaker 0: 48.54 -> 50.89
speaker 2: 51.23 -> 51.64
speaker 2: 51.86 -> 53.29
speaker 2: 53.94 -> 54.19
speaker 2: 54.76 -> 55.02
speaker 2: 55.42 -> 56.04
speaker 2: 56.73 -> 57.34
speaker 1: 57.63 -> 59.22
speaker 0: 57.84 -> 58.20
speaker 1: 59.50 -> 59.91
speaker 2: 59.65 -> 59.90

speaker 0: 0.00 -> 2.04
speaker 1: 2.25 -> 2.94
speaker 2: 2.66 -> 3.19
speaker 0: 2.89 -> 10.91
speaker 0: 11.42 -> 13.67
speaker 0: 15.00 -> 15.63
speaker 0: 16.86 -> 19.58
speaker 1: 26.61 -> 27.14
speaker 1: 27.80 -> 28.80
speaker 1: 30.04 -> 31.23
speaker 1: 32.09 -> 36.43
speaker 2: 33.66 -> 34.19
speaker 1: 37.10 -> 39.62
speaker 1: 40.61 -> 43.20
speaker 2: 44.20 -> 44.75
speaker 1: 45.17 -> 46.09
speaker 2: 46.53 -> 47.15
speaker 1: 47.15 -> 48.06
speaker 0: 48.32 -> 50.96
speaker 2: 51.00 -> 53.36
speaker 2: 53.71 -> 56.11
speaker 2: 56.51 -> 57.42
speaker 1: 57.41 -> 59.98
speaker 0: 57.61 -> 58.27
speaker 2: 59.42 -> 59.98
```
Python output for the same file:
```console
0.000 0.320 speaker_0
0.800 1.960 speaker_0
2.480 2.860 speaker_1
2.870 3.130 speaker_2
3.120 6.070 speaker_0
6.360 6.880 speaker_0
7.390 10.860 speaker_0
11.650 13.600 speaker_0
14.680 14.890 speaker_2
15.220 15.570 speaker_0
17.080 17.650 speaker_0
18.130 18.550 speaker_0
18.810 19.500 speaker_0
26.770 27.070 speaker_1
28.000 28.190 speaker_1
28.350 28.730 speaker_1
29.160 29.380 speaker_1
30.260 31.150 speaker_1
32.320 33.490 speaker_1
33.700 35.710 speaker_1
33.880 34.120 speaker_2
35.970 36.370 speaker_1
36.930 37.100 speaker_2
37.320 39.540 speaker_1
40.450 40.650 speaker_2
40.840 41.410 speaker_1
41.590 43.130 speaker_1
44.420 44.670 speaker_2
45.390 46.010 speaker_1
46.750 47.070 speaker_2
47.370 47.980 speaker_1
48.540 50.900 speaker_0
51.230 51.630 speaker_2
51.860 53.300 speaker_2
53.940 54.190 speaker_2
54.750 55.020 speaker_2
55.420 56.030 speaker_2
56.730 57.340 speaker_2
57.630 59.230 speaker_1
57.840 58.200 speaker_0
59.500 59.910 speaker_1
59.650 59.900 speaker_2
```
Output from NeMo-Speech:
```console
./build/bin/diarize_file ~/work/ai/whisper-work/models/Nemotron-3-Diarization.q8_0.gguf ~/work/ai/whisper-work/samples/ami_en2002d_2132.wav  --gpu -1
[diarize_file] 60.0s audio, 6001 frames (streaming), 25 segments
  [   0.000s -    2.039s] speaker 1
  [   2.251s -    2.939s] speaker 2
  [   2.661s -    3.189s] speaker 3
  [   2.891s -   10.909s] speaker 1
  [  11.421s -   13.669s] speaker 1
  [  15.001s -   15.629s] speaker 1
  [  16.861s -   19.579s] speaker 1
  [  26.611s -   27.139s] speaker 2
  [  27.801s -   28.799s] speaker 2
  [  30.041s -   31.229s] speaker 2
  [  32.091s -   36.429s] speaker 2
  [  33.661s -   34.199s] speaker 3
  [  37.101s -   39.619s] speaker 2
  [  40.611s -   43.209s] speaker 2
  [  44.201s -   44.749s] speaker 3
  [  45.171s -   46.089s] speaker 2
  [  46.531s -   47.149s] speaker 3
  [  47.151s -   48.059s] speaker 2
  [  48.321s -   50.959s] speaker 1
  [  51.001s -   53.359s] speaker 3
  [  53.711s -   56.109s] speaker 3
  [  56.511s -   57.419s] speaker 3
  [  57.411s -   59.979s] speaker 2
  [  57.611s -   58.269s] speaker 1
  [  59.421s -   59.979s] speaker 3
```

### segments
So after we have processed the audio file the model we will update the
state.prob vector:
```c++
    state.buf_probs.resize(time * 8 * speakers);
    ggml_backend_tensor_get(output, state.buf_probs.data(), 0, state.buf_probs.size() * sizeof(float));

    // write probabilites to probs.
    state.probs.insert(state.probs.end(),
                       state.buf_probs.begin() + prefix * 8 * speakers,
                       state.buf_probs.begin() + (prefix * 8 + valid_mel) * speakers);
```
And this just a flat vector with the size n_frames * n_speakers. For example aftr
the first chunk we would have:
```console
(gdb) p state.probs.size()
$15 = 832
```
And recall that the computation graph upsamples back to 10ms. So we would have
10ms mel frames, and a chunk_len of 13, and 8 speakers:
```console
13 * 8 = 104
104 * 8 = 832
```
This is a flat array in the shape of [n_frames, n_speakers].  So to access
frame f, we do f * n_speakers + s
```console
    0 frame0 : speaker0
    1 frame0 : speaker1
    2 frame0 : speaker2
        ...
    7 frame0 : speaker7
    8 frame1 : speaker0
    9 frame1 : speaker1
        ...
```
```c++
        for (int s = 0; s < ctx->model.hparams.n_speakers; ++s) {
            std::vector<whisper_diar_segment> segments;

            // Find contigous frames where speaker s is active
            int64_t start = -1;
            for (int64_t f = 0; f <= n_frames; ++f) {
                const float v = f < n_frames ? ctx->state.probs[f * ctx->model.hparams.n_speakers + s] : -1.0f;
                if (start < 0 && f < n_frames && v >= p.start_threshold) {
                    start = f;
                }
                if (start >= 0 && (f == n_frames || v < p.stop_threshold)) {
                    segments.push_back({start, f, s});
                    start = -1;
                }
            }
```

```c++
            // by applying the padding above it is now possible that segments
            // overlap. So we merge overlapping segments here.
            std::vector<whisper_diar_segment> merged;
            for (const auto & seg : segments) {
                if (merged.empty()) {
                    merged.push_back(seg);
                    continue;
                }

                bool t0_overlap = seg.t0 <= merged.back().t1;
                bool sil_to_short = (seg.t0 - merged.back().t1) * 10 < p.min_silence_duration_ms;

                if (t0_overlap || sil_to_short) {
                    merged.back().t1 = std::max(merged.back().t1, seg.t1);
                } else {
                    merged.push_back(seg);
                }
            }
```
After padding, this can happen:
```console
  seg1 original: [10, 20] -> padded: [0, 28]   (pushed to merged first)
  seg2 original: [13, 18] -> padded: [0, 26]   (shorter after padding, fully inside seg1) 
```
```c++
            for (const auto & seg : merged) {
                if ((seg.t1 - seg.t0) * 10 >= p.min_speech_duration_ms) {
                    result->data.push_back(seg);
                }
            }
```
