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

If we have a previous hidden state, called a prefix in the code we will add
an input.
```c++
    ggml_tensor * cache = nullptr;
    if (prefix) {
        cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, prefix);
        ggml_set_input(cache);
        ggml_set_name(cache, "cache");
        cur = ggml_concat(ctx, cache, cur, 1);
    }
```
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

### cache
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
    whisper_diar_cache_update(cache, h.data(), frames, probs_80ms.data(), 0, (right_mel + 7) / 8);
```
This is for streaming in a Sortformer which needs two things:
1. The immediate past few seconds of audio so phonetic transitions across chunk boundaries don't glitch.
2. A persistent memory of who each speaker is across minutes of conversation, without letting self-attention explode to infinity.

* The fifo (short term buffer) is a rolling window of the recent 80ms audio frames.
* spkcache (long term buffer) is anchor memory of historical speaker acoustic signatures.

```console
◄─────────────── prefix (l1 + l2) ───────────────► ◄──────────── frames ────────────►
┌───────────────────────┬──────────────────────────┬────────┬─────────────┬──────────┐
│    spkcache (l1)      │        fifo (l2)         │   lc   │ chunk_valid │    rc    │
│  (Long-term memory)   │    (Short-term memory)   │ (past) │  (NEW AUDIO)│ (future) │
└───────────────────────┴──────────────────────────┴────────┴─────────────┴──────────┘
```
l1 = state.n_spk_frames is the number of frames currently in the long term speaker cache.
l2 = state.n_fifo_frames is the number of frames in the short term fifo queue.
lc = left context is the overlapping context from the end of the previous chunk.
rc = right context is a look ahead into future frames.

Now the model evaluates the entire sequence and the output is as we saw above
80ms token probabilites) which contains predictions for all of these regions
concatenated.

This is how we get to the fifo contents using l1 to skip the speaker cache which
is first, times the number of speakers. 
```c++
const float * fifo_preds = probs + static_cast<size_t>(l1) * state.n_speakers;
```
So this will then point to the updated probabilites for the fifo.

The we have chunk_preds which skips the l1, l2 and lc to get the current new audio:
```c++
const float * chunk_preds = probs + static_cast<size_t>(l1 + l2 + lc) * state.n_speakers;
```

Then we have chunk_valid_state we are using hidden (the hidden state saved before
the models layers execute, and we only want the new values not the past (lc):
```c++
const float * chunk_valid_state = hidden + static_cast<size_t>(lc) * state.n_embd;
```
```console
(gdb) p state.n_embd
$11 = 512

(gdb) p lc
$12 = 0
```

```c++
    state.fifo.insert(state.fifo.end(),
                      chunk_valid_state,
                      chunk_valid_state + static_cast<size_t>(chunk_valid) * state.n_embd);
```
This is using iterator(const_iterator pos, InputIt first, InputIt last), so
chunk_valid_state is a raw pointer it act like a standard random access iterator
so this is saying that we will use a pointer to the start of chunk_valid_state
and then a pointer to the end of the source data.
```console
(gdb) p state.fifo.size()
$5 = 6656
```
Next we have:
```c++
    std::vector<float> fifo_preds_full(static_cast<size_t>(l2 + chunk_valid) * state.n_speakers);
```
```console
(gdb) p fifo_preds_full
$13 = std::vector of length 104, capacity 104 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0}
```
Next we are doing to copy into that the above vector, from fifo_preds, that the
number of elements will be 
```c++
    std::memcpy(fifo_preds_full.data(), fifo_preds, static_cast<size_t>(l2) * state.n_speakers * 4);

```

Notice that emitted is using std::vector's range constructor:
```c++
    // What we will actually return as the updated cache.
    std::vector<float> emitted(chunk_preds, chunk_preds + static_cast<size_t>(chunk_valid) * state.n_speakers);
```
_wip_
