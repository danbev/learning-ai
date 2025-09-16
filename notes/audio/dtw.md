### Dynamic Time Warping (DTW)
This is about aligning the transcribed text with precise timestamps in the
audio. When whisper generated text from audio it needs to determine preciely
when each word was spoken. But the encoder-decoder model does not have this
concept of timestamps. 

So the DTW algorithm looks something like this:
```
Input: x_1:N, y_1:M
Cost matrix D ε R^{N+1 x M+1}

Initialization:
for i=1 to N: D_i,0 = ∞
for j=1 to M: D_0,j = ∞
D_0,0 = 0

for i = 1 to N         // For each row
  for j = 1 to M       // For each column
    D_ij = d(x_i, y_j) + min(D_i-1,j, D_i,j-1, D_i-1,j-1)

d(x_i, y_j) = |x_i - y_j|

Get alignment: Traceback from D_N,M to D_0,0
```
So in this case there are two inputs signals that we want to compare.
```
x = [0, 2, 0, 1, 0, 0]  N = 6

2       *
      /   \
1    /     \     *
    /       \  /   \
0  *         *      *-----*
   x_1 x_2  x_3 x_4 x_5  x_6


y = [0, 0, 0, 0.5, 2, 0, 1, 0] M = 7

2            *
           /   \
1         /     \     *
         *       \  /   \
0  *----*         *      *
   x_1 x_2  x_4  x_5 x_6 x_7
         x_3

(x_3 = 0.5)
```

Initialization:
```
  +---+---+---+---+---+---+---+---+
6 | ∞ |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
5 | ∞ |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
4 | ∞ |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
3 | ∞ |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
2 | ∞ |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
1 | ∞ |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
0 | 0 | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ | ∞ |
  +---+---+---+---+---+---+---+---+
    0   1   2   3   4   5   6   7
```

```
      D0, 0
min { D0, 1
      D1, 0

        +---+---+
D1,0--> |   | * |
        +---+---+
D0,0--> |   |   |<--D0,1
        +---+---+
```
So we are taking the minium of the neighbors of the cell. Notice that we only
consider cells that have already been calculated. This is like asking what is
the path that minimizes the cost to get to this cell.

When we calculate the cost of a cell we also record which path was the cheapest
which can then be used to backtrack and get the alignment.

Now, in the case of whisper the inputs are:
```console
x = the text tokens.
y = the audio frames.
```
Now, we say above that the cost matrix used x an y and the values, but in the
case of whisper what is used is instead the KQ from the attention heads, the
ones that are specifically for the detection of features of time and audio
relationsships, those KQ values are what form the cost matrix in whisper.

So the cross attention mechanism in the transformers produces the KQ values for
all heads. But only the alignment heads are extracted and used to form the cost
matrix in the DTW algorithm.
These values are normalized and applied a median filter.

And the alignment path is how the text tokens and the audio signal frames line
up, similar to the x and y inputs in the graph above. Each point along this
path tells us, "this token corresponds to the this audio frame".

So I said that the KQ values for the alignment heads are used to form the cost
and that these are extracted from the other KQ values. This would not be efficient
so instead a mask is used to to select just the value from the alignment heads.
```c++
struct whisper_aheads_masks {
    std::vector<struct ggml_tensor *> m;    // One mask per text layer.
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};
```
```c++
static const whisper_ahead g_aheads_base_en[]   = { {3, 3}, {4, 7}, {5, 1}, {5, 5}, {5, 7} };
```
And this model has:
```console
whisper_init_from_file_with_params_no_state: loading model from 'models/ggml-base.en.bin'
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
```
So this model has 6 text layers, and 8 heads. And looking at the
`g_aheads_base_en` we can see we need masks for layer 3, 4, and 5.
```
Layer 3: Head 3
Layer 4: Head 7
Layer 5: Head 1, 5, 7

0  0 0 0 0 0 0 0 0
1  0 0 0 0 0 0 0 0
2  0 0 0 0 0 0 0 0
3  0 0 0 1 0 0 0 0
4  0 0 0 0 0 0 0 1
5  0 1 0 0 0 1 0 1
```


```c++
    typedef struct whisper_ahead {
        int n_text_layer;
        int n_head;
    } whisper_ahead;
```
This is related to DTW (Dynamic Time Warping). 
Is there about figuring out the aligment, the path through the cost matrix in
DTW?

```c++
    typedef struct whisper_aheads {
        size_t n_heads;
        const whisper_ahead * heads;
    } whisper_aheads;
```
So this only contains an array of `whisper_ahead` structs and the length of
this array.
