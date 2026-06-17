## Parakeet Local attention
This is also called limited context attention in Nemo I think.

The problem is that full self attention is O(n_time²), which is doable for shorter
audio samples but becomes an issue for larger samples.
So for a certain threshold we switch to local attention and restrict each frame
to a 257-frame window (128 past frames, itself, and 128 future frames).

Every query frame needs a different 257-wide slice of the keys. Computing this
per time frame would mean many tiny matmuls which is inefficient.
Instead we taken n_time (then number of time frames, think token sequence in an
LLM). For example, we might have:
```c++
    struct ggml_tensor * local_mask = nullptr;
    if (local_attn) {
        const int chunk = att_left + att_right;
        local_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, chunk + window_size - 1, chunk);
    }
```


```console
1538	        const int chunk = att_left + att_right;
(gdb) p att_left
$2 = 128
(gdb) p att_right
$3 = 128
(gdb) p chunk
$6 = 256

(gdb) p n_time
$4 = 20875

gdb) p local_mask->ne
$7 = {512, 256, 1, 1}
```

```console
      key-axis (128 before, 256 for the block itself, 128 after)
0  [0             511]
   [       ...       ]
   [       ...       ]    query-axis (256 queries = one chunk)
   [       ...       ]
   [       ...       ]
255[       ...       ]
```
So a query has a 257 window size, each individual query can attent to a window
of 128 left + itself + 128 right.

Lets try as simpler example with smaller values:
```console
chunk       = 4
att_left    = 2
att_right   = 2
window_size = 5
slab        = 4+5-1  = 8

         k=0  1  2  3  4  5  6  7
  q=0:     1  1  1  1  1  0  0  0   <- valid keys [0,5)
  q=1:     0  1  1  1  1  1  0  0   <- valid keys [1,6)
  q=2:     0  0  1  1  1  1  1  0   <- valid keys [2,7)
  q=3:     0  0  0  1  1  1  1  1   <- valid keys [3,8)
```
Query = one output per time-frame. Each query is asking, "given my position,
what other frames should I look at?". So query=0 is time-frame 0, query-1 is
time frame 1 etc.


Each row is a 5-wide band of 1s, shifted right by 1 per row, over a 8 wide slab.

```c++
            struct ggml_tensor * Q_cur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);

            Q_cur = ggml_reshape_3d(ctx0, Q_cur, d_head, n_head, n_time);

            if (local_attn) {
                const int  chunk         = att_left + att_right;
                const int  n_group       = (n_time + chunk - 1) / chunk;
                const int  n_time_padded = n_group * chunk;
                const int  n_kv_chunk    = chunk + window_size - 1;
                const int  n_kv_dense    = n_kv_chunk * n_group;
                const bool need_padding  = n_time_padded > n_time;

                Q_cur = ggml_cont(ctx0, ggml_permute(ctx0, Q_cur, 0, 2, 1, 3));
```
So we have already multipled cur with the current layers weights for q, and then
reshaped this for multihead attention.
```console
(gdb) p Q_cur->ne
$24 = {128, 8, 20875, 1}
```
So at this point we have 8 heads each of 128 dimensions, and we have 20875 or
these. So in memory we would have
```
0 (time 0)
   0 [0  ... 127]
   1 [0  ... 127]
   2 [0  ... 127]
   3 [0  ... 127]
   4 [0  ... 127]
   5 [0  ... 127]
   6 [0  ... 127]
   7 [0  ... 127]
...
20874 (time 20874)
   0 [0  ... 127]
   1 [0  ... 127]
   2 [0  ... 127]
   3 [0  ... 127]
   4 [0  ... 127]
   5 [0  ... 127]
   6 [0  ... 127]
   7 [0  ... 127]
```

For local attention We are now going to permute this tensor:
```
(gdb) p Q_cur->ne
$25 = {128, 20875, 8, 1}

0
   time    0 [0  ... 127]
           ...
   time 20874[0  ... 127]
...
7
   time    0 [0  ... 127]
     ...
   time 20874[0  ... 127]
```
So we now have each heads time frames consecutive in memory.

We then create two tensors, Q_u and Q_v:
```c++
    // rehaping/broadcasting.

    // baseline preference for content. "regardless of how far way a key frame
    // is, if it contains an acoustic feature I care about (like a vowel sound)
    // give it a high attention score".
    struct ggml_tensor * bias_u = ggml_reshape_3d(ctx0, layer.attn_pos_bias_u, d_head, 1, n_head);
    struct ggml_tensor * Q_u = ggml_add(ctx0, Q_cur, bias_u);

    // baseline preference for distance. "regardless of what text of audio is in
    // this that frame I inherently want to pay  more attention to frames that
    // are exactly 3 steps to my left, or 5 steps to my right".
    struct ggml_tensor * bias_v = ggml_reshape_3d(ctx0, layer.attn_pos_bias_v, d_head, 1, n_head);
    struct ggml_tensor * Q_v = ggml_add(ctx0, Q_cur, bias_v);
```
So this is creating new tensors for the added bias, one for features and one
for the distance.

Attention Score = Content Similarity + Relative Distance preference

```c++
(gdb) p Q_u->ne
$32 = {128, 20875, 8, 1}
```
Next we have:
```c++
struct ggml_tensor * Q_u_padded = need_padding ?
    ggml_pad_ext(ctx0, Q_u, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) : Q_u;
```
So if we don't need any padding we can just use Q_u from above. But if we
need padding which my current session does we call ggml_pad_ext.
So before padding we have:
```console
(gdb) p Q_u->ne
$32 = {128, 20875, 8, 1}
```

```c
struct ggml_tensor * ggml_pad_ext(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                  lp0, amount to left pad ne0
            int                  rp0, amount to right pad ne0
            int                  lp1, amount to left pad ne1
            int                  rp1, amount to right pad ne1
            int                  lp2, amount to left pad ne2
            int                  rp2, amound to right bad ne2
            int                  lp3, amount to left pad ne3
            int                  rp3  amount to right pad ne3
            ) {
```
We have lp for left-pad, and rp for right pad.
```
    ggml_pad_ext(ctx0, Q_u, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) : Q_u;
                                      ↑
                                      rp1
```
So we are saying that we want to right pad ne1 our second dimension by:
```console
(gdb) p n_time_padded - n_time
$39 = 117

(gdb) p 20875 + 117
$40 = 20992
```
And the we shape the padded Q_u:
```c++
    Q_u_padded = ggml_reshape_4d(ctx0, Q_u_padded, d_head, chunk, n_group, n_head);
```
```console
1630  Q_u_padded = ggml_reshape_4d(ctx0, Q_u_padded, d_head, chunk, n_group, n_head);
(gdb) p Q_u_padded->ne
$41 = {128, 20992, 8, 1}
(gdb) p d_head
$42 = 128
(gdb) p chunk
$43 = 256
(gdb) p n_group
$44 = 82

(gdb) p Q_u_padded->ne
$45 = {128, 256, 82, 8}
```
So now we have:
```
0  (head)
   0 (chunk group)
      0 (time frames)  [0    127]
                          ...     (hidden features of each frame) 
                    255[0    127]
      ...
   81 (chunk group)
      0 (time frames  [0    127]
                          ...
                   255[0    127]
      ...
7 (head)
   0 (chunk group)
      0 (time frames)  [0    127]
                          ...
                    255[0    127]
      ...
   81 (chunk group)
      0 (time frames) [0    127]
                         ...
                   255[0    127]
```

Next we have:
```c++
struct ggml_tensor * K_padded = ggml_pad_ext(ctx0, K_cur, 0, 0, att_left, att_right, 0, 0, 0, 0);
                                                                 ↑          ↑
                                                                 lp1       rp1
```
So this will pad both left and right pad the second dimension by 128 in our case:
```console
(gdb) p K_cur->ne
$48 = {128, 20875, 8, 1}

(gdb) p att_left
$49 = 128
(gdb) p att_right
$50 = 128
```
And recall that each audio frame wants to look exactly 128 frames into the past
and 128 frames into the future to gather context.

Lets look at the first frame, frame 0, it want to look at 128 frames into the
past and 128 frames into the future. The future is not an issue but the past
is, that is reading negative or out of bounds. And we have a similar situation
at the end. By adding 128 frames of zeros to the front and back, the timeline
for each head is safely cushioned
```console
    Linear Memory Address --->
=============================================================================================
[ HEAD 0 ]
  --> [ Ghost Past ]   --> 128 frames of pure Zeros
  --> [ Real Audio ]   --> Frames 0 to 20874 (Your actual Key features)
  --> [ Ghost Future ] --> 128 frames of pure Zeros
=============================================================================================
```
```console
(gdb) p K_cur->ne[1]
$54 = 20875
(gdb) p K_padded->ne[1]
$55 = 21131
```

Next we are checking if n_kv_dense is greater then the padded K tensor:
```c++
const int chunk      = att_left + att_right;
const int n_kv_chunk = chunk + window_size - 1;
const int n_kv_dense = n_kv_chunk * n_group;

if (n_kv_dense > K_padded->ne[1]) {
    K_padded = ggml_pad_ext(ctx0, K_padded, 0, 0, 0, n_kv_dense - K_padded->ne[1], 0, 0, 0, 0);
                                                     ↑
                                                     rp1
}
```
```console
(gdb) p n_kv_chunk  (number of Key frames required per group)
$58 = 512
(gdb) p n_group     (total number of groups we are processing)
$63 = 82
(gdb) p n_kv_chunk * n_group
$59 = 41984

(gdb) p n_kv_dense (total number of slots required if every single one of our
82 groups had its own completely independent non-overlapping window of 512 keys.
$57 = 41984

(gdb) p K_padded->ne
$61 = {128, 21131, 8, 1}
(gdb) n
1637	                struct ggml_tensor * K_chunk = ggml_view_4d(ctx0, K_padded,
(gdb) p K_padded->ne
$62 = {128, 41984, 8, 1}
```
```console
Linear Memory Address --->
=============================================================================================
[ HEAD 0 ]
  --> [ Context Past Padding ] -> 128 frames of Zeros
  --> [ Your Real Audio Keys ] -> 20,875 frames of audio data
  --> [ Context Future Padding]-> 128 frames of Zeros
  --> [ Math Alignment Padding]-> 20,853 frames of pure Zeros
=============================================================================================
```

Next we have:
```c++
    struct ggml_tensor * K_chunk = ggml_view_4d(ctx0, K_padded,
            d_head, n_kv_chunk, n_group, n_head,
            K_padded->nb[1],                  // stride for frames 512
            (size_t) chunk * K_padded->nb[1], // stride for groupd 131072
            K_padded->nb[2],                  // stride for head 21495808
            0);
    K_chunk = ggml_cont(ctx0, K_chunk);
```
```console
(gdb) p K_padded->ne         (before)
$64 = {128, 41984, 8, 1}
(gdb) p d_head
$65 = 128
(gdb) p n_kv_chunk
$66 = 512
(gdb) p n_group
$67 = 82
(gdb) p n_head
$68 = 8

(gdb) p K_padded->nb[1]
$13 = 512   bytes to move to the next frame

(gdb) p K_padded->nb[1] * chunk
$14 = 131072 bytes to move to the next group. 131072/512 = 256

(gdb) p K_padded->nb[2]
$15 = 21495808  to move to the next head.

So that will result in:
(gdb) p K_chunk->ne
$69 = {128, 512, 82, 8}
```
So we are saying that we have 128 features per frame and we have which 512
frames, 82 groups, and 8 heads. 

And we are setting nb0 to 512 (K_padded->nb[1]) bytes which makes sense as
512/4=128, so to move to the next frame we have to advance 512 bytes.
We then set nb[1] to 131072, chunk * K_padded->nb[1] which is 256 * 512. So to
move to the next group it should move 131072 bytes. But notice that this only
256 frames and not 512!
```
Group 0:
Starts at byte 0 (frame 0)
Reads 512 frames from Frame 0 to Frame 511

Group 1:
Starts at byte 131072 (frame 256).
Reads 512 frames from Frame 256 to frame 767.
```
Keep in mind that this is preparing the K tensor for usage later in the code
and this is done for computation efficiency.

We are grouping queries into chunks of 256, and we want to compute attention for
all those queries at the same time using a single ggml_mat_mul. And the attention
needs keys too.

Query 0 needs to look 128 frame backwards. It needs Key -128
Query 255 need to look 128 frames forward. It needs Key 255+128=383

The total span is:
```
From Key -128 to Key 383
383 - (-128) + 1 = 512 keys
```
And this is
```console
$69 = {128, 512, 82, 8}
```

```    
0 (head)
    0
       0   [0    127]
              ...
       511 [0    127]
    ...
    81
       0   [0    127]
              ...
       511 [0    127]
...
7 (head)
    0
       0   [0    127]
              ...
       511 [0    127]
    ...
    81
       0   [0    127]
              ...
       511 [0    127]
```

```c++
    struct ggml_tensor * content_scores = ggml_mul_mat(ctx0, K_chunk, Q_u_padded);
```
```console
(gdb) p K_chunk->ne
$31 = {128, 512, 82, 8}

(gdb) p Q_u_padded->ne
$30 = {128, 256, 82, 8}
```
So first we check that the outer dimensions match which they do. 
* Head 0, Group 0 Queries will only multiply with Head 0, Group 0 Keys.
* Head 7, Group 81 Queries will only multiply with Head 7, Group 81 Keys.

And inside one group and head we are left with two 2D grids:
* K_chunk   : [128, 512] 512 frames each with 128 hidden dimensions.
* Q_u_padded: [128, 256] 256 frames each with 128 hidden dimensions.

```console
               Key 0   Key 1   Key 2  ...  Key 256  ...  Key 511
             +-------+-------+-------+---+---------+---+---------+
    Query 0  |  Dot  |  Dot  |  Dot  |   |   Dot   |   |   Dot   | <-- Q_0 scores all 512 keys
             +-------+-------+-------+---+---------+---+---------+
    Query 1  |  Dot  |  Dot  |  Dot  |   |   Dot   |   |   Dot   |
             +-------+-------+-------+---+---------+---+---------+
    Query 2  |  Dot  |  Dot  |  Dot  |   |   Dot   |   |   Dot   |
             +-------+-------+-------+---+---------+---+---------+
      ...    |       |       |       |   |         |   |         |
    Query 255|  Dot  |  Dot  |  Dot  |   |   Dot   |   |   Dot   | <-- Q_255 scores all 512 keys
             +-------+-------+-------+---+---------+---+---------+
```
```console
(gdb) p content_scores->ne
$1 = {512, 256, 82, 8}
```
Next we have:
```c++
                struct ggml_tensor * content_scores = ggml_mul_mat(ctx0, K_chunk, Q_u_padded);
                content_scores = ggml_view_4d(ctx0, content_scores,
                        window_size, chunk, n_group, n_head,
                        (size_t) (chunk + window_size) * content_scores->nb[0],
                        content_scores->nb[2],
                        content_scores->nb[3],
                        0);
                content_scores = ggml_cont(ctx0, content_scores);
                content_scores = ggml_reshape_3d(ctx0, content_scores, window_size, n_time_padded, n_head);
                if (need_padding) {
                    content_scores = ggml_view_3d(ctx0, content_scores,
                            window_size, n_time, n_head,
                            content_scores->nb[1],
                            content_scores->nb[2],
                            0);
                }
```
Because the query window moves forward by 1 frame at every step, the valid 257
keys form a diagonal band across our 512 * 256 matrix:


With a sequence length of n_time = 20875, a standard global self-attention
mechanism would require a massive matrix of 20875 * 20875 ~ 435 million elements
per head.

Attention Score = Content Similarity + Relative Distance preference


### Simplified
* Total audio sequence: 4 frames [0, 1, 2, 3]
* Every frame can only look 1 step left and 1 step right (window_size = 3).
* We process queries in blocks of 2 (chunk = 2).

Split the queries:
We take our 4 audio frames and split them into 2 independent groups.
```console
GROUP 0: [ Query 0, Query 1 ]
GROUP 1: [ Query 2, Query 3 ]
```

Gathering the Keys (K):
This is where our overlapping window trick happens. To process Group 0's queries
simultaneously, we need to gather every single key that either Query 0 or
Query 1 might want to look at.
* Query 0 wants to look left, so it needs a Padding frame (P), Frame 0, and Frame 1.
* Query 1 wants to look right, so it needs Frame 0, Frame 1, and Frame 2.

To satisfy both queries at once, Group 0 is given a window of 4 Keys total. Look
at how Group 0 and Group 1 overlap in memory:
```console
PHYSICAL TIMELINE:  [ Padding, Frame 0, Frame 1, Frame 2, Frame 3, Padding ]
                    |------------------------------------|
                             GROUP 0 KEYS: [ P, 0, 1, 2 ]
                                        |------------------------------------|
                                                 GROUP 1 KEYS: [ 1, 2, 3, P ]
```
Notice that frames 1 and 2 are shared by both groups. The overlapping view trick
lets us recycle these frames in RAM without copying them yet.

The hardware collision:
Now, the hardware multiplies Group 0's 2 Queries against its 4 Keys. This
generates a clean 4 columns * 2 rows grid of attention scores:
```console
GROUP 0 KEYS
             Key P   Key 0   Key 1   Key 2
           +-------+-------+-------+-------+
   Query 0 | Valid | Valid | Valid | Waste |  <-- Query 0 can't look at Key 2 (too far right)
Q  --------+-------+-------+-------+-------+
   Query 1 | Waste | Valid | Valid | Valid |  <-- Query 1 can't look at Key P (too far left)
           +-------+-------+-------+-------+


Memory Address:  [0]   [1]   [2]   [3]    [4]   [5]   [6]   [7]
                 ---   ---   ---   ---    ---   ---   ---   ---
Actual Data:    | V0  | V1  | V2  |  W  |  W   | V0  | V1  | V2 |
```
The hardware did this in one single flash. We got all our valid scores, but we
also inherited those two "Waste" blocks because hardware can only output perfect
rectangles.

The stride shift trick:
We cannot pass that 4x2 grid to the Softmax because the "Waste" scores will ruin
the percentages. We need to isolate the Valid blocks.

The computer starts at Address 0 and reads 3 elements straight across:
```console
Memory Address:  [0]   [1]   [2]
                 ---   ---   ---
Virtual Row 0:  | V0  | V1  | V2|
```
This is where the stride parameter (nb[1] = 5 elements) alters the calculation.
To find where the next row starts, the pointer does not look at where Row 0
ended. It goes back to the start of Row 0 (Address 0) and adds the stride value:
```console
Next Row Address = 0 + 5 = 5 --------------------+
                                                 ↓  
Memory Address:  [0]   [1]   [2]   [3]    [4]   [5]   [6]   [7]
                 ---   ---   ---   ---    ---   ---   ---   ---
Actual Data:    | V0  | V1  | V2  |  W  |  W   | V0  | V1  | V2 |
```
Starting exactly at Address 5, the computer reads its row width of 3 elements
straight across:
```
Memory Address:  [5]   [6]   [7]
                 ---   ---   ---
Virtual Row 1:  | V0  | V1  | V2  |
```
By setting the row skip size to 5, the pointer stepped right over the waste data.
If we stack the virtual rows into a 2D matrix, it reads cleanly:
```console
Slot 0    Slot 1    Slot 2
                +--------+--------+--------+
Virtual Row 0:  |   V0   |   V1   |   V2   |  (Read from Addresses 0, 1, 2)
                +--------+--------+--------+
Virtual Row 1:  |   V0   |   V1   |   V2   |  (Read from Addresses 5, 6, 7)
                +--------+--------+--------+
```


The Chop (Q_u_padded):
We took a massive, unmanageable audio timeline and sliced it into clean,
independent 256-frame query chunks (n_group).

The Cushion (K_padded):
We padded the keys on both sides so edge frames wouldn't crash the system when
looking into the past or future.

The Overlap (K_chunk):
We used a smaller step size (256) than window size (512) so adjacent chunks
could seamlessly share context without duplicating data in RAM.

The Collision (content_scores):
We let the hardware slam the queries and keys together in parallel, giving you
all the valid scores plus some rectangular waste.

The Clean-Up (stride shift):
We changed the jump rule to 513, effortlessly stepping over the waste elements,
leaving you with a perfect, clean matrix of local attention scores ready for the
Softmax layer.

Next we have:
```console
(gdb) p content_scores->ne
$6 = {257, 256, 82, 8}
```
```
                content_scores = ggml_cont(ctx0, content_scores);
                content_scores = ggml_reshape_3d(ctx0, content_scores, window_size, n_time_padded, n_head);
```
```console
(gdb) p content_scores->ne
$11 = {257, 20992, 8, 1}
```
Recall that our original audio size was 20875 and we had to pad it to 20992. The
next part is to "undo" that padding:
```c++
                // remove padding if padding was applied (truncating to n_time).
                if (need_padding) {
                    content_scores = ggml_view_3d(ctx0, content_scores,
                            window_size, n_time, n_head,
                            content_scores->nb[1],
                            content_scores->nb[2],
                            0);
                }
```
```console
(gdb) p content_scores->ne
$16 = {257, 20875, 8, 1}
```
Next we have the relative position score, and we then calculate the attention
score and run it through softmax:
```c++
                struct ggml_tensor * rel_pos_scores = ggml_mul_mat(ctx0, pos, Q_v);

                // attention_score = content scores + relative position scores
                struct ggml_tensor * attn_scores = ggml_add(ctx0, content_scores, rel_pos_scores);

                attn_scores = ggml_soft_max_ext(ctx0, attn_scores, attn_mask, 1.0f / std::sqrt(d_head), 0.0f);
```
This matrix doesn't care about audio features. It purely evaluates: "For every
time frame (20875), how much does this specific head (8) inherently prefer looking
at a frame that is X steps away (257)?"
The softmax converts to probabilities.
```console
(gdb) p attn_scores->ne
$18 = {257, 20875, 8, 1}
```
Every single one of our 20875 timeline frames now contains a clean vector of
257 perfectly scaled, safely masked percentage weights showing exactly how much
it cares about its surrounding neighbors.

So we are going to use these with the Value tensor which currently looks like
this:
```console
(gdb) p V_cur->ne
$25 = {128, 20875, 8, 1}
```

Next we have the following with will pad the time frames to 20992 again:
```c++
                struct ggml_tensor * probs_padded = need_padding ?
                    ggml_pad_ext(ctx0, attn_scores, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) : attn_scores;
                                                              ↑
                                                              rp ne1
```
```console
(gdb) p n_time_padded
$19 = 20992
(gdb) p n_time
$20 = 20875
(gdb) p n_time_padded - n_time
$21 = 117
```
This is the same padding we applied for Q
```c++
                struct ggml_tensor * Q_u_padded = need_padding ?
                    ggml_pad_ext(ctx0, Q_u, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) : Q_u;
```
So that will pad the attention scores to be hardware friendly just like before:
```console
(gdb) p probs_padded->ne
$26 = {257, 20992, 8, 1}
```
Next we reshape that tensor in to our local attention grouping:
```c++
                probs_padded = ggml_reshape_4d(ctx0, probs_padded, window_size, chunk, n_group, n_head);
```
```console
(gdb) p probs_padded->ne
          time frames
             ↓
$30 = {257, 256, 82, 8}
        ↑
    local attention scores
```

```c++
                probs_padded = ggml_pad_ext(ctx0, probs_padded, 0, chunk, 0, 0, 0, 0, 0, 0);
                                                                    ↑
                                                                    rp ne[0]
```
```console
(gdb) p probs_padded->ne
$32 = {513, 256, 82, 8}
```
So every attention score now has 256 zeros applied to the end.
257 Valid attention scores + 256 Padding zeros = 513 total elements in a row.

```c++

                probs_padded = ggml_view_4d(ctx0, probs_padded,
                        n_kv_chunk, chunk, n_group, n_head,
                        (size_t) n_kv_chunk * probs_padded->nb[0],
                        probs_padded->nb[2],
                        probs_padded->nb[3],
                        0);
                probs_padded = ggml_cont(ctx0, probs_padded);
                probs_padded = ggml_mul(ctx0, probs_padded, local_mask);
```
```console
(gdb) p n_kv_chunk 
$2 = 512
(gdb) p chunk
$3 = 256
(gdb) p n_group
$4 = 82
(gdb) p n_head
$5 = 8
(gdb) p n_kv_chunk
$6 = 512
(gdb) p n_kv_chunk * probs_padded->nb[0]
$7 = 2048

(gdb) p probs_padded->ne
$8 = {512, 256, 82, 8}

```
The physical RAM array holds rows that are exactly 513 elements long. Our view
overrides this by claiming a row is 512 elements wide, with a row-to-row jump
size of exactly 512 elements (2048 bytes).

Because the step size (512) is 1 element shorter than the physical row length
(513), the starting pointer for each consecutive row progressively lags behind
the true physical row boundary.

window_size = 3: Every query has a clean vector of 3 valid attention percentages.
Query 0's scores are: [A, B, C]
Query 1's scores are: [X, Y, Z]
chunk = 2: We are processing 2 queries in this block.
n_kv_chunk = 4: The Value matrix runway width that we need to match.

Tail padding:
The code takes our compact 3-element rows and pads the right side by chunk (2)
Query 0 becomes: [A, B, C, 0, 0]
Query 1 becomes: [X, Y, Z, 0, 0]
```
Flat memory:
```console
Address: [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9]
         ---  ---  ---  ---  ---  ---  ---  ---  ---  ---
Data:   | A  | B  | C  | 0  | 0  | X  | Y  | Z  | 0  | 0  |
        \_____________________/  \_____________________/
                 Row 0                    Row 1
```
Reverse stride view:
We tell GGML to look at this 5-element-row tape using a virtual rulebook:
* Pretend the row width is only 4 elements (n_kv_chunk = 4)
* When jumping to the next row, skip forward by exactly 4 elements worth of bytes.
The computer starts at Address 0 and reads our requested virtual width of
4 elements:
* Slot 0 = Address 0 -> A
* Slot 1 = Addrees 1 -> B
* Slot 2 = Addrees 2 -> C
* Slot 3 = Addrees 3 -> 0

To find where Virtual Row 1 begins, the pointer goes back to the start of Row 0
(Address 0) and adds our virtual stride parameter (4):
Next Row Address = 0 + 4 = 4

Address 4 does not contain Query 1's data yet—it contains the extra trailing
zero padding left behind by Row 0.
The pointer under-shot the true physical boundary of Row 1 (Address 5) by
exactly 1 slot.
Starting exactly at this under-shot position (Address 4), the computer reads 4
elements straight across:
* Slot 0 = Address 4 -> 0
* Slot 1 = Addrees 5 -> X
* Slot 2 = Addrees 6 -> Y
* Slot 3 = Addrees 7 -> Z

```console
Slot 0   Slot 1   Slot 2   Slot 3
        +--------+--------+--------+--------+
Row 0:  |   A    |   B    |   C    |   0    |  (Read from Addresses 0, 1, 2, 3)
        +--------+--------+--------+--------+
Row 1:  |   0    |   X    |   Y    |   Z    |  (Read from Addresses 4, 5, 6, 7)
        +--------+--------+--------+--------+
```

```console
1696	                probs_padded = ggml_mul(ctx0, probs_padded, local_mask);
(gdb) p probs_padded->ne
$10 = {512, 256, 82, 8}
(gdb) p local_mask->ne
$9 = {512, 256, 1, 1}
```

```
        +-------------------------------------------------------+
Row 0:  | [257 Probabilities] | [255 absolute, hard-masked 0s]  |
        +-------------------------------------------------------+
Row 1:  | [1 Zero] | [257 Probabilities] | [254 hard-masked 0s] |
        +-------------------------------------------------------+
Row 2:  | [2 Zeros] | [257 Probabilities] | [253 hard-masked 0s]|
        +-------------------------------------------------------+
```
Lets take a look at how the local_mask is populated later:
```c++
    // set local attention skew mask
    if (struct ggml_tensor * local_mask = ggml_graph_get_tensor(gf, "local_mask")) {
        const int n_k = local_mask->ne[0]; // 512 columns
        const int n_q = local_mask->ne[1]; // 256 rows

        std::vector<float> mask_data(n_q * n_k); // 131072
        const int window_size = n_k - n_q + 1; // 257
        // loop over all 256 rows
        for (int q = 0; q < n_q; ++q) {
            // loop over each column (512)
            for (int k = 0; k < n_k; ++k) {
                const int rel = k - q;
                // if we are inside of the the window then set to 1.0, otherwise 0.0f.
                mask_data[q * n_k + k] = (rel >= 0 && rel < window_size) ? 1.0f : 0.0f;
            }
        }
        ggml_backend_tensor_set(local_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }
```
Lets take a smaller example:
n_k (Columns) = 4
n_q (Rows) = 2
window_size = $4 - 2 + 1 = 3
```
         k=0    k=1    k=2    k=3
       +------+------+------+------+
q = 0: | 1.0f | 1.0f | 1.0f | 0.0f |
       +------+------+------+------+
q = 1: | 0.0f | 1.0f | 1.0f | 1.0f |
       +------+------+------+------+
```
When the engine executes ggml_mul(ctx0, probs_padded, local_mask), it performs
an element-wise multiplication. It lines up our virtual tensor view with this
physical mask grid and multiplies them together:
```console
OUR STRIDE-SHIFTED PROBABILITIES             THE LOCAL MASK GRID
       +----+----+----+----+                  +------+------+------+------+
Row 0: | A  | B  | C  | 0  |        X         | 1.0f | 1.0f | 1.0f | 0.0f |
       +----+----+----+----+                  +------+------+------+------+
Row 1: | 0  | X  | Y  | Z  |                  | 0.0f | 1.0f | 1.0f | 1.0f |
       +----+----+----+----+                  +------+------+------+------+
```
So the probabilites are generated/calculated by a backend, which may have
for some reason not produded a 0 for a waste value, prehaps it was a small
decimal number instead. By multiplying by the local mask we are cleaning, or
making sure that if that was the case we have still have valid probabilites
after this operation.

Next we have:
```c++
                struct ggml_tensor * V_padded = ggml_pad_ext(ctx0, V_cur, 0, 0, att_left, att_right, 0, 0, 0, 0);
                if (n_kv_dense > V_padded->ne[1]) {
                    V_padded = ggml_pad_ext(ctx0, V_padded, 0, 0, 0, n_kv_dense - V_padded->ne[1], 0, 0, 0, 0);
                }
                V_padded = ggml_cont(ctx0, ggml_permute(ctx0, V_padded, 1, 0, 2, 3));
                struct ggml_tensor * V_chunk = ggml_view_4d(ctx0, V_padded,
                        n_kv_chunk, d_head, n_group, n_head,
                        V_padded->nb[1],
                        (size_t) chunk * V_padded->nb[0],
                        V_padded->nb[2],
                        0);
                V_chunk = ggml_cont(ctx0, V_chunk);

                cur = ggml_mul_mat(ctx0, V_chunk, probs_padded);
```
