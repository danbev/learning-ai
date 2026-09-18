## Multidimensional RoPE (M-RoPE)
When using images or video with a transformer when using RoPE this will flatten
the visual patches into a 1D sequence which breaks spatial geometry.

Pixels below or above might be separated by an entire row length in 1D position
space.
```console
            col 0       col 1       col 2
row 0:   [ P(0,0) ]  [ P(0,1) ]  [ P(0,2) ]
row 1:   [ P(1,0) ]  [ P(1,1) ]  [ P(1,2) ]
row 2:   [ P(2,0) ]  [ P(2,1) ]  [ P(2,2) ]
```
So here we can see that P(0,0) is right next to P(0,1) horizontally and also
vertically it has P(1.0) below.

A standard vision encoder flattens this 2D grid, row by row, left to right
```console
Index:    0          1          2          3          4          5          6          7          8
Patch: [ P(0,0) ] [ P(0,1) ] [ P(0,2) ] [ P(1,0) ] [ P(1,1) ] [ P(1,2) ] [ P(2,0) ] [ P(2,1) ] [ P(2,2) ]
           |                                 ↑
           +---------------------------------+
                vertical neighbor
```

So in this case we have 9 "tokens" even through we are dealing with an image.
```console
token 0: [0 ... d]   P(0,0)
token 1: [0 ... d]   P(0,1)
token 2: [0 ... d]   P(0,2)
token 3: [0 ... d]   P(1,0)
token 4: [0 ... d]   P(1,1)
token 5: [0 ... d]   P(1,2)
token 6: [0 ... d]   P(2,0)
token 7: [0 ... d]   P(2,1)
token 8: [0 ... d]   P(2,2)

sequence lenght: 9
channel dimension: d
```

So horizontal pairs are still just the same distance away, but the vertical pairs
are row width apart.

Standard [RoPE](rope.md) rotates the query and keys so that thier attention
scores depend on the relative distance.
```console
∆m = |pos_i - pos_j|
```
The high frequence bands in RoPE decay quickly as ∆m grows. So horizontal neighbors
have ∆m = 1 and attention will treat them as tight local context. But for the
vertical neighbors which have ∆m = 3 (but would be something like 28 in a really
world use case) so RoPE subjects them to high frequency phase oscilations and
decay treating them like tokens in a different sentence.

Is is also possible for vision models to have arbitary image resolutions, so one
may format patches as 14x56 instead of 28x28 so the vertical step size jumps
from 28 to 56.

What M-RoPE does is take the grid from above:
```console
P(0,0)  P(0,1)  P(0,2)
P(1,0)  P(1,1)  P(1,2)
P(2,0)  P(2,1)  P(2,2)
```
So like we looked at before lets focus on P(0,0):
```console
P(0,0)                      height 0, width: 0  (h: 0, w: 0)
P(0,1) horizontal neighbor  height 0, width: 1  (h: 0, w: 1)
P(0,1) vertical neighbor    height 1, width: 0  (h: 1, w: 0)
```

Recall that in RoPE the head dimension is d/2 which is a 2d rotational rotary
pairs. For M-RoPE we still have pairs but they are now subsets of height and
width:
```console
                 d
|S_h| + |S_w| = --
  ↑              2
  |
Cardinality (the total number of elements in this set)

Example:
|S_h| = 16, |S_w| = 16
 16 + 16 = 32 = d/2
```

For a query or key component pair i:
* if i is part of S_h then we rotate the row index by h
```console
R_θ_i_h = (cos(hθ_i)  -sin(hθ_i)
          (sin(hθ_i)   cos(hθ_i)

P(0,0), h=0 -> angle = 0 * θ_i = 0   (no rotation)
P(0,1), h=0 -> angle = 0 * θ_i = 0   (no rotation)
P(1,0), h=1 -> angle = 1 * θ_i = 0_i (rotated by 1 x θ_i)
```

```console


R_θ_i_w = (cos(wθ_i)  -sin(wθ_i)
          (sin(wθ_i)   cos(wθ_i)

P(0,0), w=0 -> angle = 0 * θ_j = 0   (no rotation)
P(0,1), w=1 -> angle = 1 * θ_j = θ_j (rotated by 1 * θ_j)
P(1,0), w=0 -> angle = 0 * θ_j = 0   (no rotation)

```

Lets continue with our 3x3 grid from above and lets say that our channel/feature
dimension is 4. So every token has a 4 element vector:
```console
  [q₀]
  [q₁]
  [q₂]
  [q₃]
```
And recall that RoPE operates on pairs of number of this 4 element vector:
```console
Pair₀ = (q₀, q₁)
Pair₁ = (q₂, q₃)
```
Now, this is where the S_h an S_w sets come into play. S_h will rotate (q₀, q₁)
by the angle of h and S_w will rotate (q₁, q₂) by the angle of w.

```console
                       h w
token 0: [0 ... d]   P(0,0)
token 1: [0 ... d]   P(0,1)
token 2: [0 ... d]   P(0,2)
token 3: [0 ... d]   P(1,0)
token 4: [0 ... d]   P(1,1)
token 5: [0 ... d]   P(1,2)
token 6: [0 ... d]   P(2,0)
token 7: [0 ... d]   P(2,1)
token 8: [0 ... d]   P(2,2)

```
Now, lets look at our grid again and use a dimension of 4:
```console
            
   [ P(0,0) ]       [ P(0,1) ]       [ P(0,2) ]
   [ P(1,0) ]       [ P(1,1) ]       [ P(1,2) ]
   [ P(2,0) ]       [ P(2,1) ]       [ P(2,2) ]

   [ 0,  1,  2,  3] [ 4,  5,  6,  7] [ 8, 9, 10,  11]
        token 0         token 1         token 2
   [12, 13, 14, 15] [16, 17, 18, 19] [20, 21, 22, 23]
        token 3         token 4         token 5
   [24, 25, 26, 27] [28, 29, 30, 31] [32, 33, 34, 35]
        token 6         token 7         token 8

Flattened:
[0, 1, 2, 3][4, 5, 6, 7][8, 9, 10, 11][12, 13, 14, 15][16, 17, 18, 19][20, 21, 22, 23][24, 25, 26, 27] [28, 29, 30, 31][32, 33, 34, 35]
  token 0     token 1     token 2       token 3        token 4            token 5          token 6        token 7          token 8
```
Recall that in RoPE each pair that we rotate has its own frequency, so θ₀ for
pair 0, and θ₁ for pair one.
In standard RoPE we have the flattened index m ε {0, 1, 2,...,8} and it rotates
both pairs by m:
```console
Token 0 (Index m = 0):
  Pair₀: rotated by 0 · θ₀
  Pair₁: rotated by 0 · θ₁

Token 1 (Index m = 1) — 1 step right in the image:
  Pair₀: rotated by 1 · θ₀
  Pair₁: rotated by 1 · θ₁

Token 3 (Index m = 3) — 1 step down in the image:
  Pair₀: rotated by 3 · θ₀
  Pair₁: rotated by 3 · θ₁
```
In RoPE, the dot product between Token A and Token B depends on the relative
phase shift ∆m = m_b - m_a:
```console
Token 0 attending to Token 1:
∆m = m_token1 - m_token0
∆m = 1 - 0 = 1
Pair_0 shifts by 1 * θ
Pair_1 shifts by 1 * θ
So the attention score will treat token 1 as an immediate neighbor

Token 0 attending to Token 3:
∆m = m_token3 - m_token0
∆m = 3 - 0 = 3
Pair_0 shifts by 3 * θ
Pair_1 shifts by 3 * θ
So the attention score will treat token 3 as three times further away than
token 1, even though token 3 is also "touching" token 0 but vertically.
```

Now, lets look what happens with M-RoPE:
```console
Token 0: h = 0, w = 0
  Pair₀ (Height): rotated by 0 · θ₀
  Pair₁ (Width):  rotated by 0 · θ₁

Token 1: h = 0, w = 1 (1 step right)
  Pair₀ (Height): rotated by 0 · θ₀
  Pair₁ (Width):  rotated by 1 · θ₁

Token 3: h = 1, w = 0 (1 step down)
  Pair₀ (Height): rotated by 1 · θ₀
  Pair₁ (Width):  rotated by 0 · θ₁
```

Nothing changes to the actual flattened array it is the same as in standard
RoPE. Instead we have this metadata associated with a token with its (h, w)
coordinates.

But we also have to take into account the dot product when using m-rope.
We effectively split the token into two independant sensors, one vertical sensor
(S_h) that measures row differences ∆h, and a horizontal (S_w) that measures the
difference in column ∆w.

So our token vector consists of two halves:
```console
token = [q₀, q₁, q₂, q₃]

q = [q_h]  k= [k_h]
    [q_w]     [k_w]
```
When we compute the dot product between two tokens we do following, and recall
that the dot product of two vectors is just the sum of their sub-vector dot
products so we can do this computation using two terms (both dot product)
```console
Token A . Token B

q_A^T k_B = (q_A^T_h k_B_h) + (q_A^T_w k_B_w)
             vertical attn     horizontal attn
             score              score
```
```console
// For a token at grid position (h, w) with query vector q:
// 1. Rotate the height channels (Pair 0) by angle = h * theta_0
float new_q0 = q[0] * cos(h * theta_0) - q[1] * sin(h * theta_0);
float new_q1 = q[0] * sin(h * theta_0) + q[1] * cos(h * theta_0);

// 2. Rotate the width channels (Pair 1) by angle = w * theta_1
float new_q2 = q[2] * cos(w * theta_1) - q[3] * sin(w * theta_1);
float new_q3 = q[2] * sin(w * theta_1) + q[3] * cos(w * theta_1);

// Write back in-place
q = [new_q0, new_q1, new_q2, new_q3];
```
And we do the same with the k vector. So the complete timeline is:
* Compute projections: q = xW_q, k = xW_k
* Apply M-RoPE, rotate the first half of q and k by h, and the second half by w.
* Run standard attention: dot(q, k) / sqrt(d)


```console
q = [q₀, q₁, q₂, q₃]

Pair 0 = (q₀, q₁)  --> Height sensor (h), rotated by θ₀
Pair 1 = (q₂, q₃)  --> Width sensor  (w), rotated by θ₁
```
θ₀ and θ₁ are not the same speed. In standard RoPE the frequencies decrease
accross the pairs, so θ₀ > θ₁. We can think of θ₀ as the seconds hand of a clock
which spins fast, and θ₁ the hour hand that turns very slowly.
```console
Pair 0 (Height) gets θ₀ (the fast seconds hand of the clock)
Pair 1 (Width)  gets θ₁ (the slow hour hand of the clock)
```
In the physical image, moving 1 patch down and 1 patch right are the exact same
physical distance. But they will use different frequencies.

On in our examples above we only had 2 pairs but in real situations/models we
will have many pairs, for example 4 pairs in an 8 element vector. And like we
said the frequencies decrease from higher frequency to lower:
```console
θ₀ (fastest) > θ₁ (fast) > θ₂ (slow) > θ₃ (slowest)
```

```console
Index:      Pair 0      Pair 1       Pair 2      Pair 3
Speed:     [Fastest]   [ Fast ]     [ Slow ]    [Slowest]
            \________  _______/      \_________  _______/
                     v                         v
               Height Sensor             Width Sensor
```
So if height gets {θ₀, θ₁} only fastish hands, it can see vertical details just
fine but over larger vertical distances the angles spin out of control.
And similar for width but the opposite, it can track tokens far apart but will
have trouble with close neighbors.

Interleaved M-RoPE (IM-RoPE) instead does the following:
```console
Index:      Pair 0      Pair 1       Pair 2      Pair 3
Speed:     [Fastest]   [ Fast ]     [ Slow ]    [Slowest]
               |          |            |           |
               v          v            v           v
             Height     Width        Height      Width
```
Height gets {θ₀, θ₂}. Width gets  {θ₁, θ₃}.

```c++
    // example M-RoPE:
    //  given sections = [t=4, y=2, x=2, 0]
    //  given a single head with size = 18 --> [000000000000000000]
    //  GGML_ROPE_TYPE_MROPE   n_dims = 16 --> [ttttyyxxttttyyxx00] (cos/sin are applied in NEOX ordering)
    //  GGML_ROPE_TYPE_IMROPE  n_dims = 16 --> [ttyxttyxttyxttyx00] (interleaved M-RoPE, still NEOX ordering)
    //  note: the theta for each dim is computed the same way as ggml_rope_ext, no matter the section
    //        in other words, idx used for theta: [0123456789... until n_dims/2], not reset for each section
    GGML_API struct ggml_tensor * ggml_rope_multi(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            struct ggml_tensor  * b,
            struct ggml_tensor  * c,
            int                   n_dims,
            int                   sections[GGML_MROPE_SECTIONS],
            int                   mode,
            int                   n_ctx_orig,
            float                 freq_base,
            float                 freq_scale,
            float                 ext_factor,
            float                 attn_factor,
            float                 beta_fast,
            float                 beta_slow);
```
Normal layout which is what we have been using for our pairs in our examples
group pairs next to each other in memory:
```console
normal layout : (q[0], q[1]) (q[2], q[3])
```
NEOX ordering which is used by GPT-NeoX, LLama, Qwen etc) split the vector down
the middle instead:
```console
                         +------------+
                         |            |
normal layout : (q[0], q[1]) (q[2], q[3])
                   |            |
                   +------------+

neox layout   : (q[0], q[2]) (q[1], q[3])
                  pair 0        pair 1
```
In the comment above we have a head_size of 18, so we have 18 floats in memory,
and n_dims is 16. So only the first 16 floats undergo RoPE:
```
 [0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16 17]
                                                         0  0 
16 elements, 8 pairs.
sections = [t=4, y=2, x=2, 0]
time  : 4 pairs
height: 2 pairs
width : 2 pairs
```
In the original (block-based) M-RoPE the pairs would be partitioned as:
```console
 [0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15]

```

Neox ordering the pairs are never adjacent in memory:
```console
First half of memory: indices 0, 1,  2,  3,  4,  5,  6,  7
Second half         : indices 8, 9, 10, 11, 12, 13, 14, 15

Pair 0: (q[0], q[8])              t
Pair 1: (q[1], q[9])
Pair 2: (q[2], q[10])
Pair 3: (q[3], q[11])
Pair 4: (q[4], q[12])
Pair 5: (q[5], q[13])
Pair 6: (q[6], q[14])
Pair 7: (q[7], q[15])

q[0]  is in Pair 0 (t) -> t
q[1]  is in Pair 1 (t) -> t
q[2]  is in Pair 2 (t) -> t
q[3]  is in Pair 3 (t) -> t
q[4]  is in Pair 4 (y) -> y
q[5]  is in Pair 5 (y) -> y
q[6]  is in Pair 6 (x) -> x
q[7]  is in Pair 7 (x) -> x

Result of first half: ttttyyxx

q[8]  is in Pair 0 (t) -> t
q[9]  is in Pair 1 (t) -> t
q[10] is in Pair 2 (t) -> t
q[11] is in Pair 3 (t) -> t
q[12] is in Pair 4 (y) -> y
q[13] is in Pair 5 (y) -> y
q[14] is in Pair 6 (x) -> x
q[15] is in Pair 7 (x) -> x

Result of first half: ttttyyxx

Together            : ttttyyxxttttyyxx00
GGML_ROPE_TYPE_MROPE: ttttyyxxttttyyxx00
```

Now, lets look at IM-RoPE. We can see that we have the sections [t=4, y=2, x=2, 0]
which produces ttyxttyxttyxttyx00. So we must allocate 4 pairs to t, 2 pairs to
y and 2 pairs to x.
In block mode (MRoPE) we grouped them contiguously:
```console
Pairs: [t, t, t, t, y, y, x, x]
```
In interleaved mode we distribute them evenly across all 8 pairs by repeating
the unit pattern [t, t, y x]
```console
Pair 0:    t    θ₀ (Highest fequency)
Pair 1:    t    θ₁ (High)
Pair 2:    y    θ₂ (High)
Pair 3:    x    θ₃ (Mid-High)
Pair 4:    t    θ₄ (Mid-Low)
Pair 5:    t    θ₅ (Low)
Pair 6:    y    θ₆ (Low)
Pair 7:    x    θ₇ (Lowest)

t pairs: 0, 1, 4, 5
y pairs: 2, 6
x pairs: 3, 7

Pair 0: (q[0], q[8])              t
Pair 1: (q[1], q[9])
Pair 2: (q[2], q[10])
Pair 3: (q[3], q[11])
Pair 4: (q[4], q[12])
Pair 5: (q[5], q[13])
Pair 6: (q[6], q[14])
Pair 7: (q[7], q[15])

q[0]  is in Pair 0 (t) -> t
q[1]  is in Pair 1 (t) -> t
q[2]  is in Pair 2 (y) -> y
q[3]  is in Pair 3 (x) -> x
q[4]  is in Pair 4 (t) -> t
q[5]  is in Pair 5 (t) -> t
q[6]  is in Pair 6 (y) -> y
q[7]  is in Pair 7 (x) -> x

Result of first half: ttyxttyx

q[8]  is in Pair 0 (t) -> t
q[9]  is in Pair 1 (t) -> t
q[10] is in Pair 2 (y) -> y
q[11] is in Pair 3 (x) -> x
q[12] is in Pair 4 (t) -> t
q[13] is in Pair 5 (t) -> t
q[14] is in Pair 6 (y) -> y
q[15] is in Pair 7 (x) -> x

Result of second half: ttyxttyx

Together             : ttyxttyxttyxttyx00
GGML_ROPE_TYPE_IMROPE: ttyxttyxttyxttyx00
```



__wip__


* if i is part of S_w then we rotate the row index by w


M-RoPE (Multi-dimensional RoPE) is used by vision-language models to encode
position across multiple dimensions simultaneously. Instead of a single position
integer per token, each token carries 3 position values. 

For example:
```
[temporal, height, width]
```
This lets image patches carry their 2D spatial coordinates (row, column) while
text tokens carry their sequential position.
