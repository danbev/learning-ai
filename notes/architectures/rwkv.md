## Receptance Weight Key Value (RWKV): Reinventing RNNs for the Transformer era
Simliar to [Mamba](./mamba.md) this model architecture avoids the transformers
O(N²) complexity by using an RNN (or an SSM).

So with this, like Mamba, we have efficient training and also efficient
inference (which scales linearly with the length of the sequence and not
exponentially like it does with transformers). And there are no approximation
techniques like in the Performer/Reformer/Linear Transformer (perhaps others)
either instead this is exact.

It is possible to [decompose attention](../transformer.md#decomposing-as-vector-operations)
which is a way to work around the memory limits of having the Q, K, and V
matrices stored in memory, though it might mean more computation time as we
have to compute then sequentially and not in parallel.

There are currently 6 versions of the RWKV model and this document will focus
on v5 (Eagle) and v6 (Finch) which are the latest versions as of this writing
and also the version that are implemented in llama.cpp.

_I initially started this document reading the RWKV-4 paper which is why at the
moment there are still notes that are based on that paper. This will be updated
as I go through the new paper_.

Lets take a look at inference in this architecture:
```
input sequence   = ["Dan"        "loves"      "ice"         "cream"    ]
input tokens     = [ 223         45,          1212            67       ]
input embeddings = [ [0.5, -03], [0.7, 0.2], [-0.1, 0.8], [ 0.3, -0.5] ]

           +----------------------+
           |   input_embeddings   |
           +----------------------+
                      ↓
           +----------------------+
           |       LayerNorm      |
           +----------------------+
                      |
                      +-----------------------------+
Time mixing           ↓                             |
           +----------------------+                 |   
           |       LayerNorm      |                 |
           +----------------------+                 |
                      ↓                             |
           +--------------------------------------+ |
           |               μ                      | |
           | G = (μ_g ⊙ x_t + (1 - μ_g) ⊙ x_t-1)Wg| |
           | R = (μ_r ⊙ x_t + (1 - μ_r) ⊙ x_t-1)Wr| |
           | K = (μ_k ⊙ x_t + (1 - μ_k) ⊙ x_t-1)Wk| |
           | V = (μ_v ⊙ x_t + (1 - μ_v) ⊙ x_t-1)Wv| |
           +--------------------------------------+ |
              |      |    |        |        |       |
           +-----+ +----+ +--+  +-----+  +-----+    |       
           |  G  | | R  | |w |  |  K  |  |  V  |    |
           +-----+ +----+ +--+  +-----+  +-----+    |
              |       |    |      |        |        |  
            +-------+ |    | +------------------+   |
            | SiLU  | |    +-|   WKV Operator   |   |
            +-------+ |      +------------------+   |
               |      |               |             |
               |      +------------->(*)            |
               |                      |             |
               |           +----------------------+ |   
               |           |       LayerNorm      | |
               |           +----------------------+ |
               |                      |             |
               |                      |             |
               |          +-----+     |             |
               +----------| *Wo |-----+             |
                          +-----+                   |
                             |                      |
                          +-----+                   |
                          | out |                   |
                          +-----+                   |
                             |                      |
                            (+)---------------------+
                             |
                             +----------------------+
Channel mixing               |                      |
                 +----------------------+           |
                 |       LayerNorm      |           |
                 +----------------------+           |
                             |
           +--------------------------------------+ |
           |               μ'                     | |
           | G'= (μ_g ⊙ x_t + (1 - μ_g) ⊙ x_t-1)Wg| |
           +--------------------------------------+ |
                 |                   |              |
            +------+            +-------------+     |
            |  G'  |            |  MLP        |     |
            +------+            +-------------+     |
                 |                   |              |
            +-------+                |              |
            |Sigmoid|                |              |
            +-------+                |              |
                 |                   |              |
                 |     +------+      |              |
                 +-----|      |------+              |
                       +------+                     |
                          |                         |
                         (+)------------------------+
                          |
                          |

``` 

### Linear Interpolation (lerp) in Eagle (RWKV-5)
This is pretty much the same as in RWKV-4 so I've kept my notes from that
and I'll follow up with the notation used in the RWKV-5 paper after this.
```
input embeddings = [ [0.5, -03], [0.7, 0.2], [-0.1, 0.8], [ 0.3, -0.5] ]

x_t   = [0.5, -0.3]
x_t-1 = [ 0.0, 0.0] (first token so there is no previous token)
μ_r   = [0.2, 0.9]

Linear interpolation recap:
result = t * a + (1 - t) * b

The following is doing a linear interpolation between the current token and the
previous token using the μ_r vector:
R = μ_r ⊙ x_t + (1 - μ_r) ⊙ x_t-1

R = [0.2, 0.9] ⊙ [0.5, -0.3] + (1 - [0.2, 0.9]) ⊙ [0.0, 0.0]
R = [0.1, -0.27] + (1 - [0.2, 0.9] ⊙ [0.0, 0.0]
R = [0.1, -0.27] + [0.8, 0.1] ⊙ [0.0, 0.0]
R = [0.1, -0.27] + 0
R = [0.1, -0.27]

μ_r   = learnable parameter vector.
x_t   = current token embedding.
x_t-1 = previous token embedding.
⊙     = element-wise multiplication (Hadamard product).
```
The same operations are done for the K and V vectors but they have their own
`μ_k` an `μ_v` vectors.
If a value in these mu vectors is 1 then the current value of the token
embeddings is used. And if it is 0 then the previous value of the token
embedding would be used. And any value in between would be a linear
interpolation between the two.

In the RWKV-5 paper the notation is a little different:
```
lerp_ם(a, b) = a + (b -a ) ⊙  μ_ם

a = x_t-1
b = x_t

lerp_ם(x_t-1, x_t) = x_t-1 + (x_t - xt_-1) ⊙  μ_ם
                     x_t-1 + (x_t ⊙ µ□) - (x_t-1 ⊙ µ□)
                     (x_t ⊙ µ□) + (x_t-1 ⊙ (1 - µ□))
                     µ□ ⊙ x_t + (1 - µ□) ⊙ x_t-1

ם = one of the μ vectors (g, r, k, v)
```
But it is really the same thing, just different notation. Also note that there
is an matrix multiplication in the first equation which is specific for each
μ vector (this can be seen in the diagram above).

### Data-Dependent Linear Interpolation (ddlerp) in Finch (RWKV-6)
Now this is a new concept in the RWKV-6 paper and it is used to calculate the
receptance, key, and value vectors.
There is a function named LoRA (Low Rank Adaptation) which I written about in
[lora.md](../lora.md) and I was somewhat confused about its usage here. My
understanding was that LoRA was used to reduce the dimensionality of matrices
and I did not see how that would be applicable here. In the RWKV-6 paper it is
not used for parameter reduction but instead for data-dependent linear shift
mechanism. Is is called LoRA because of the similar structure of the LoRA update
function:

```
lora□(x) = λ□ +tanh(xA□)B□

lora(x) = λr + tanh(xAr) Br

λr is a learnable vector
Ar and Br are small learnable matrices
x is the input

lora□(x) = λ□ +tanh(xA□)B□
ddlerp□(a,b) = a + (b − a) ⊙ lora□(a +(b − a) ⊙ µx)
```
Note that there is a µx vector which is trained and this is used for g, r, k,
and v. And notice that the vectors and matrices in the lora function are
specific to the current component (g, r, k, v).

For example:
```
input sequence   = ["Dan"        "loves"      "ice"         "cream"    ]
input tokens     = [ 223         45,          1212            67       ]
input embeddings = [ [0.5, -0.3], [0.7, 0.2], [-0.1, 0.8], [ 0.3, -0.5] ]

Let's assume:
D = 2 (embedding dimension)
μr = [0.6, 0.7]
λr = [0.1, 0.2]
Ar = [[0.1, 0.2], [0.3, 0.4]]
Br = [[0.5, 0.6], [0.7, 0.8]]

For the second token "loves":

a = x_t   = [0.7, 0.2]   (current embedding for "loves")
b = x_t-1 = [0.5, -0.3]  (previous embedding for "Dan")

Step 1: Calculate (b - a) ⊙ μr
[0.5, -0.3] - [0.7, 0.2] = [-0.2, -0.5]
[-0.2, -0.5] ⊙ [0.6, 0.7] = [-0.12, -0.35]

Step 2: Add this to a
[0.7, 0.2] + [-0.12, -0.35] = [0.58, -0.15]

Step 3: Apply LoRA function
x = [0.58, -0.15]
tanh(xAr) = tanh([0.58*0.1 + (-0.15)*0.3, 0.58*0.2 + (-0.15)*0.4])
           = tanh([0.013, 0.056])
           ≈ [0.013, 0.056]

tanh(xAr)Br = [0.013*0.5 + 0.056*0.7, 0.013*0.6 + 0.056*0.8]
             = [0.0455, 0.0526]

lora(x) = [0.1, 0.2] + [0.0455, 0.0526] = [0.1455, 0.2526]

Step 4: Final ddlerp calculation
ddlerpr(a, b) = [0.7, 0.2] + ([-0.2, -0.5] ⊙ [0.1455, 0.2526])
              = [0.7, 0.2] + [-0.0291, -0.1263]
              = [0.6709, 0.0737]
```

### Eagle (RWKV-5) Time mixing
The forumla given in the paper looks like this:
```
□t = lerp□(xt ,xt−1) W□, 

□  = ∈ {r ,k, v, g }
Example:
r_t = lerp_r(xt ,xt−1) W_r 

This is represented by the μ "box" in the diagram above.

w = exp(−exp(ω))
This is represented by the w "box" in the diagram above.

                                 t-1
wkv_t = diag(u) * K_t^T * v_t +   Σ  diag(w)^t-1-i * K_i^T * v_i 
                                 i=1

```
The diag function is creating a diagonal matrix from a vector. And the u vector
is a learned parameter that is part of the Weighted Key Value (WKV) computation.
The "time-first" u is initialized to
```
r0(1 − i/(D−1)) + 0.1((i + 1) mod 3)
```
This is represented by the WKV Operator "box" in the diagram above.
So, `diag(u)` will create a matrix where the diagonal is the u vector and the
rest of the matrix is zeros. This matrix will be multipled by the transpose of
the K matrix which contains key values after the linear interpolation. And
that will then be multiplied by the V matrix which contains the values after
the linear interpolation also.

To this we add the sum all the past tokens up to the current token but not
includig the current token (t-1):
```
 t-1
  Σ  diag(w)^t-1-i * K_i^T * v_i 
 i=1
```
`K_i^T * v_i` is the key and value product for that token.
`diag(w)^t-1-i` is the decay factor applied to that token. Notice that tokens in
the past will have a larger exponent value which will make the decay factor
resulting in more decay for those tokens.

Alright I don't quite understand what the above formula is doing and I need to
break this down a little so let walk through this with an example.
```
Embedding dim  : 4
Attention heads: 2 (each head will deal with 4/2 = 2 dimensions)

Left hand term: diag(u) * K_t^T * v_t
u       = [0.9 0.7]
diag(u) = [ 0.9 0.0]
          [ 0.0 0.7]

K_t     = [0.3, 0.5]  (row vector)
K_t^T   = [0.3]
          [0.5]

v_t     = [0.2 0.4]   (row vector)

So lets start with computing K_t^T * v_t:
   [0.3] [0.2 0.4] = [0.06 0.12]
   [0.5]             [ 0.1  0.2] 


The we multiply this with diag(u):
   [ 0.9 0.0] [0.06 0.12] = [0.054 0.108]
   [ 0.0 0.7] [ 0.1  0.2]   [ 0.07  0.14]


w       = [0.8, 0.6] (decay factor)
k_t-1   = [0.4, 0.2] (key vector from previous step (t-1))
k_t-1^T = [0.4]
          [0.2]
v_t-1   = [0.1, 0.3]  (value vector from previous time step (t-1))

diag(w) = [0.8 0.0]
          [0.0 0.6]

We start with computing K_t-1^T * v_t-1:
[0.4] [0.1 0.3] = [0.04 0.12]
[0.2]             [0.02 0.06]

Then we multiply this with diag(w):
[0.8 0.0] [0.04 0.12] = [0.032 0.096]
[0.0 0.6] [0.02 0.06]   [0.012 0.036]

So both the left and right hand produce 2x2 vectors which are then added:

   [0.054 0.108]  + [0.032 0.096] = [0.086 0.204]
   [ 0.07  0.14]    [0.012 0.036]   [0.082 0.176]
```

The output will be the result of all the heads concatenated together and then
multipled by `W_o`.
```
o_t = concat( SiLU(g_t) ⊙ LayerNorm(r_t * wkv_t)) W_o
```
Now the result from wkv operation above (`wkv_t`) will be multiplied by the
retention vector (`r_t`) which is calculated in the previous step. This is a
learned parameter and is used to control how much of the information is retained
from the previous time step. This is then passed through the LayerNorm function
and then multipled, element wise, by the result of the SiLU activation function
applied to the `g_t` vector. And like we mentioned above this done for all heads
which are then concatenated together and multipled by `W_o`.

### Eagle (RWKV-5) Channel mixing)
TODO:

_wip_


### WKV Operator
Now in RWKV instead of using the Q, K, and V matrices the formula looks like
this:
```
                 Σ exp(w_t_i+ k_i) . v_i
Att+(W, K, V)_t = -----------------
                 Σ exp(w_t_i+ k_i)

t = current token (current time step of position of the token in the sequence).
i = previous token (previous time step of position of the token in the sequence).
```
Notice that we are still taking a weighted sum of the values, but we are using
weights that are learned during training, and not the Query values. The keys
are still the same as in the original attention mechanism and contain
information about the current token, but the Query is gone. And notice also that
the operation is addition and not multiplication before the exponentiation.

And notice that this we can see the softmax operation in the forumula above:
```
                 Σ exp(w_t_iᵢ+ k_i) 
                 -----------------
                 Σ exp(w_t_iᵢ+ k_i)
```

So `w` is a learned vector and is called a time decay factor which controls how
quickly the influence of previous tokens decays.

So each entry in this vector, which would have the same size as the models
embedding dimensions/channels, would determine how important each feature is
over time:
```
w_t_i = −(t − i) w

t = current token (current time step of position of the token in the sequence).
i = previous token (previous time step of position of the token in the sequence).
w = learned decay vector where each entry is constraied to be non-negative.
```
So for each channel/feature in the embedding there is an entry in the `w`
vector.
```
Token sequence length = 4
Embedding dimension   = 2


Lets say the learned w looks like this (remember that these values must be
non-negative):

   w = [0.2, 0.9]

First token in the sequence (there is not previous token for this entry)
w_0_0 = -(0 - 0) * 0.2 = 0
w_0_1 = -(0 - 0) * 0.9 = 0

w_1_0 = -(1 - 0) * 0.2 = -0.2
w_1_1 = -(1 - 0) * 0.9 = -0.9

w_2_0 = -(2 - 0) * 0.2 = -0.4
w_2_1 = -(2 - 0) * 0.9 = -1.8

w_3_0 = -(3 - 0) * 0.2 = -0.6
w_3_1 = -(3 - 0) * 0.9 = -2.7
```
Now we need to keep in mind that these values will then be added to the
respective key vectors for each token in the sequence.
```
    Σ exp(w_t_i + k_i)
    -----------------
    Σ exp(w_t_i + k_i)
```
So this is summing over all the tokens in the sequence. And notice that we
are adding the `w_t_i` values to the `k_i` values. Lets just make explicit with
the above example:
```
          0           1           2               3
   x = ["Dan",      "loves"    , "ice"      , "cream"]
   k = [ [0.5, -03], [0.7, 0.2], [-0.1, 0.8], [ 0.3, -0.5] ]

   w_t_i + k_i

Learned w vector:
w = [0.2, 0.9]

Key vectors:
k_0 = [0.5, -0.3]  (for "Dan")
k_1 = [0.7, 0.2]   (for "loves")
k_2 = [-0.1, 0.8]  (for "ice")
k_3 = [0.3, -0.5]  (for "cream")

Calculations for w_t_i + k_i:

For t = 0 ("Dan"):
w_0_0 + k_0 = [-(0-0)*0.2 + 0.5, -(0-0)*0.9 - 0.3] = [0.5, -0.3]

For t = 1 ("loves"):
w_1_0 + k_0 = [-(1-0)*0.2 + 0.5, -(1-0)*0.9 - 0.3] = [0.3, -1.2]
w_1_1 + k_1 = [-(1-1)*0.2 + 0.7, -(1-1)*0.9 + 0.2] = [0.7, 0.2]

For t = 2 ("ice"):
w_2_0 + k_0 = [-(2-0)*0.2 + 0.5, -(2-0)*0.9 - 0.3] = [0.1, -2.1]
w_2_1 + k_1 = [-(2-1)*0.2 + 0.7, -(2-1)*0.9 + 0.2] = [0.5, -0.7]
w_2_2 + k_2 = [-(2-2)*0.2 - 0.1, -(2-2)*0.9 + 0.8] = [-0.1, 0.8]

For t = 3 ("cream"):
w_3_0 + k_0 = [-(3-0)*0.2 + 0.5, -(3-0)*0.9 - 0.3] = [-0.1, -3.0]
w_3_1 + k_1 = [-(3-1)*0.2 + 0.7, -(3-1)*0.9 + 0.2] = [0.3, -1.6]
w_3_2 + k_2 = [-(3-2)*0.2 - 0.1, -(3-2)*0.9 + 0.8] = [-0.3, -0.1]
w_3_3 + k_3 = [-(3-3)*0.2 + 0.3, -(3-3)*0.9 - 0.5] = [0.3, -0.5]
```
Notice that we have different decay values for each feature in the embedding.


### Sigmoid
The R vector is passed through the Sigmoid activation function which squashes
the values between 0 and 1. This is important as it controls how much of the
information is retained. So each value in this vector will be passed through
the Sigmoind function.

### LayerNorm x2 (Small Init Embeddings)
This struck me as somewhat odd that there would be two LayerNorm operations
after each other. But this seems like has to do with "Small Init Embeddings"
which is mentioned in section 3.4 of the paper.

A LayerNorm is defiend like this:
```
     
        x - μ
y = γ ( ------) + β
          σ

x = input
μ = mean
σ = standard deviation
β = bias (learned)
γ = scale (learned)
```
And having two will mean that both have different learnable parameters.
The embedding values will be normalized twice, with each normalization
potentially emphasizing different aspects of the input due to the separate
learnable parameters.
In llama.cpp there is a function named `llm_build_norm`:
```c++
static struct ggml_tensor * llm_build_norm(
        struct ggml_context * ctx,
         struct ggml_tensor * cur,
        const llama_hparams & hparams,
         struct ggml_tensor * mw,
         struct ggml_tensor * mb,
              llm_norm_type   type,
         const llm_build_cb & cb,
                        int   il) {
```
Where 'mw' is γ and 'mb' is β.


### Time decay vector
How this actually works is that we have a `vector` w which is learned, and it
tells how much the past matters for each dimension.

Each dimension in the w vector represents a feature, also called a channel in 
contexts like image processing. In image processing and image can have multiple
channels, like red, green, and blue for color images. These represent different
types of information (features) of the image. CNN documents/papers would
probably refer to channels as that is one of the main types that they were
designed for (at least initially), that is to process images. In the transformer
architecture this would be called features. 

And the value in each dimension determines how the influence of each feature
decays over time. The idea being that some features might loose their relavance
over time (faster decay), while others might be more important (slower decay).

For example:
```
       +----+
       |0.0 |
       +----+
       |0.9 |
       +----+
       |0.8 |
       +----+
       |-0.7|
       +----+
       |0.6 |
       +----+
```
So w would have to be the same length as the embeddings length in reality.

Then, wt,i is calculated using:
```
wt,i = -(t - i)w
```
The `(t - i)` part is calculating the relative position of the two tokens. It
tells use how far back the the sequence token i is from the current token t.

So, for our example above we would perhaps have something like the following,
and lets say we are currently looking at t=1, i=3 (column).
Recall that t is the current token in the input sequence, and i represents
another token in the input sequence (typically a previous token).
```
(t - 1) = (1 - 3) = -2
```
This tells us how far back in the sequence the token at position i (3) is from
the current token at position `t`. In this case it is 2 tokens ahead which is
why we have a negative number.

```
wt,i = -(t - i)w

wt,i = -(1 - 3)w
wt,i = -(-2)w
wt,i = 2w

       wt,i:
       +----+
       |2.0 |
       +----+
       |1.8 |
       +----+
       |1.6 |
       +----+
       |-1.4|
       +----+
       |1.2 |
       +----+
```
And this wt,i vector would be added to the Kᵢ vector which represents the
decay-adjusted weights for this specific interaction.

The negative sign in front of (t−i) is important because it ensures that the
influence of previous tokens decays as you move forward in the sequence.

The concept of decay here means that the further back a token is in the sequence
(relative to the current position), the less influence it should have. This is
a common assumption in many sequence models, reflecting the idea that recent
information is often more relevant than older information.
The following is an attempt to visualize this:
```
Input sequence represented as words instead of token embeddings:
"Dan" "loves" "ice" "cream"
  1     2       3       4

Table for calculating wt,i, where the t are the rows and i are the columns.

                          i
      | Dan (1) | loves (2) | ice (3) | cream (4) |
---------------------------------------------------
Dan   |  -(1-1) |  -(1-2)   | -(1-3)  |  -(1-4)   |
(1)   |    0    |   -1      |   -2    |    -3     |
---------------------------------------------------
loves |  -(2-1) |  -(2-2)   | -(2-3)  |  -(2-4)   |
(2)   |    -1   |    0      |    1    |    2      |  t
---------------------------------------------------
ice   |  -(3-1) |  -(3-2)   | -(3-3)  |  -(3-4)   |
(3)   |    -2   |    -1     |    0    |     1     |
---------------------------------------------------
cream |  -(4-1) |  -(4-2)   | -(4-3)  |  -(4-4)   |
(4)   |   -3    |    -2     |   -1    |    0      |
---------------------------------------------------
```
So we have a vector w which contains values that determine how important each
feature is over time. And this is used to modulate (changes/controls) the key
vector with the decay-adjusted weights, reflecting how the relationship between
tokens changes based on their relative position and feature decay rates.

In RWKV, we have the following important components:
* R (Receptance) - A vector which is the receiver and integrator of past information.
Similar to the hidden state in an RNN perhaps?
* W (Weight) - A vector containing positional decay information (trained)
* K (Key) - A matrix containing information about the current token.
* V (Value) - A matrix containing information about the current token.

Ealier we mentioned that in the RWKV model they refer to features as channels
which is good to keep in mind when we see component like channel-mixing, so we
can think of it as feature-mixing.

Lets take a look at the formula for the R vector:
```
r_t = W_r * (μ_r ⊙ x_t + (1 - μ_r) ⊙  x_t-1)

Where:
x_t   = current input token at time t
x_-1  = previous input token at time t-1
μ_r   = (mu) scalar variables? that determines how much of the previous token to
        mix in with the current token. This is a learned value.
⊙     = element-wise multiplication.
1-μ_r = scalar value which determines how much of the current token to mix in
        with the previous token.
W_r   = Weight matrix for the R vector.
```
Now, lets just keep in mind that xₜ and xₜ-1 are vectors of token embeddings and
they are the same length the embedding length. And recall that each dimension in
the W vector represents a feature (channel.

### Time-mixing
First lets take a look at what is called time-mixing which is about integrating
information from different time steps which enables the model to effectivly
remember and use past information when making predictions about the current or
future states.
μᵣ is a scalar value that is learned during training, but just a scalar value
and if it is closer too 1, say 0.7 that would mean that the current input has
more influence, and (1 - μᵣ) would then be 0.3 and specifies how much influence
the last token vector has. These are then used to scale the current input vector
and the last input token:
```
(μᵣ⊙ xᵣ + (1 - μᵣ) ⊙  xₜ-1)
```
After scaling the two vectors they are added together which blends the current
token vector with the previous token vector. This is then multiplied by the
learned weight matrix Wᵣ to produce the R vector.

One thing that confused my a little was the usage of element-wise multiplication
symbol ⊙, because if μᵣ is a scalar then that would be the same thing but I
think this is done for clarity and to make it more general.

In the time-mixing stage we also have:
```
k₁ = Wₖ* (μₖ⊙ xᵣ+ (1 - μₖ) ⊙  xₜ-1)
v₁ = Wᵥ* (μᵥ⊙ xᵣ+ (1 - μᵥ) ⊙  xₜ-1)
```
Notice that in the transformer architecture we also have K an V but those were
copies of the input token embeddings. In this case K and V are produced only
by the current input, and the previous input.
These are basically doing the same thing as the R vector, but for the K and V
vectors. So we are mixing in information from the previous token with the
current token, and then we are using the learned weight matrix to produce the
K and V vectors.

```
r₁ = Wᵣ* (μᵣ⊙ xᵣ+ (1 - μᵣ) ⊙  xₜ-1)
k₁ = Wₖ* (μₖ⊙ xᵣ+ (1 - μₖ) ⊙  xₜ-1)
v₁ = Wᵥ* (μᵥ⊙ xᵣ+ (1 - μᵥ) ⊙  xₜ-1)

      ᵢ₌₀
wkv = ----------------------------------------------
      ₜ₋₁
      Σ exp(-(t - 1 - i)w+kᵢ) + exp(u+k)
      ᵢ₌₀

oₜ = Wₒ* (σ(r₁) ⊙  wkvₜ)
```
This can be visualized as:
```
      +------------+
      |   Out      |
      +------------+
          ↑
          |
        +---+
    +---| ⊙ |---+
    |   +---+   |
  +---+    +--------------+
  | σ |    |      WKV     | 
  +---+    +--------------+
    ↑         ↑         ↑
    |         |         |
  +---+     +---+     +---+
  | R |     | K |     | V |
  +---+     +---+     +---+
    |          |          |
    +----------+----------+
               |
            +-----+
            |  μ  |
            +-----+
               ↑         
               |
               |
```


### WKV Operator
In the following note that `w` is a vector and contains values that determine
how important each feature is over time. And this is used to modulate the
key vector with the decay-adjusted weights.

So this is what the WKV operator is doing:

```
      ₜ₋₁
      Σ exp(-(t - 1 - i)w+kᵢ ⊙ vᵢ + exp(u+kₜ) ⊙ vₜ)   
      ᵢ₌₀
wkv = ----------------------------------------------
      ₜ₋₁
      Σ exp(-(t - 1 - i)w+kᵢ) + exp(u+k)
      ᵢ₌₀
```
Now, `t` is the sequence of token embeddings. So above we are summing over all
the tokens in the sequence. So if we had 10 tokens in the sequence we would
get -(10 - 1 - i). And we would get the following values for all tokens:
```
-(10 - 1 - 0) = -9
-(10 - 1 - 1) = -8
-(10 - 1 - 2) = -7
-(10 - 1 - 3) = -6
-(10 - 1 - 4) = -5
-(10 - 1 - 5) = -4
-(10 - 1 - 6) = -3
-(10 - 1 - 7) = -2
-(10 - 1 - 8) = -1
-(10 - 1 - 9) =  0
```
So lets take the first entry where i=0:
```
exp((-9)w + kᵢ ⊙ vᵢ + exp(u+kₜ) ⊙ vₜ)   
```
And `w` is a vector so this will scale each value in the vector by -9. And that
will then be added to the kₜvector.

### Channel-mixing
Now, this is about mixing information from different features (channels) within
a single token. So this is dealing with our features/channels (the different
dimensions in the token embedding vector). 

```
r'₁ = Wᵣ* (μ'ᵣ⊙ xᵣ+ (1 - μ'ᵣ) ⊙  xₜ-1)
k'₁ = Wₖ* (μ'ₖ⊙ xᵣ+ (1 - μ'ₖ) ⊙  xₜ-1)
```
Note that we don't have the V vector here, and I am not sure why that is?

The model performs computations using linear projections of inputs, essentially
transforming the inputs into a space where they can be more effectively analyzed
and combined. 

### llama.cpp RWKV implementation
The version implemented in llama.cpp is RWKV-6 (Finch).

```console
$ cd fundamentals/llama.cpp
$ make simple-prompt
$ run-rwkv-simple-prompt:
```
