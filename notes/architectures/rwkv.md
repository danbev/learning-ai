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

Lets take a look at inference in this architectur:
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
                      ↓                             |
           +----------------------------------+     |
           |               μ                  |     |
           | R = μ_r ⊙ x_t + (1 - μ_r) ⊙ x_t-1|     |
           | K = μ_k ⊙ x_t + (1 - μ_k) ⊙ x_t-1|     |
           | V = μ_v ⊙ x_t + (1 - μ_v) ⊙ x_t-1|     |
           +----------------------------------+     |
               |            |            |          |
            +-----+      +-----+      +-----+       |       
            |  R  |      |  K  |      |  V  |       |
            +-----+      +-----+      +-----+       |
               |            |            |          |  
            +--------+   +------------------+       |
            |Sigmoid |   |   WKV Operator   |       |
            +--------+   +------------------+       |
               |                  |                 |

``` 


Interpolation 
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

```
     Time decay matrix

         T
  +---+---+----+----+
  |   |   |    |    | 3
  +---+---+----+----+
  |   |   |    | 8  | 2
  +---+---+----+----+
  |   |   |    |    | 1
  +---+---+----+----+
  |   |   |    |    | 0
  +---+---+----+----+
   0    1   2    3
```
So we might read this as token 2 will interact with token 3 with a weight of 8.
Since we are adding this with Kᵢwe are also mixing in some information about the
current token.

How this actually works is that we have a `vector` w which is learned, and it
tells how much the past matters for each dimension.

Each dimension in the W vector represents a feature, also called a channel in 
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
symbol ⊙, because if μᵣ is a scalare then that would be the same thing but I
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

__wip__

### Channel-mixing
Now, this is about mixing information from different features (channels) within
a a single token. So this is dealing with our features/channels (the different
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

```console
$ cd fundamentals/llama.cpp
$ make simple-prompt
$ run-rwkv-simple-prompt:
```

Lets take an input sequence of the string "Dan loves ice cream" and let the
current token be `x_t`. We calculate the K vector for the current token, and we
do the same for the value vector.

So for each token in the sequence we would:
```
K_t = W_K * x_t
V_t = W_V * x_t

W_K = Weight matrix for the K vector
W_V = Weight matrix for the V vector
t   = current token
```
The Receptance (R) vector is calculated using the following formula:
```
R_t = W_r * σ(W_R * x_t + B_R)

σ   = Sigmoid activation function
W_r = Weight matrix for the R vector
x_t = current token
B_R = Bias vector for the R vector
```
The sigmoid will ensure that the values in the R vector are between 0 and 1.
This is important as it controls how much of the previous information is
retained. 
The is also something called a time decay factor which is called `w` and is a
learned vector which controls how quickly the influence of previous tokens
decays.

The WKV operations (Weighted Key Value) looks something like this:
```
output = WKV(K_t, V_t, R_t, state_t-1)
```
