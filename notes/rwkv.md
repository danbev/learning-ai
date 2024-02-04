## Receptance Weight Key Value (RWKV): Reinventing RNNs for the Transformer era
Simliar to [Mamba](./mamba.md) this model architecture avoids the transformers
O(N²) complexity by using an RNN (or an SSM).

So with this, like Mamba, we have efficient training and also efficient
inference (which scales linearly with the length of the sequence and not
exponentially like it does with transformers). And there are no approximation
techniques like in the Performer/Reformer/Linear Transformer (perhaps others)
either instead this is exact.

It is possible to [decompose attention](./transformer.md#decomposing-as-vector-operations)
which is a way to work around the memory limits of having the Q, K, and V
matrices stored in memory, though it might mean more computation time as we
have to compute then sequentially and not in parallel.

Now in RWKV instead of using the Q, K, and V matrices the formula looks like
this:
```
                 Σ exp(Wₜᵢ+ kᵢ) . vᵢ
Att+(W, K, V)ₜ = -----------------
                 Σ exp(Wₜᵢ+ kᵢ)
```
Notice that we are still taking a weighted sum of the values, but we are using
weights that are learned during training, and not the query values. The keys
are still the same as in the original attention mechanism and contain
information about the current token, but the query is gone. And notice also that
the operation is addition and not multiplication before the exponentiation.

So W is a learned matrix (actually not entirely true but we will see how this
works shortly) and is the same for all calculations:
```
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

Lets take a look at the forumal for the R vector:
```
r₁ = Wᵣ* (μᵣ⊙ xᵣ+ (1 - μᵣ) ⊙  xₜ-1)

Where:
xₜ   = current input token at time t
xₜ-1 = previous input token at time t-1
μᵣ   = (mu) scalar variables? that determines how much of the previous token to mix in
        with the current token. This is a learned value.
⊙    = element-wise multiplication
1-μᵣ = scalar value which determines how much of the current token to mix in with
       the previous token.
Wᵣ   = Weight matrix for the R vector
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


### Channel-mixing
Now, this is about mixing information from different features (channels) within
a a single token. So this is dealing with our features/channels (the different
dimensions in the token embedding vector). 
__wip__



The model performs computations using linear projections of inputs, essentially
transforming the inputs into a space where they can be more effectively analyzed
and combined. 
