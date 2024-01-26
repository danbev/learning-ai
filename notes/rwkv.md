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
Since we are adding this with Kᵢ we are also mixing in some information about
the current token.

How this actually works is that we have a vector w which is learned, and it
tells how much the past matters for each dimension.

Each dimension in the W vector represents a feature, also called a channel in 
context like image processing. And the value in each dimension determines how
the influence of each feature decays over time. The idea being that some
features might loose their relavance over time (faster decay), while others
might be more important (slower decay).

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
