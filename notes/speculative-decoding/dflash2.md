### DFlash2
Is an extension to dflash, similar to how dspark is an extension, and addresses
the multi-modal collisions and suffix acceptance decay in block diffusion.

There is an issue with dflash which is inherent to the parallel processing which
is when we do parallel decoding we are predicting the complete "sentence" in one
go. There is no way for any token to know about the token prior to it or after
it as they get generated at the same time. And just like dspark is an extension
that addresses this issue, dflash2 also addresses this issue but in a different
way.

Standard non-causal self-attention lacks a strong local directional constraint,
causing the draft quality to degrade toward the end of the block (suffix decay).

So dspark solved this with a small network layer where tokens are are
sequentially passed through, enabling them to take the token before them into
account. This does have a cost and this is what dflash2 tries to avoid. Instead
of just drafting a single token for each posistion it will draft k tokens.
This is like a matrix where there is a row of k tokens for each position. And
then we can think of this as each one of these having connections to each other,
and there are probabilities for each paths (this would normally be static from
perhaps a vocabulary count or something but it is dynamic in this case). And these
are used to "walk" throw this matrix (is makes what makes it a lattice) and we
will get a score for each path. The path with the highest score contains the
tokens that are the most probably.

So instead of just drafting a single token for each position, DFlash2 drafts k
top candidate tokens for every slot in parallel, forming an n x k grid
(a "lattice" or trellis).
So instead of:
```console
pos0 : token
pos1 : token
pos2 : token

```

So in a single draft the model will output candidates for every position at the
same time, perhaps something like this:
```console
                  sequence pos 0        sequence pos 1        sequence pos 2
                     (i = 0)               (i = 1)               (i = 2)
                 +-------------+       +-------------+       +-------------+
candidate k=0    | [0] "new"   |       | [0] "red"   |       | [0] "car"   |
                 +-------------+       +-------------+       +-------------+

                 +-------------+       +-------------+       +-------------+
candidate k=1    | [1] "used"  |       | [1] "sports"|       | [1] "truck" |
                 +-------------+       +-------------+       +-------------+

                 +-------------+       +-------------+       +-------------+
candidate k=2    | [2] "fresh" |       | [2] "fast"  |       | [2] "bike"  |
                 +-------------+       +-------------+       +-------------+
```
Notice that we have the token positions as the columns and the rows are the
top k candidates for each position. For example, pos 0 has the top 3 candidates:
"new", "used", "fresh".

```console
Position 0 Candidates                         Position 1 Candidates

┌─────────────────┐                           ┌─────────────────┐
│   [0] "new"     │─────────────┬────────────>│   [0] "red"     │
└─────────────────┘             │             └─────────────────┘
                                │
┌─────────────────┐             ├────────────>┌─────────────────┐
│   [1] "used"    │─────────────┼────────────>│   [1] "sports"  │
└─────────────────┘             │             └─────────────────┘
                                │
┌─────────────────┐             └────────────>┌─────────────────┐
│   [2] "fresh"   │──────────────────────────>│   [2] "fast"    │
└─────────────────┘                           └─────────────────┘
```
So we could pick any of pos 0 candidates, and each could pick any of the candidates
in the next position, giving as a 3 * 3 = 9 unique connections.

Now, multi-modal collisions happen because position i doesn't know which tokenx
position i-1 chose. This creates "ambiguity", there are multiple possible token
combinations, but only some are grammatically coherent.
Unlike DSpark, which runs a small neural network sequentially token-by-token,
DFlash2 avoids running any sequential neural network layers on the GPU.

To resolve token ambiguity without running a sequential model pass:
1. A lightweight candidate selector evaluates a k x k transition matrix between adjacent
   positions (pos_i-1 -> pos_i).

```console
              pos1[0]"red"  pos1[1]"sports" pos1[2]"fast"
pos0[0]"new"   [  1.2            3.5              0.9  ]
pos0[1]"used"  [  0.8            2.0              0.4  ]
pos0[2]"fresh" [  0.5            0.1              1.1  ]
```

2. The transition scores are generated dynamically on-the-fly using the target model's
   projected hidden state h, steering the choices toward coherent phrasing.

3. A C++ runtime (dflash.cpp) walks the lattice using dynamic programming
   (Viterbi search) to find the path that maximizes total probability.

Result: Multi-modal collisions are eliminated, order is preserved, and suffix
acceptance rates remain high—all while keeping the draft phase 100% parallel.

### Two-Tap Dynamic Convolution
Tap refers to a filter coefficient, the kernel size. Dynamic means that the weights
are generated dynamically on the fly based on the current hidden state (not static).
So Two-tap means k=2, that we have a convolution size of 2. So at any position
i in the draft the layer operates on the hidden representation of the current
token i and i-1.

### Suffix decay
This refers to a sharp drop in token acceptance rates as we move from the beginning
of the draft block, the prefix, to the end of the block, the suffix).
```
[anchor token,   d_0,  d_1, ..., d_n  ]
  pos0           pos1  pos2 ..., pos_n
```
The first position that comes after the anchor token, which is the token that
the target model actually predicted is usually very accurate, often an 85-90%
acceptance rate. Next, we have pos_2 which is trying to predict a token
conditioned on pos1 which has not been verified by the target model yet.
pos_n tries to predict a token based on n-1 unverified, hypothetical tokens. Because
every predicted slot carries a small margin of error, that uncertainty compounds
exponentially as you go deeper into the block. By Position 6, the draft model is
effectively trying to predict the future based on a stack of guesses.

Standard autoregressive LLMs enforce strict causal attention: token i can only
look backward at token i-1, enforcing a strict left-to-right cause-and-effect chain.

In parallel block diffusion, the draft model uses non-causal (bidirectional)
attention so that all n tokens in the block can look at each other simultaneously
in a single forward pass.

In non-causal attention, Slot 4 attends to Slot 1, 2, 3, 5, and 6 all at once.
It treats the whole block as a pool of information rather than a strict left to
right timeline. Non-causal attention lacks a built-in mechanism to force Slot 4
to obey the exact output of Slot 3. Instead of predicting "What word specifically
follows Slot 3?", Slot 4 predicts "What word fits generally into this entire
6-token region?"

The draft model outputs tokens that look locally plausible in isolation, but
fail to form a strict, causal chain.
