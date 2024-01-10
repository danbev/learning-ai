## Long Short-Term Memory (LSTM)
This builds upon [rnn](./rnn.md) and was developed to address the issue of
vanishing and exploding gradients which makes plain RNNs hard to train.

We have the RNN example from [rnn](./rnn.md) and we saw, we can call the
connection between the layers the "short-term memory":
```
inputs [0, 0]

   Input₀         W₁        b₁        y₁          w₃         b₂      output
  +------+      +----+     +---+     +----+      +---+      +--+    +-----+
  |  0   | ---> |1.2 | --> |+0 | --> |ReLU| ---> |1.5| ---> |0 | -->|     |
  +------+      +----+ (+) +---+     +----+      +---+ (+)  +--+    +-----+
                            W₂          ↓                          todays predicted value
                           +----+       | y₁
                        +--|-0.5|-------+
                        |  +----+
                        |
   Input₁         W₁    |   b₁        y₁          w₃         b₂      output
  +------+      +----+  |  +---+     +----+      +---+      +--+    +-----+
  |  0   | ---> |1.2 | --> |+0 | --> |ReLU| ---> |1.5| ---> |0 | -->|     |
  +------+      +----+ (+) +---+     +----+      +---+ (+)  +--+    +-----+
                                                                    tomorrows predicted value
```
LSMTs are a special kind of RNNs that have a "long-term memory" in addition to
the "short-term memory". 

LSTM use the Sigmoid activation function and the tanh activation function, and
recall that sigmoid returns a value between 0 and 1, and tanh returns a value
between -1 and 1.

The Long-Term memory can be modified using multiplication, and addition but
there are not weights or biases involved. This is called a "cell state":
```
Long-Term Memory (Cell State)

      L₀                    L₁
    +----+        +----+   +---+       +----+
----|    |--------|mul |--→|   |-------|sum |-----------------------------------→
    +----+        +----+   +---+       +----+
                    ↑
                +-------+
                |Sigmoid|
                +-------+
                  ↑
                +----+
                |    | b₁
                +----+
      W₂          ↑
    +-----+    +----+
    |     |---→|sum |
    +-----+    +----+
      |           ↑
    +----+        |
----|    |---------------------------------------------------------------------
↑   +----+        |
|     S₀          |
|               +---+
|               |   | w₁
|               +---+
|                 ↑
|                 |
|               +---+
|               |   |
|               +---+
|               input
|
Short-Term Memory (Hidden State)
```
Lets take a look at the first steps of the LSTM:
```
y₁ = (S₀ * W₂) + (input * W₁) + b₁
y₁ = sigmoid(y₁)

L₁ = L₀ * y₁
```
The sigmoid output determines the percentage of the long-term memory that will
be kept/rembembered. The long-term memory is then multiplied by the this
percentage producing L₁. If this value is low it causes the LSTM to discard or
forget this information which is why it is often called the "forget gate" and
allows the LSTM to focus on only the relevant information in a sequence.

That was the first stage of the LSTM. Now lets look at the second stage:
```
Long-Term Memory (Cell State)

      L₀                    L₁
    +----+        +----+   +---+         +----+
----|    |--------|mul |--→|   |---------|sum |-----------------------------------→
    +----+        +----+   +---+         +----+
                    ↑
                +-------+      +-------+  +---+   +-----+
                |Sigmoid|      |Sigmoid|->|mul|<--|tanh |
                +-------+      +-------+  +---+   +-----+
                  ↑               ↑                 ↑
                +----+          +----+            +----+
                |    | b₁       |    |b₂          |    | b₃
                +----+          +----+            +----+
      W₂          ↑     W₄        ↑        W₆       ↑
    +-----+    +----+  +---+    +----+   +---+    +----+
    |     |---→|sum |  |   |--->|sum |   |   |---→|sum |
    +-----+    +----+  +---+    +----+   +---+    +----+
      |           ↑      |        ↑        ↑         ↑
    +----+        |      |        |        |         |
----|    |---------------+-----------------+-----------------------------------
↑   +----+        |               |                  |
|     S₀          |               |                  |
|               +---+           +---+              +---+
|               |   | w₁        |   | W₃           |   | W₅
|               +---+           ---+               +---+
|                 ↑               ↑                  ↑
|                 |---------------+------------------+------------------------->
|                 |
|               +---+
|               |   |
|               +---+
|               input
|
Short-Term Memory (Hidden State)
```
Lets start from the far right.
```
y₂ = tanh( (S₀ * W₆) + (input * W₅) + b₃ )
```
This value gives a new candidate value for the long-term memory. This is called
the "candidate value" because it is not yet known if it will be used to update
the long-term memory. This is where the middle block comes into play, it will
determine how much (percentage) of the candidate value will be used to update
the long-term memory. Notice that this middle block is very similar to the
forget gate, except that it uses a different set of weights and biases, and that
was also used to determine how much of the long-term memory to keep/forget.
The calculation of the middle block is:
```
y₃ = sigmoid( (S₀ * W₄) + (input * W₃) + b₂ )
L₂ += (y₃ * y₂)
```
This stage is called the "input gate".

The final stage is to update the hidden state (short-term memory).
```
Long-Term Memory (Cell State)

      L₀                    L₁                     L₂ 
    +----+        +----+   +---+         +----+   +---+
----|    |--------|mul |--→|   |---------|sum |-->|   |------------------------+------→
    +----+        +----+   +---+         +----+   +---+                        |
                    ↑                                                          ↓
                +-------+      +-------+  +---+   +-----+        +-------+    +----+
                |sigmoid|      |sigmoid|->|mul|<--|tanh |        |sigmoid|-+  |tanh|
                +-------+      +-------+  +---+   +-----+        +-------+ |  +----+
                  ↑               ↑                 ↑                ↑     |    |
                +----+          +----+            +----+           +---+   |  +---+
                |    | b₁       |    |b₂          |    | b₃        |   |b₄ +-→|mul|S₁
                +----+          +----+            +----+           +---+      +---+
      W₂          ↑     W₄        ↑        W₆       ↑      W₈        ↑          |
    +-----+    +----+  +---+    +----+   +---+    +----+  +---+    +---+      +---+
    |     |---→|sum |  |   |--->|sum |   |   |---→|sum |  |   |--->|sum|      |   | S₂
    +-----+    +----+  +---+    +----+   +---+    +----+  +---+    +---+      +---+
      |           ↑      |        ↑        ↑         ↑      ↑        ↑          |
    +----+        |      |        |        |         |      |        |          |
----|    |---------------+-----------------+----------------+-------------------+----→
↑   +----+        |               |                  |               |
|     S₀          |               |                  |               |
|               +---+           +---+              +---+           +---+
|               |   | w₁        |   | W₃           |   | W₅        |   | w₇
|               +---+           ---+               +---+           +---+
|                 ↑               ↑                  ↑               ↑
|                 |---------------+------------------+---------------+
|                 |
|               +---+
|               |   |
|               +---+
|               input
|
Short-Term Memory (Hidden State)
```
The calculation of the short-term memory is:
```
y₄ = sigmoid( (S₀ * W₈) + (input * W₇) + b₄ )
S₁ = y₄ * tanh(L₂)
```
This new value for the short-term memory is the output of this LSTM unit and
is therefore called the "output gate".

So what we have seen so far is that a single LSTM unit has three gates/stages.
Initially the long-term memory and short-term memory are set to zero.
When we have an input sequence we pass in the first value of the sequence and
it goes through a LSTM unit like we have seen above. The output of this LSTM
unit is the short-term memory. The short-term memory and the long-term memory
are updated as part of this processing. 
After that we pass in the second value of the sequence and it goes through the
same process. When all the inputs have been processed the output value of the
entire LSTM is the short-term memory of the last LSTM unit.
