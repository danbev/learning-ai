### Recurrent Neural Networks (RNN)
While RNNs might be a little outdated and have been more or less replaced with
Long Short Term Memory (LSTM)s, and Transformers, I've found the need to
understand them in order to fully understand newer acrhitectures like Mamba.

Recurrent Neural Networks are similar to multi-layer perceptrons, except that
they have a "memory" which allows them to remember previous inputs. This is
useful for tasks like predicting the next word in a sentence, or the next
character in a sequence.

The input to the RNN is a sequence of values, so below input would be a sequence
of number:
```
   input          W₁        b₁        y₁          w₃         b₂      output
  +------+      +----+     +---+     +----+      +---+      +--+    +-----+
  |      | ---> |    | --> |   | --> |ReLU| ---> |   | ---> |  | -->|     |
  +------+      +----+ (+) +---+     +----+      +---+ (+)  +--+    +-----+
                        ↑               ↓ 
                        |  +---+        | 
                        +--|   |--------+
                           +---+
                            w₂
```
Notice that the output from y1 can go both forward towards the output, but it
can also loop back to the addition of the bias of the previous step.
Lets look at a concrete example to visialize this better:
```
inputs [yesterdays value, todays value, tomorrows value ?]
inputs [0, 0]
        |
+-------+
|  Input          W₁        b₁        y₁          w₂         b₂      output
| +------+      +----+     +---+     +----+      +---+      +--+    +-----+
+→|  0   | ---> |1.2 | --> |+0 | --> |ReLU| ---> |1.5| ---> |0 | -->|     |
  +------+      +----+ (+) +---+     +----+      +---+ (+)  +--+    +-----+
                        ↑               ↓ 
                        |  +---+        | 
                        +--|   |--------+
                           +---+

y₁ = relu((0*1.2) + 0)
y₁ = relu(0)
y₁ = 0
output = (0*1.5) + 0 = 0
output = is the predicted value of "today"
```
In our case we are not interested in the predicted output of "today", which is
the next sequence of the input, since we already know that. What we are
interested in is the predicted output of "tomorrow".

In this case we will let y₁ loop back and go through the multiplication of w₂
and the addition of b₂. This will give us the predicted value of "tomorrow":
```
inputs [0, 0]
           |
+----------+
|  Input          W₁        b₁        y₁          w₂         b₂      output
| +------+      +----+     +---+     +----+      +---+      +--+    +-----+
+→|  0   | ---> |1.2 | --> |+0 | --> |ReLU| ---> |1.5| ---> |0 | -->|     |
  +------+      +----+ (+) +---+     +----+      +---+ (+)  +--+    +-----+
                        ↑               ↓ 
                        |  +----+       | y₁
                        +--|-0.5|-------+
                           +----+

  (today * w₁) + (y₁ * w₂)
                 [yesterdays value]

(0 * 1.2) + (0 * 1.5) + 0
y₁' = relu((0*-0.5) + 0)
y₁' = relu(0)
y₁' = 0
```
Now, that is not really easy to follow as we have different input values and
need to keep track of the loops. 
```
inputs [0, 0]

   Input₀         W₁        b₁        y₁          w₂         b₂      output
  +------+      +----+     +---+     +----+      +---+      +--+    +-----+
  |  0   | ---> |1.2 | --> |+0 | --> |ReLU| ---> |1.5| ---> |0 | -->|     |
  +------+      +----+ (+) +---+     +----+      +---+ (+)  +--+    +-----+
                                        ↓                          todays predicted value
                           +----+       | y₁
                        +--|-0.5|-------+
                        |  +----+
                        |
   Input₁         W₁    |   b₁        y₁          w₂         b₂      output
  +------+      +----+  |  +---+     +----+      +---+      +--+    +-----+
  |  0   | ---> |1.2 | --> |+0 | --> |ReLU| ---> |1.5| ---> |0 | -->|     |
  +------+      +----+ (+) +---+     +----+      +---+ (+)  +--+    +-----+
                                                                    tomorrows predicted value
```
In this case we now have two input to he RNN and not just one as in the first
example. It also has two outputs. Notice to the value of yesterday and today
are used to predict the value of tomorrow.
If we have more input values then we just "unroll" the recurrent neural network
and add more inputs and outputs.
Note that the weights and biases are the same for all inputs so increasing the
the number of inputs does not increase the number of weights and biases that
have to be trained.

One issue with RNNs is the vanishing gradient problem, and the exploding
gradient problem. This is because the gradient is calculated by multiplying the
gradient of the previous step with the weights of the previous step. This means
that the gradient can either explode or vanish depending on the weights. Too 
large weights and the gradients will be large and will "jump" around, too small
weights and the gradient steps will be too small and the maximum number of
iterations migth be reached before the network has converged.
A solution for this is to use LSTM networks instead of RNNs.
