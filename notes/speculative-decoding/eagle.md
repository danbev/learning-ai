## Eagle speculative decoding
Framework for Lightweight Autoregressive Decoding.

### Eagle 1
Similar to Medusa, this technique does not require a separate draft model, but
instead adds a small head to the main model.

This is autoregessive, but the standard speculative decoding approach is also
autoregressive so how is this different?  
Standard speculative decoding is just like a smaller model, that predicts the
next token. It works in "token" space so it outputs token ids. Eagle 1 changes
this and instead predicts the next hidden state for the next token.

```console
Standard speculative decoding:
Input tokens ->  Draft model -> softmax -> Token x_{t+1}

Medusa:
Head 0  Takes h_t -> Predicts x_{t+1}
Head 1: Takes h_t -> Predicts x_{t+2}
Head 2: Takes h_t -> Predicts x_{t+3}

Eagle 1:
Input embedding ->  Eagle Head -> Predicted hidden state h_{t+1} ->
h_{t+1}         ->  Eagle Head -> Predicted hidden state h_{t+2} ->
h_{t+2}         ->  Eagle Head -> Predicted hidden state h_{t+3} ->
```
Comparing Eagle 1 with Medusa we can see that the heads in Eagle 1 are not
independent of each other, but instead they are chained together. So the second
head can use the information from the first head to make a better prediction for
the second token. This is guessing the next word's features, then using that
guess to imagine the features of the word after it etc.
