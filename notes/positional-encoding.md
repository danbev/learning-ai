## Positional Encoding
Positional encoding is used in transformers to give the model some notion of
position of the tokens in a word sequence. This is done by adding a positional
encoding vector to the input embedding vector. In an RNN, the position of the
token in the sequence is encoded in the hidden state of the RNN which is because
RNNs are sequential models. Transformers are not sequential models and so we
need to add some notion of position to the input embedding vector.

What we want to do is add some notion of position to the input embedding vector.
So how about just adding an integer that gets incremented for each token in the
sequence?
```
embedding = [0.1, 0.2, 0.3, 0.4]
positional_encoding = [0, 1, 2, 3]
embedding + positional_encoding = [0.1, 1.2, 2.3, 3.4]
```
An issue with this is that the sequences can be very large and that will effect
the gradients. But perhaps the integer can be normalized to a value between 0
and 1. This has an issue where it is not possible to know how many tokens are in
the sequence.

So instead of adding an integer to the embedding vector, we can
add a vector that is calculated using a formula. This vector is called the
positional encoding vector. The positional encoding vector is added to the input
embedding vector. 

The positional encoding matrix is calculated using the following formula:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where `pos` is the position of the token in the sequence.
The `i` variable confused me for a while as I read that it was the index of the
token sequence. But it is not and it is used as a column index for the matrix of
positional encoding values. For example, if we have a token embedding that is a
vector of four tokens, then the positional encoding will be a matrix of four
rows and four columns:
```console
d_model = 4

          i=0          i=0        i=1          i=1
pos  0   [ 0.          1.          0.          1.        ]
pos  1   [ 0.84147098  0.54030231  0.00999983  0.99995   ]
pos  2   [ 0.90929743 -0.41614684  0.01999867  0.99980001]
pos  4   [ 0.14112001 -0.9899925   0.0299955   0.99955003]


pos₀₀ = sin(0 / 10000^((2*0)/4)) = sin(0 / 10000^(0/4) = sin(0/10000^0)    = sin(0/1)  = sin(0)   = 0
pos₀₁ = cos(0 / 10000^((2*0)/4)) = cos(0 / 10000^(0/4) = cos(0/10000^0)    = cos(0/1)  = cos(0)   = 1
pos₀₂ = sin(0 / 10000^((2*1)/4)) = sin(0 / 10000^(1/4) = sin(0/10000^0.25) = sin(0/10) = sin(0)   = 0
pos₀₃ = cos(0 / 10000^((2*1)/4)) = cos(0 / 10000^(1/4) = cos(0/10000^0.25) = cos(0/10) = cos(0)   = 1

pos₁₀ = sin(1 / 10000^((2*0)/4)) = sin(1 / 10000^(0/4) = sin(1/10000^0)    = sin(1/1)  = sin(0)   = 0.84147098
pos₁₁ = cos(1 / 10000^((2*0)/4)) = cos(1 / 10000^(0/4) = cos(1/10000^0)    = cos(1/1)  = cos(0)   = 0.54030231
pos₁₂ = sin(1 / 10000^((2*1)/4)) = sin(1 / 10000^(1/4) = sin(1/10000^0.25) = sin(1/10) = sin(0.1) = 0.00999983
pos₁₃ = cos(1 / 10000^((2*1)/4)) = cos(1 / 10000^(1/4) = cos(1/10000^0.25) = cos(1/10) = cos(0.1) = 0.99995

pos₂₀ = sin(2 / 10000^((2*0)/4)) = sin(2 / 10000^(0/4) = sin(2/10000^0)    = sin(2/1)  = sin(0)   = 0.90929743
pos₂₁ = cos(2 / 10000^((2*0)/4)) = cos(2 / 10000^(0/4) = cos(2/10000^0)    = cos(2/1)  = cos(0)   = -0.41614684
pos₂₂ = sin(2 / 10000^((2*1)/4)) = sin(2 / 10000^(1/4) = sin(2/10000^0.25) = sin(2/10) = sin(0.2) = 0.01999867
pos₂₃ = cos(2 / 10000^((2*1)/4)) = cos(2 / 10000^(1/4) = cos(2/10000^0.25) = cos(2/10) = cos(0.2) = 0.99980001

pos₃₀ = sin(3 / 10000^((2*0)/4)) = sin(3 / 10000^(0/4) = sin(3/10000^0)    = sin(3/1)  = sin(0)   = 0.14112001
pos₃₁ = cos(3 / 10000^((2*0)/4)) = cos(3 / 10000^(0/4) = cos(3/10000^0)    = cos(3/1)  = cos(0)   = -0.9899925
pos₃₂ = sin(3 / 10000^((2*1)/4)) = sin(3 / 10000^(1/4) = sin(3/10000^0.25) = sin(3/10) = sin(0.3) = 0.0299955
pos₃₃ = cos(3 / 10000^((2*1)/4)) = cos(3 / 10000^(1/4) = cos(3/10000^0.25) = cos(3/10) = cos(0.3) = 0.99955003
```

