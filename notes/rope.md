## Rotary Positional Embeddings (RoPE)
This is a type of positional encoding which is used in PaML, GPT-Neo and GPT-J,
and LLAMA (1 & 2).

I've written about positional encoding in
[positional-encoding.md](positional-encoding.md) which discussed absolute
positional embeddings.

There is an issue with absolute positional embeddings which is that after the
model has been trained on a certain sequence length, it will not be able to
handle longer sequence lengths very well, if at all. The intuition here is that
if we imagine the input embeddings being vectors being moved around according
to the sinusoidal functions (sine and cosine) then the vectors (think of them
as 2d vectors) will move around without any pattern to them. The llm will not
be able to see a pattern in this but instead learn that the positions are the
way they are. If we then try to add more tokens to the input the llm will not
be able to handle this very well. It is kinda like memorizing the answers to
exam questions instead of actually learning the material. You might do alright
on the exam (the context length you trained on) but if you get a question that
is not exactly the same as the ones you memorized you will not be able to
answer it.

The goal here is the same, to introduce positional encoding but instead of
adding this to the embeddings it will add them to the query and key matrices by
rotating them. The idea is to make the dot product of the query
and key vectors position-aware, encoding the relative positions of tokens into
the attention mechanism.

When rotating the query and key vectors they are rotated in a certain way that
is not caotic like the absolute positioning. For each position they are
rotated a certain "fixed" amount of degrees (theta).

Rotation:
```
 [cos(θ) -sin(θ)]          θ = theta, the angle
 [sin(θ)  cos(θ)]
```
Rotating a vector does not change the length of the vector, it only changes the
direction of the vector.

![image](./rotation-org.png) ![image](./rotation-rotated.png)

Overlapping orginal and rotated vectors:
![image](./rotation-both.png)

Notice that the first vector in the origin is not visable in this last image but
you can see it in the first image. And notice that the lenghts of the vectors
are the same, only the angles are different.

In the attention mechanism of transformers, the similarity between tokens is
computed as the dot product of their query and key vectors. Normally, without
positional encoding, this similarity only reflects the content of the tokens.
With RoPE, the similarity becomes a function of both content and relative
position.

In RoPE each dimension is rotated by a different angle which is a function of
both the position in the sequence and the dimension. So the angle encodes the
position information. So the formula for the angle needs take the position
index into account.

So a rotation is applied to each dimension of the query and key vectors. These
are then used to calculate the attention scores. The attention scores now have
taken the positional information into account.

The rotation is done pairwise, for example (dᵢ, dᵢ₊₁) where dᵢ is the dimension
and dᵢ₊₁ is the next dimension. It is like we are doing two dimensional rotations
for each entry in the query/key matrices.
We apply the rotation like we saw above:
```
 [cos(θp, i) -sin(θp, i)]          θ = theta, the angle
 [sin(θp, i)  cos(θp, i)]
```
Where `θp, i` is the rotation angle for the i-th dimension pair and p is the
position in the sequence.

The angle is calculated as follows:
```
θp,i = p x w^i
```
Where `w` is a constant which determines how much the angle changes with each
dimension. `i` is the dimension pair (index?) and `p` is the position in the
sequence.


Let say we have the following sentence:
```
The cat sat on the mat.
```
And lets say we have two dimensions in our embedding space. We can then imaging
`Cat` is a vector. And lets say we have the word `cat` somewhere in the vector
space as well. Now, in our sentence the word `cat` is the second word so this
would be a separate vector only rotated by a certain amount. If the word comes
even later in the sentence the vector would be rotated even more.

The following image shows the relative positional embeddings for the sentence
above with the original word vectors and the rotated word vectors:

![image](./rope.png)

So the original points are the vectors for the words as if we were not using
any rotations at all. Then the rotated points are the vectors for the words
to show how they have been rotated for this specific sentence.

Now, even if we added words to the start of the sentence or to the end of the
sentence, when we look at 'cat' and 'sat' they will still have the same angle
theta between them. So the relative position of the words is still the same. So
this gives us both positional endcoding and relative positional encoding in a
single type of embedding technique instead two separate techniques.


```
                ^-2(i-1)
Θ = { θᵢ = 10000 ------- ,  i ∈ {1, 2, ..., d/2 }
                    d
```
So, upper-case theta is a set of angles where each angle is calculated as
10000 raised to the power of -2(i-1). Notice that this is a set of pairs, we
have d/2 and we rotate each pair. `i` is the index of the pair, ranging from
1 - d/2.
The angles are then used in generating rotations matrices for positional
encodings.

`d` is the dimension of the embedding space, which for llama would be 4096.
10000 is a constant base used in the computation of angles. In llama.cpp this
is a parameter named `freq_base` I think.
```
θ₀ = 10000^(-2(0-1))/4096
   = 10000^(2/4096)
   = 10000^(2/4096)
   = 10000^(0.00048828125)
   = 1.004507364
```
And then if we do a few more using
[rope-rotations.py](../fundamentals/python/src/rope-rotations.py)]:
```
1.0045073642544624
1.0
0.9955128609158502
0.9910458562488609
0.9865988956531019
0.9821718891880378
0.9777647473167089
0.9733773809039202
0.9690097012144389
0.9646616199111993
```

The rotation is angle-based and dimension-specific, meaning that pairs of
features (dimensions) within each token's embedding vector are rotated by
specific angles

Like if I have the sentence "Dan loves icecream", That might be tokenized in to
[2223, 25, 883, 10033] and some embeddings which might looks like this:
```
2223 : [1 2 3 4 5 6 7 8]
25   : [9 8 7 6 5 4 3 2]
883  : [1 2 3 4 5 6 7 8]
10033: [9 8 7 6 5 4 3 2]
```
The rotation will be applied for each pair for features in the embeddings and
the same rotation will be applied for the same positions of the embedddings:
```
        r1    r2    r3    r4
2223 : [1 2] [3 4] [5 6] [7 8]
25   : [9 8] [7 6] [5 4] [3 2]
883  : [1 2] [3 4] [5 6] [7 8]
10033: [9 8] [7 6] [5 4] [3 2]
        i=0   i=1   i=2   i=3
```
Now, we also want to take the position of the token embeddings in the sequence
into account and this is done by...

```
〔f_q(Xₘ, m), f_k(Xₙ, n)〕 = g(xₘ, xₙ, m-n)
```
〔〕is supposed to be angle brackets to indicate the dot product of two vectors
The vectors are the output of the functions `f_q` and `f_k`. And recall that
the dot product measures the similarity between the vectors. 
`f_q(Xₘ, m)` is the query vector for the m-th token in the sequence and
`f_k(Xₙ, n)` is the key vector for the n-th token in the sequence.
`g(xₘ, xₙ, m-n)` is a function that takes the embeddings of the query and key
embeddings, and as the third argument the relative position distance between the
two tokens. 

The expression `<f_q(Xₘ, m), f_k(Xₙ, n)> = g(xₘ, xₙ, m-n)` conveys that the
similarity between the query representation of a token at position m and the key
representation of a token at position n can be understood or represented as a
function of their respective embeddings and their relative positions (m-n).

```
f_q(Xₘ, m)
```
Just to clarify this `f_q` is a function that takes a "row" from the query
matrix. Each row in this matrix represents an token in the sequence. So Xₘ is
passing in one for these rows:
```
m₀    2223 : [1 2 3 4 5 6 7 8]
m₁    25   : [9 8 7 6 5 4 3 2]
m₂    883  : [1 2 3 4 5 6 7 8]
m₃    10033: [9 8 7 6 5 4 3 2]
```
And m is the position of that token in the sequence. So for a concrete example:
```
f_q([1 2 3 4 5 6 7 8], 2)
```

```
f_q(Xₘ, m) = (W_q xₘ)e^(imθ)

Where `W_q` is the query weight matrix, `xₘ` is the m-th row of the query matrix,
and `θ` is the rotation angle for the m-th position.
So, we take the W_q matrix and multiply it with the m-th row of the query matrix:
```
     W_q                  x₂
 [1 2 3 4 5 6 7 8]       [0]   [x₀]
 [1 2 3 4 5 6 7 8]       [1]   [x₁]
 [1 2 3 4 5 6 7 8]       [2]   [x₂]
 [1 2 3 4 5 6 7 8]       [3] = [x₃]
 [1 2 3 4 5 6 7 8]       [4]   [x₄]
 [1 2 3 4 5 6 7 8]       [5]   [x₅]
 [1 2 3 4 5 6 7 8]       [6]   [x₆]
 [1 2 3 4 5 6 7 8]       [7]   [x₇]
      8x8                8x1  
```
What happens then is not that we are raising the resulting vector elements to
e^imθ but instead we are applying a transformation which involved complex
numbers. Think of this as taking pairs of elements and rotating them, and how
much is determined by their position in the sequence and theta.

Recall that Euler's formula is:
```
e^iΘ = cos(Θ) + i sin(Θ)
```
But what does that mean? Well, it means that we can represent complex numbers
as a combination of a real part and an imaginary part. The real part is the
cosine part and the imaginary part is the sine part. So, if we have a complex
number `a + bi` we can represent it as `r(cos(θ) + i sin(θ))` where `r` is the
magnitude of the complex number and `θ` is the angle of the complex number.

And just to clarify this for myself, we we can have a varable like m in the
exponentiation:
```
e^imΘ = cos(m * Θ) + i sin(m * Θ)
```
So in this case m is scaling the angle theta before calculating the cosine and
sine of the angle.

We can rewrite the above formula as
```
Original:
f_q(Xₘ, m) = (W_q xₘ)e^(imθ)

Rewritten:
f_q(Xₘ, m) = (W_q xₘ) (cos(m * θ) + i sin(m * θ))

m is the position of the token in the sequence.
θ is the rotation angle for the m-th position.
```

Each element of the output vector is just a number, think of it as a number
on the number line (or x-axis). It is not a vector so we can't rotate it.
What we are going to do is take pairs of elements of the output vector and use
one as the real number and one as the imaginary part of a complex number, which
we can rotate.

Lets take the first pair:
```
                            y 

 [x₀]  => (x₀, x₁)          ^
 [x₁]                       |
                            |
                        x₁  |--------*
                            |        |
                            |        |
                            +---------->  x
                                     x₀ 
```
And this would be a vector from the origin to the point (x₀, x₁). And we can
also represent this as a complex number:
```
z = x₀ + i x₁
```
Now, we want to rotate the above vector by an angle θ. We can do this by:
```
z * e^iθ
```
Which can be rewritten as:
```
z * (cos(θ) + i sin(θ))
```
And if we expand z we get:
```
(x₀ + i x₁) * (cos(θ) + i sin(θ))
```
This will result in a new vector in the complex plane, which is a rotation of
the original vector. The real part of this will give use the new x₀ coordinate,
and the imaginary part will give us the new x₁ coordinate.

Multipliying these two complex numbers, the first is the vector in the complex
plane, and the second is the rotation operation, which gives us:
```
[complex vector] * [rotation] = [rotated complex vector]

(x₀ + ix₁) * (cos(θ) + i sin(θ))
```
We can expand that to the following terms when we distribute:
```
x₀ * cos(θ)
x₀ * i sin(θ)
ix₁ * cos(θ)
ix₁ * i sin(θ)
```
And we can apply the multiplication:
```
x₀ * cos(θ)    = x₀ cos(θ)                                Real part
x₀ * i sin(θ)  = ix₀ sin(θ)                               Imaginary part
ix₁ * cos(θ)   = ix₁ cos(θ)                               Imaginary part
ix₁ * i sin(θ) = i²x₁ sin(θ) = -x₁ sin(θ)       (i² = -1) Real part
                              (-1x₁ sin(θ))
```
We can combine the real and imaginary parts to get the new vector:
```
[ real part         ]   [ imaginary part       ]

x₀ cos(θ) - x₁ sin(θ) + i(x₀ sin(θ) + x₁ cos(θ))

[  new_x₀           ]    [  new_x₁              ]
```
The result of the rotation for this pair will be:
```
   [new_x₀]    [x₀ cos(θ) - x₁ sin(θ)]
   [new_x₁]    [x₀ sin(θ) + x₁ cos(θ)]
```
And this is done for all pairs in the output vector.

Notice the we can represent the rotation as a matrix by taking out the x₀ and
x₁:
```
 [x₀ cos(θ) - x₁ sin(θ)]
 [x₀ sin(θ) + x₁ cos(θ)]

 [cos(θ) -sin(θ)]  [x₀]
 [sin(θ)  cos(θ)]  [x₁]
```
And theta is taken from the set of angles we calculated earlier (I think):
```
                ^-2(i-1)
Θ = { θᵢ = 10000 ------- ,  i ∈ {1, 2, ..., d/2 }
                    d
```
Now, I think that 10000 is the `base_freq` parameter in llama.cpp and perhaps
that -2 is the `freq_scale`.

#### beta_fast and beta_slow (blending)
Imagine a model trained up to a context length of 512 tokens, and you wish to
extend its capabilities to handle up to 1024 tokens. A blending range might be
set from 400 to 600 tokens. In this range:

Positions closer to 400 would use predominantly interpolated embeddings, as
they're closer to the trained range. As positions move towards 600, there's an
increasing reliance on extrapolated embeddings.
Beyond 600 tokens, the model uses purely extrapolated embeddings for positional
information.
The parameters beta_fast and beta_slow control the blending of interpolated and
extrapolated embeddings.
