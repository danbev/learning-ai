## Mobile Vision Language Model (MobileVLM)
The goal here is to have a vision language model for resource-constrained
device like mobile phones and IoT devices. The models are called  multimodal
vision language model (MMVLM).

For resource contstrained devices memory usage and computational cost are things
that have to be considered for LLM processing, but also energy consumption is
a consideration when using battery powered devices.

* MobileLVM paper: https://arxiv.org/pdf/2312.16886.pdf
* MobileLVM 2 paper: https://arxiv.org/pdf/2402.03766.pdf

The MobileLVM is composed of three main components:
1. A visual encoder  like a [ViT](vit.md)
2. An LLM tailored for resource constrained devices (MobileLLaMA)
3. An efficient projector (light weight downsampling projector (LDP))

### Achitecture
An input image Xᵥ is passed through the visual encoder F_enc to produce patch
embeddings f. These patch embeddings are then projected by the efficient
projector P to produce Hᵥ. The language model LLM is then used to process both
the patch embeddings and the text embeddings to produce the final output. Note
that this is part of the training process where the input comes in pairs of
images and text inputs.
```
f = F_enc(Xᵥ)
Hᵥ = P(f)

Xᵥ    = Input Image ε R^(HxWxC) (H=height, W=width, C=channels)
F_enc = Visual Encoder(Xᵥ) ε R^(Nᵥx Dᵥ) (Dᵥ=patch embedding dimension)
        Nᵥ = HW/P²          (P=patch size)
f     = patch embeddings produced by F_enc.
P     = Efficient Projector.
Hᵥ    = Patch embeddings projected by P.
```
Notice that the patch size is calculated dynamically based on the height and
width of the image and the patch size. 

```
H_q = ε R^(Nₜ x Dₜ) (Dₜ = size of the token embedding space)
```
So during training these two are combined to produce a matrix of sequence
tokens which are the input to the LLM. Notice that both are now in the format
of tokens as the patch embeddings have been projected into the token embedding
space and can be handled by the LLM. So this would describe how a model like
MobileVLM is trained. At inference time this trained model is used with only
a single input like an image. So it would take the image and produce the patch
embeddings and project them into the token embedding space just like in training
but there is not text input at this first stage. Instead the LLM will predict
the next token based on the "patch" embeddings. After it has predicted the next
token that predicted token will become part of the sequence of tokens and used
to predict the next token and so on. So this autoregressive, where auto means
"self" and "regressive" comes from the idea of regression which is above
predicting a value based on previous obervations, process leverages both the
visual context (initially) and the textual context (as it builds up) to generate
coherent and relevant text output.
```
                   L
p(Y_a | H_v,H_q) = ∏   p(yᵢ | H_v,H_q, y < i)
                   ᵢ=1
H_v   = Patch embeddings projected by P.
H_q   = Token embeddings.
y < 1 = the tokens already generated.
```
Notice that the last condition probability specifies that the probability of
yᵢ is conditioned on the image embeddings H_v, the token embeddings H_q, and
the previous tokens, the (y < i) part. This is the autoregressive part of the
model.

The probability of of response Y_a given the image embeddings H_v and the token
embeddings H_q.


### Effiecient Projector
The projector between the vision encoder and the language model is critical in
aligning multimodal features. There are currently two main approaches to this
which are [Q-Former](blip2.md) and MLP (Multi-Layer Perceptron).
This paper introduces a new approach which is suitable for resource-constrained
devices called the Light Weight Downsampling Projector (LDP).
"To keep spatial information and to minimize the computational cost, we make use
of a convolution with a stride of 2, which reduces 75% visual tokens. This
design significantly boosts the overall inference speed."
This sounds like they are reducing the number of patch embeddings (not the
dimensionality of the patch embeddings).
So way we have an image which we split into patches, then convert them into
patch embeddings, add positional encodings, and then remove some of them this
must impact the quality of the image representation. Just think about if we
removed to patches that have one patch inbetween them and there is a sign in
the image and the text of the sign spans all three patches, we would loose
information about would we not?
Yes, this is correct and is why the paper says that the LDP is designed to
minimize the loss of information. I thiink


The process described by ..
can be understood as taking the visual embeddings f produced by the encoder,
which are in the visual domain and potentially very high-dimensional, and
transforming them through the projector P into a new set of embeddings 
H that are suitable for integration with the text embeddings. This
transformation by P would adjust the embeddings to ensure they are in a
compatible format for the language model, which could involve changing their
dimensionality to match the expected input size of the language model.

## Light Weight Downsampling Projector (LDP)


