## Generative Deep Learning
The book I'm reading is Generative Deep Learning 2nd Edition by David Foster.

### Introduction
The kind of deep learning that I'm familar with is called descrimative which
is when we are trying to predict/classify something. Like we might have a set of
images that we want to determine what they are. We use a training set which
contains examples and labels of what the images represent. The training of the
model, the weights of the model, will allow unseen images to be classified.
This is deterministic as if we have classified an image once, running it
though the system again will return the same classification.
```
 p(y|x)
```
The goal is to model the probability of a label y, given some x (called an
observation).

In generative deep learning the output is probabilistic and the output is
sampled many time, we pass the input through the system multiple times that is,
and want to have different "suggestions". And here the goal is to generate a
new image (or something like text, video, music etc) and not classify an
existing image.

```
 p(x)
```
This is saying that the return value of this function is the probablity of
observing an observation `x`.

This was really interesting from the first chapter of the book
```
Current neuroscientific theory suggests that our perception of reality is not a
highly complex discriminative model operating on our sensory input to produce
predictions of what we are experiencing, but is instead a generative model that
is trained from birth to produce simulations of our surroundings that accurately
match the future. Some theories even suggest that the output from this
generative model is what we directly perceive as reality
```

