## Generative Deep Learning


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
The goal is to model the probablity of a label y given some x (called an
observation).

In generative deep learning the output is probabilistic and the output is
sampled many time, we pass the input through the system multiple times that is
and want to have different "suggestions". And here the goal is to generate a
new image and not classify an existing image.

```
 p(x)
```

