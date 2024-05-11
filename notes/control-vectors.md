## Representation Engineering (control vectors/steering vectors/concept vectors)
This is something that is added at inference time to control the models
behaviour. This is done without any changes to the prompt or any fine tuning of
the model.

So this is a vector which can be applied (added?) to a models layer. This is
probably a tenor of the shape:
```
  control vector for layer1 [....]
  control vector for layer2 [....]
  ...
  control vector for layern [....]
``` 
And these would be applied to the hidden states of the model at each layer.

Paper: https://arxiv.org/abs/2310.01405
Llama.cpp PR: https://github.com/ggerganov/llama.cpp/pull/5970
