## Activation-aware Weight Quantization (AWQ) for Deep Neural Networks
AWQ is a post training quantization method but quantizes the weights in a way
that takes the data distribution into consideration. Like in quantization I've
used in llama.cpp (quantize example) it will perform quantization on all the
weights.

So weigths are the parameters of the nueral network and learned during training.
We have inputs that are fed into the neural network, and they pass through
various layers, being multiplied and transformed by the weights. The output of
each layer, after applying a non-linear function, is called the activation.
The activation are the response/result of the layer to the inputs it was given.

So traditionaly quantization is done on the weights without any other
considerations, it just quantized the weights of the model, like converting to
a smaller bit size.

What AWQ does it is takes the activation, that is the result of a layer after it
has been multiplied by the weights (and the non-linear function), and uses that
to quantize the weights. For example, lets say the an activation is consitently
low, then the weights that are used to produce that activation can be quantized
more aggressively (making them smaller bit size) with out affecting performance
(losing important information).

### Process
1. Collect statistics of activations by running the model on data, not data used
for training, to understand how activations behave.

2. Use the stats collected in step 1 to determine how to quantize the weights.
This might involve using smaller bit sizes for weights that are involved involved
in activations that are consistently low. And it might mean leaving weights that
are involved in activations that are consistently high at a higher bit size.

3. Quantize the weights using the information from step 2.

So with AWQ we should be able to see better performance, that is accurace of
the model, and at the same time also quantize which leads requiring less memory
at inference time.
