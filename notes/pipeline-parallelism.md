## Pipeline Parallelism (PP)
For prompt processing we use Pipeline Parallelism. Lets say we have two GPUs as
our model cannot fit on a single GPU.

When we offload to multiple GPUs this is by default done by layers/blocks.
So lets say we have a model with 42 layers and two GPUs, the first 21 layers
```
GPU_A: will have the bottom 21 layers
GPU_B: will have the upper  21 layers
```
When the model is loaded the weight tensors are read from the .gguf model file
and are then layers for each GPU are copied to the GPUs global memory. So GPU_A
will only have 21 layers, and likewise GPU_B will only have 21 layers. These will
never move. The only thing that moves between the GPUs are the activation tensors.
Flow of processing:
```
1. Input prompt enters GPU_A
2. GPU_A processes the input through its 21 layers
3. The activation tensor output from GPU_A is sent to GPU_B over the PCIe bus
4. GPU_B processes the activation tensor through its 21 layers
5. GPU_B produces the final output (the next logit scores)
```
The reason this is called pipeline is because of how the data flows through it
like a pipeline. This is controlled by the `--n-gpu-layers` option which

So to recap, when we have PP we split the layers of the model, often for larger
models that can't fit on the hardware. The models layers are split, the complete
layers amoung the available GPUs. And there process is like a pipeline where one
GPU has the first layers, processes the input and send the output of the layers
that it is responsible for to the next GPU which continues the processing.

The same thing happens for token generation. We have the input token which does
through the pipeline.
