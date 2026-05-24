###  Hailo
This is a NPU (Neural Processing Unit) that can be used with the Raspberry Pi 5.
It is designed for AI applications and can accelerate machine learning tasks.

I have a Pi 5 with a Pi AI HAT+2 (Hailo-10H AI Accelerator chip).

Pre-assemble:
![image](../../notes/images/pi-pre.jpeg)

Assembled:
![image](../../notes/images/pi.jpeg)

(HAT = Hardware Attached on Top, a standard for add-on boards for Raspberry Pi)

Exploration code can be found in [npu/hailo](../../npu/hailo).

### Architecture
This chip is different from a GPU which is a more general computation device. It
is designed specifically for AI computations. The chip itself consists of
computation unit with memory locations close by to them. A language model is
first compiled into a hailo execution file (hef) which the used to configure a
device. The flow is that of a dataflow architecture where the chip maps the
model layers of a neural network physically onto an internal fabric of
interconnected compute blocks, conton units, and localized SRAM memory. There is
no instruction fetch or global memory which we might be used to in a CPU or a
GPU.

A HEF can contain one or more neural networks
```console
$ hailortcli parse-hef hefs/qwen2.5.hef
HEF Compatible for: HAILO15H, HAILO10H

Network group name: base_model__prefill, Multi Context - Number of contexts: 124
    Network name: base_model__prefill/qwen2_prefill96
        VStream infos:
            Input  qwen2_prefill96/input_layer3 UINT8, NHWC(1x96x1536)
            Input  qwen2_prefill96/input_layer1 UINT16, FCR(1x96x1536)
            Input  qwen2_prefill96/input_layer2 UINT8, FCR(1x96x24576)
            Input  qwen2_prefill96/input_layer6 UINT8, FCR(1x96x256)
            Input  qwen2_prefill96/input_layer5 UINT8, FCR(1x96x256)
            Input  qwen2_prefill96/input_layer4 UINT8, FCR(1x96x1536)
            Output qwen2_prefill96/qwen2_block29_conv1 UINT8, NHWC(1x1x37984)
            Output qwen2_prefill96/qwen2_block29_conv2 UINT8, NHWC(1x1x37984)
            Output qwen2_prefill96/qwen2_block29_conv3 UINT8, NHWC(1x1x37984)
            Output qwen2_prefill96/qwen2_block29_conv4 UINT8, NHWC(1x1x37984)

Network group name: base_model__tbt, Multi Context - Number of contexts: 95
    Network name: base_model__tbt/qwen2_tbt
        VStream infos:
            Input  qwen2_tbt/input_layer3 UINT8, NHWC(1x1x1536)
            Input  qwen2_tbt/input_layer2 UINT8, NHWC(1x1x24576)
            Input  qwen2_tbt/input_layer5 UINT8, NHWC(1x1x256)
            Input  qwen2_tbt/input_layer1 UINT16, NHWC(1x1x1536)
            Input  qwen2_tbt/input_layer4 UINT8, NHWC(1x1x1536)
            Input  qwen2_tbt/input_layer6 UINT8, NHWC(1x1x256)
            Output qwen2_tbt/qwen2_block29_conv1 UINT8, NHWC(1x1x37984)
            Output qwen2_tbt/qwen2_block29_conv2 UINT8, NHWC(1x1x37984)
            Output qwen2_tbt/qwen2_block29_conv3 UINT8, NHWC(1x1x37984)
            Output qwen2_tbt/qwen2_block29_conv4 UINT8, NHWC(1x1x37984)
```
We first have a compability range which specifies which chips this HEF can run on.
Then we have the neural network groups, which in this case there are two. There
is one for the prefill, the prompt processing stage, and one for the token
generation (token by token) stage.

If a model is small enough the entire network maps onto the silicon completely and
we are good to go. Data streams into the chip, flows through the static hardware
routing, and streams out. This is referred to as a single-context.
The qwen model above is too large to fit on the chip, so it is split into multiple
contexts. My understanding of this is that the first layer will be will be configured
by the driver to configure the routing registers, and load the weights/biases for
the first layer. Input data will stream accross the PCIe bus and be processes
through the first layers configuration, and the intermediate results are held
in a internal temp buffer. Then the first context is switched/swapped with the
second context. And this continues.

Notice the number `96` in the network name:
```console
    Network name: base_model__prefill/qwen2_prefill96
```
This is the sequence length, the number of tokens that are processed in a single
execution. This is also different from what we might be used to in llama.cpp where
the sequence length is dynamic. But since the Hailo compiler has to physically
map the model's layers and internal attention matrices directly onto silico
nmemory cells, the input sizes must be completely static.

Also notice that we have not just one input either, but 6 different inputs.
```console
            Input  qwen2_prefill96/input_layer3 UINT8,  NHWC(1x96x1536)
            Input  qwen2_prefill96/input_layer1 UINT16, FCR (1x96x1536)
            Input  qwen2_prefill96/input_layer2 UINT8,  FCR (1x96x24576)
            Input  qwen2_prefill96/input_layer6 UINT8,  FCR (1x96x256)
            Input  qwen2_prefill96/input_layer5 UINT8,  FCR (1x96x256)
            Input  qwen2_prefill96/input_layer4 UINT8,  FCR (1x96x1536)

FCR = ?
```
Ignore the numbering of inputs and outputs, these are generated by the compiler
and don't necessarily have to be in order. But I think the order they are in
is somewhat logical, the first line is the input token embeddings:
```console
Input  qwen2_prefill96/input_layer3 UINT8,  NHWC(1x96x1536)
```
This is our prompt data which is in UNIT8. So we would lookup the tokens ids to
get 1536 dimensional embeddings for each of the 96 tokens. And we would also need
to quantize them to UINT8.

Following that we have:
```console
            Input  qwen2_prefill96/input_layer1 UINT16, FCR (1x96x1536)
```
Notice that this tensor has a different data type, UINT16. This is for RoPE and
need to remain in higher precision.

Then we have the attetion mask:
```console
            Input  qwen2_prefill96/input_layer2 UINT8,  FCR (1x96x24576)
```

In llama.cpp where we might have static tensors that are used in layers in Hailo
it does not want to use its precious memory cells for this so they are instead
passed in by the caller.
```console
        Input  qwen2_prefill96/input_layer6 UINT8,  FCR (1x96x256)
```
In qwen2, they have 12 query heads, and it used grouped query attention so multiple
query heads share the same key and value heads. So we have 12 query heads, but
only 2 key and value.
```console
2 KV heads * 128 dimensions per head = 256
```
TODO: take a closer look at this later when implmenting and exactly what data
we need for input_layer 4, 5, and 6.

And then we have the output:
```console
            Output qwen2_prefill96/qwen2_block29_conv1 UINT8, NHWC(1x1x37984)
            Output qwen2_prefill96/qwen2_block29_conv2 UINT8, NHWC(1x1x37984)
            Output qwen2_prefill96/qwen2_block29_conv3 UINT8, NHWC(1x1x37984)
            Output qwen2_prefill96/qwen2_block29_conv4 UINT8, NHWC(1x1x37984)

37984 * 4 = 151936 bytes
```
The vocabulary size of the Qwen2.5 tokenizer is exactly 151936 tokens.

_wip_
