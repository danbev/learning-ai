## Backend Scheduling Issue (maybe)
This is an issue I've run into when trying to get a multi-modal vision model
to work with the new Vision API.

My goal with this was to get some experience with the new Vision API and also
with a multi-modal model that uses cross-attention, like Llama 3.2 Vision, so
that I can hopefully contribute to this part of the project in the future.

To get something working I looked at Ollama's support for Llama 3.2 Vision
Instruct and the model the [provide](https://ollama.com/x/llama3.2-vision).
They have two models, one for the language model and one for the vision encoder.
In our case I made the assumption that we only want one model so that that is
what I opted for.

I wanted to follow the new Vision API and the Llava example that was provided
in https://github.com/ggerganov/llama.cpp/pull/9687. So I used the same image to
try to reproduce the same/simliar output.

### The Issue
While developing/debugging the model I added a number of tensors that are copies
of tensors used in the computation graph so that I could inspect their output
if the original tensor gets resued by the backend schdular, which I think is
something that it can do with tensors that are part of the graph. So this is a
way to inspect the output of a tensor which might get reused by the backend
scheduler.

So I added tensors like this:
```c++
    struct ggml_tensor * inp_raw_copy = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, image_size_width, image_size_height, n_channels, n_tiles);
    ggml_set_input(inp_raw_copy);
    ggml_set_name(inp_raw_copy, "inp_raw_copy");
    ggml_build_forward_expand(gf, inp_raw_copy);
```
Now, running the [example] with this code will produce a pretty resonable
output:
```console
The image shows a close-up of the Eiffel Tower in Paris, France. The tower is
made of metal and has a dark gray color. It is shaped like a square with four
sides, and it has a flat top. The background of the image is a light gray color.
```
This might not be perfect but at least the image is described and the vision
encoder produces something that the language model can also work with.

Now, if I comment out the code above, the output will be different. The output
will be something like:
```console
"The image shows a gray background..."
```

I initially thought it was because the image patch embeddings were not
being generated correctly, but when I've checked the output image patch
embeddings (uncommenting the code in `encode_image_with_ca_vision`) using:
```console
$ sha256sum image_patch_embeddings.bin
319cc0572866e3b165d5f59dc3e5709b87ec503ff6af10e3cd487d21e2ad17ab  image_patch_embeddings.bin
```
The image patch embeddings are the same with this code commmented out or not,
so it does not seem like removing this tensor has an effect on the image patch
embeddings (the vision encoder).

I also noticed that if I increase the number of layers that I offload to the GPU
this also effect the output. For example, if I change the number of layers from
30 to 36 the I will also see the output above with the "gray background".

It seems to me like if I make a change to the computation graph of the vision
model this can have an effect on the language model which I was not expecting
(not saying it is wrong as I'm unfamiliar the inner workings of the backend
scheduler). Almost like the graph are shared but I was thinking that they would
not be after calling `ggml_backend_sched_reset(ctx.sched)`.

Does anyone recognize this issue, or have any ideas where I should start looking
to try to figure this out?

The [example] contains the steps to convert the model, quantize it, and also run
it.

[example]: https://github.com/danbev/llama.cpp/tree/vision-api-mllama-example/examples/simple-vision-mllama#simple-vision-mllama-example
