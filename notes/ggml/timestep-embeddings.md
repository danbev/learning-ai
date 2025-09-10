## Timestep Embeddings
So I was looking into an issue in ggml where there was a windows specific test
that was failing intermittently. After looking into it I found the that the
issue was due to the zero padding for odd dimensions in
`ggml_compute_forward_timestep_embedding_f32`.  The issue was that currently if
an odd dimension is used, the padding check incorrectly uses the dimension value
for indexing. For example, with dim=15:

Elements 0-6 are set to cosine values
Elements 7-13 are set to sine values
Element 14 is left uninitialized (contains garbage)
Element 15 is correctly set to zero

This fix changes embed_data[dim] to embed_data[2 * half] so that element 14 (the
first unused element) is properly set to zero as well as the last element.

So what are these timestep embeddings?  
My understand is that they are used in diffusion models to encode the current
timestep of the diffusion process into a vector that can be used by the neural
network.

Let start by looking in `ggml.c`:
```c++
// ggml_timestep_embedding

struct ggml_tensor * ggml_timestep_embedding(
        struct ggml_context * ctx,
        struct ggml_tensor  * timesteps,
        int                   dim,
        int                   max_period) {
    int actual_dim = dim;
    if (dim % 2 != 0) {
        actual_dim = dim + 1;
    }

    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, actual_dim, timesteps->ne[0]);

    ggml_set_op_params_i32(result, 0, dim);
    ggml_set_op_params_i32(result, 1, max_period);

    result->op     = GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    return result;
}
```
So this function takes in a timestep tensor a dimension and a max period. And
notice that it checks if the dimension is odd and if so it adds one to make it
even.  So the actual dimension if we pass in 15 will be 16.  Then it set up
the parameters and notice the dim is the original dim passed in (so 15 in this
case). And the operation is GGML_OP_TIMESTEP_EMBEDDING.

And then we can look at the forward_compute function for the cpu backend which
we can find in `ggml-cpu.c`:
```c++
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);
    ...
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        ...
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        ...
```
This will land in `ggml/src/ggml-cpu/ops.cpp`:
```c++
void ggml_compute_forward_timestep_embedding(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}


static void ggml_compute_forward_timestep_embedding_f32(
    const ggml_compute_params * params,
    ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // Dimension will be the original dim passed in, for example 15
    const int dim = ggml_get_op_params_i32(dst, 0);
    const int max_period = ggml_get_op_params_i32(dst, 1);

    int half = dim / 2; // for dim=15, half=7

    // i is the number of timesteps being processed.
    for (int64_t i = 0; i < ne00; i++) {
        // cast so that we can do pointer arithmetic
        float * embed_data = (float *)((char *)  dst->data +  i*nb1);

        // loop through half the dimension (will updated both halfs in the same loop)
        // j is the frequency index being computed.
        for (int64_t j = ith; j < half; j += nth) {
            // src0 contains the timestep values
            float timestep = ((float *)src0->data)[i];

            float freq = (float) expf(-logf(max_period) * j / half);
            // multiply the frequency by the timestep
            float arg = timestep * freq;

            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }

        if (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
            embed_data[dim] = 0.f;
        }
    }
}
```
So the idea here is that we have a timesteps vector which just contains scalar
values representing the current timestep in the diffusion process, so one entry
might be 12.
```
j=0: freq = 1.000000    (fast oscillation)
j=1: freq = 0.268270    (medium-fast)
j=2: freq = 0.071969    (medium)
...
j=6: freq = 0.000373    (very slow oscillation)
```
These frequencies create different "time scales":
* High frequency (j=0): Captures fine-grained differences between nearby timesteps
* Low frequency  (j=6): Captures broad patterns across many timesteps
 
And for each frequency we multiply the timestep by the frequency
```
arg = 12 * 1.0   = 12.0 (fast oscillation at timestep 12)
arg = 12 * 0.268 = 3.22 (medium oscillation at timestep 12)
```
```
Position 0 (cos): 0.8439    Position 7 (sin): -0.5366   [from j=0, high freq]
Position 1 (cos): -0.9970   Position 8 (sin): -0.0776   [from j=1, med freq]
Position 2 (cos): 0.6497    Position 9 (sin): 0.7602    [from j=2, med freq]
...
```
Every timestep gets a unique vector pattern and similar timesteps get similar
vectors.
This is why diffusion models can learn that "step 10 needs different processing
than step 500" - the embeddings make these temporal positions distinguishable
and meaningful to the neural network.

This is pretty similar to absolute positional embeddings used in transformers:
* LLM Position Embeddings: "Where am I in this sentence/document?"
* Timestep Embeddings    : "Where am I in this denoising process?"

The formulas are nearly identical:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
```
embed[i]      = cos(timestep * freq_i)
embed[i+half] = sin(timestep * freq_i)

where freq_i  = 1 / 10000^(i/half)
```
In a transformer that uses absolute positional embeddings, the positional vector
is added to the token embeddings at the input layer. But for a diffusion model
the process is different:
```
Input: noisy_image (tensor) + timestep (scalar like 250)
```
Generate timestep embedding vector:
```
timestep_scalar = 250
timestep_embedding = ggml_timestep_embedding(timestep_scalar, dim=128)
// Creates: [0.2, -0.8, 0.5, 0.1, ...] (128-dimensional vector)
```
```
timestep_embedding → MLP layers → conditioning_vector
```
This often expands the embedding (e.g., 128 → 512 dimensions) and learns optimal
representations.
Unlike transformers, the timestep conditioning is injected at multiple points
throughout the network:
ResNet blocks, attention layers, etc. 

The key insight is that every layer needs to know the timestep because:
* Early layers: Detect what level of noise to expect
* Middle layers: Apply appropriate feature transformations
* Late layers: Generate the right amount of denoising
