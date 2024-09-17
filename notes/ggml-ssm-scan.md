###  Mamba State Space Model Scan Operation
This is the paralell scan operation for the SSM block in Mamba.
There is a standalone example in [ssm_scan.c](../fundamentals/ggml/src/ssm_scan.c)


### Implementation
```c
static void ggml_compute_forward_ssm_scan_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // s
    const struct ggml_tensor * src1 = dst->src[1]; // x
    const struct ggml_tensor * src2 = dst->src[2]; // dt
    const struct ggml_tensor * src3 = dst->src[3]; // A
    const struct ggml_tensor * src4 = dst->src[4]; // B
    const struct ggml_tensor * src5 = dst->src[5]; // C

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nc  = src0->ne[0]; // d_state
    const int64_t nr  = src0->ne[1]; // d_inner
    const int64_t n_t = src1->ne[1]; // number of tokens per sequence
    const int64_t n_s = src0->ne[2]; // number of sequences in the batch

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    const int ir  = ir1 - ir0;
```
In the example I've set the threads to 1 to simplify stepping through the code. This
enables us to ignore the threading code.
```console
(lldb) p nc
(const int64_t) 16
(lldb) p nr
(const int64_t) 8
(lldb) p n_t
(const int64_t) 4
(lldb) p n_s
(const int64_t) 1
```
Perhaps `nc` stands for number of channels?

The main loop will loop over all the sequences, which in our case is only 1:
```c
    for (int i3 = 0; i3 < n_s; ++i3) {
```
Next it will loop over all the tokens in each sequence:
```c
        for (int i2 = 0; i2 < n_t; ++i2) {
```
Next we have a pointer to the data in the s tensor:
```c
            const float * s0 = (const float *) ((const char *)
                src0->data + ir0*(src0->nb[1]) + i3*(src0->nb[2]));
```
Now, the const char cast is there to enable pointer arithmetic. `src0` is the s
tensor which is the tensor after the input embeddings have been projected into
the inner state dimensions, gone through the convolution layer, and the Silu
operation.
The `s` tensor is the current state of the system.

So it will look like something like this:
```
                d_state
         0 [0  ...        15]
         1 [0  ...        15]
         2 [0  ...        15]
         3 [0  ...        15]     d_inner
         4 [0  ...        15]
         5 [0  ...        15]
         6 [0  ...        15]
         7 [0  ...        15]
```
And since we are only using one thread we can ignore ir0 as it will be zero:
```c
            const float * s0 = (const float *) ((const char *)
                src0->data + i3*(src0->nb[2]));
```
And we only have one sequence which is represented by i3 so this is also zero at
this point:
```c
            const float * s0 = (const float *) ((const char *)
                src0->data;
```
So for this iteration `s0` is simply a pointer to the beginning of s data.

Next we have our x tensor, which is the input to the SSM block. This is the
output of the input embeddings->projection layer->convolution layer->Silu:
```c
            const float * x  = (const float *) ((const char *)
                src1->data + ir0*(src1->nb[0]) + i2*(src1->nb[1]) + i3*(src1->nb[2]));
```
This looks like this:
```
             d_inner
  token 0 [0  ...  7]
  token 1 [0  ...  7]     seq_len
  token 2 [0  ...  7]
  token 3 [0  ...  7]
```
And again we can do the same simplification as above:
```c
            const float * x  = (const float *) ((const char *)
                src1->data;
```
So x will be a pointer to the beginning of x data.

Then we have dt (delta):
```c
            const float * dt = (const float *) ((const char *) src2->data + ir0*(src2->nb[0]) + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {d_inner, n_t, n_s}
```
This has the same shape as x:
```
            d_inner
  token 0 [0  ...  7]
  token 1 [0  ...  7]     seq_len
  token 2 [0  ...  7]
  token 3 [0  ...  7]
```
And with the same simplification as above:
```c
            const float * dt = (const float *) ((const char *)
                src2->data;
```
So dt will be a pointer to the beginning of dt data.

Next we have the A tensor (state transition matrix):
```c
            const float * A  = (const float *) ((const char *) src3->data + ir0*(src3->nb[1])); // {d_state, d_inner}
```
And A looks like this:
```
                d_state
         0 [0  ...        15]
         1 [0  ...        15]
         2 [0  ...        15]
         3 [0  ...        15]     d_inner
         4 [0  ...        15]
         5 [0  ...        15]
         6 [0  ...        15]
         7 [0  ...        15]
```
And we'll simplify this as well:
```c
            const float * A  = (const float *) ((const char *)
                src3->data;
```
Next we have the B tensor (input state transition matrix):
```c
            const float * B  = (const float *) ((const char *) src4->data +  i2*(src4->nb[1]) + i3*(src4->nb[2])); // {d_state, n_t, n_s}
```
```
            d_state
  token 0 [0  ...  7]
  token 1 [0  ...  7]     seq_len
  token 2 [0  ...  7]
  token 3 [0  ...  7]
```
And again we'll simplify this:
```c
            const float * B  = (const float *) ((const char *)
                src4->data;
```
Next we have the C tensor (output transistion matrix):
```c
            const float * C  = (const float *) ((const char *) src5->data +  i2*(src5->nb[1]) + i3*(src5->nb[2])); // {d_state, n_t, n_s}
```
And this looks like this:
```
            d_state
  token 0 [0  ...  7]
  token 1 [0  ...  7]     seq_len
  token 2 [0  ...  7]
  token 3 [0  ...  7]
```
And we'll simplify this as well:
```c
            const float * C  = (const float *) ((const char *)
                src5->data;
```
Next we have `y` which is the ouput (y = ...):
```c
                  float * y  = (      float *) ((      char *) dst->data  + ir0*(src1->nb[0]) + i2*(src1->nb[1]) + i3*(src1->nb[2])); // {d_inner, n_t, n_s}
```
And this looks like this:
```
  0  [0                          ...                    159]
```
Note that this is not const so we can expect it to be updated.
```
                  float * y  = (      float *) ((      char *)
                      dst->data;
````
So this is currently just a pointer to the dst tensor data which is our
`y` tensor in the example code:
```console
(lldb) p dst
(ggml_tensor *) 0x000000013e000fc0
(lldb) p dst->name
(char[64]) "y"
(lldb) p dst->ne
(int64_t[4])  ([0] = 160, [1] = 1, [2] = 1, [3] = 1)
```
So `y` will be the output of this function. Note that this is a one dimensional
tensor and was created in the `ggml_ssm_scan` function:
```c
    // concatenated y + ssm_states
    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ggml_nelements(x) + ggml_nelements(s));
```
So it looks like the this will contain both the output values (y) and the
ssm_states.

Next we have an `s` tensor:
```c
                  float * s  = (      float *) ((      char *) dst->data  + ir0*(src0->nb[1]) + i3*(src0->nb[2]) +     src1->nb[3]);  // {d_state, d_inner, n_s}
```
Notice that this is also a pointer to dst data. If we simplify this we get:
```c
                  float * s  = (      float *) ((      char *)
                      dst->data  + src1->nb[3]);
```
```console
(lldb) p src1->nb[3]
(const size_t) 128
```
So this is a pointer to the 128th element in the dst tensor data. Why 128?
So I think that they output y for each dimension is stored first in this tensor
so there will be 128 elements for the y values. 
```
  
  0                                   127            159
  [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15][1   2   3   4]
              y outputs 16*8=128           states 4*8=32
```

After that we have:
```
            if (i2 > 0) { s0 = s; }
```
So if we are not at the first token we set `s0` to the value of `s`, the last
computed state (not the output) I think.

After that we hae a final loop which is iterating over all the dimensions of the
inner state (`d_inner`) (in our case 8):
```
            for (int i1 = 0; i1 < ir; ++i1) {

```

```c
                float dt_soft_plus = dt[i1] <= 20.0f ? log1pf(expf(dt[i1])) : dt[i1];
```
So here we are taking the first element of delta and checking if it is less than
or equal to 20.0, and if it is then we pass it to the `log1pf` function which
computes the natrual logarithm for the delta value for this inner state. And if
the current delta value is greater than 20.0 we just use the delta value as is.

soft plus is defined as:
```
f(n) = ln(1 + e^n)
of 
f(n) = log(1 + exp(n))
```
In this case the + 1 is performed by log1pf (log plus 1 float).

Next we multiply the delta value with the current x value:
```c
                float x_dt = x[i1] * dt_soft_plus;
```
So we will have one value for each inner state dimension. This because this is
"broadcasted" (used for all the channels/dimensions below).
This is where the input is "mixed" with the delta value making the delta input
dependent.

Next we will iteratate over the `d_state` (16 in our case): 
```c
                float sumf = 0.0f;
                for (int i0 = 0; i0 < nc; ++i0) {
                    int i = i0 + i1*nc;
                    // state = prev_state * dA + dB * x
                    float state = (s0[i] * expf(dt_soft_plus * A[i])) + (B[i0] * x_dt);
                    // y = rowwise_dotprod(state, C)
                    sumf += state * C[i0];
                    s[i] = state;
                }
                y[i1] = sumf;
```

```c
                    float state = (s0[i] * expf(dt_soft_plus * A[i])) + (B[i0] * x_dt);
```
Just as a reminder the state space model is defined as:
```
h_t = Ā h_{t-1} + B̂ x_t

Where:
A_bar = is the state transition matrix.
B_bar = input projection matrix.
x_t   = the input at time t.
h_t   = the hidden state at time t.
h_t-1 = the previous hidden state.
```
Now, lets start from the right side and we can see that we are multiplying the
`x_dt` with the input transition matrix `B[i0]`. And `x_dt` is the
input value with the delta time step incorporated. Though it might not look like
it this is also the descretization of the input projection matrix. This is done
by the multiplication of `x_dt` which is possible because for small values
of delta `∫(0 to Δt) exp(A*τ) * B dτ` can be approximated by `Δt * B`.

Then we have the descretization of A which is done by `dt_soft_plus * A[i]`.
The `expf` function is part of the zero-hold order (ZOH) descretization which

And then we have the previous state value `s0[i]` multiplied by the descretized
state transition matrix.
And also notice that the new state is stored in `s[i]` for the next iteration.

So that gives as the updated state of the system. This is then multiplied by the
output transition matrix `C[i0]` and summed and stored in `sumf`. And this
state is also stored in s[i] for the next iteration (next token iteration that
is).
So that was the first iteration of the first channel/feature and we will then
do that same for the second channel/feature.
So notice that we do this operation for each channel/feature/dimension in the
inner state. 

```
                        (Channels/Features/Dimensions)
     0     1    2     3     4     5     6     7    8     9   10   11   12   13    14   15
    +--+  +--+ +--+  +--+  +--+  +--+  +--+  +--+ +--+ +--+ +--+ +--+ +--+ +--+ +--+ +--+
    |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  | |  | |  | |  | |  | |  | |  | |  |
    +--+  +--+ +--+  +--+  +--+  +--+  +--+  +--+ +--+ +--+ +--+ +--+ +--+ +--+ +--+ +--+
     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    
    +--+  +--+ +--+  +--+  +--+  +--+  +--+  +--+ +--+ +--+ +--+ +--+ +--+ +--+ +--+ +--+
SSM |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  | |  | |  | |  | |  | |  | |  | |  |
    +--+  +--+ +--+  +--+  +--+  +--+  +--+  +--+ +--+ +--+ +--+ +--+ +--+ +--+ +--+ +--+
     |     |    |     |     |     |     |     |    |    |    |    |    |    |    |    |
     +-----+----+-----+-----+-----+-----+-----+----+----+----+----+----+----+----+----+
                                        |
                                      +---+
                                      | + |
                                      +---+
                                        |
                                        ↓
                                      +---+
                                      | y |
                                      +---+
```
And this is for the first token. We then do the same for the rest of the tokens
in the sequence and the internal state is updated by the previous state. And
the output is stored in the y tensor which is the output to the next layer.
Just to recap we iterated over the sequences (`n_s`), then the number of tokens
in the sequence (`n_t`), then the number of states (`d_state`), and then the
channels in each state (`d_inner`).

So there will be a y value for each dimension of the inner state. But the state
of the system will be update for each channel.
```

```

_wip_

