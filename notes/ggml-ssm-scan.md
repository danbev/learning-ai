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
The main loop will loop over all the sequences, which in our case is only 1:
```c
    for (int i3 = 0; i3 < n_s; ++i3) {
```
Next it will loop over all the tokens in each sequence:
```c
        for (int i2 = 0; i2 < n_t; ++i2) {
```
Next we have:
```c
            const float * s0 = (const float *) ((const char *)
                src0->data + ir0*(src0->nb[1]) + i3*(src0->nb[2]));
```
Now, the const char cast is there to enable pointer arithmetic. `src0` s tensor
which is the tensor after then input embeddings have been projected into the
inner state dimensions, gone through the convolution layer, and the Silu operation.
So it will look like this:
```
             d_state
         0 [0  ...  7]
         1 [0  ...  7]     d_inner
         2 [0  ...  7]
         3 [0  ...  7]
         4 [0  ...  7]
         5 [0  ...  7]
         6 [0  ...  7]
         7 [0  ...  7]
```
And since we are only using one thread we can ignore ir0 as it will be zero:
```c
            const float * s0 = (const float *) ((const char *)
                src0->data + i3*(src0->nb[2]));
```
And we only have one sequence which is represented by i3 so this is also zero at this point.
```c
            const float * s0 = (const float *) ((const char *)
                src0->data;
```
So for this iteration `s0` is simply a pointer to the beginning of s data.

Next we have our x tensor:
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

Next we have the A tensor
```c
            const float * A  = (const float *) ((const char *) src3->data + ir0*(src3->nb[1])); // {d_state, d_inner}
```
And A looks like this:
```
             d_state
         0 [0  ...  7]
         1 [0  ...  7]     d_inner
         2 [0  ...  7]
         3 [0  ...  7]
         4 [0  ...  7]
         5 [0  ...  7]
         6 [0  ...  7]
         7 [0  ...  7]
```
And we'll simplify this as well:
```c
            const float * A  = (const float *) ((const char *)
                src3->data;
```
Next we have the B tensor:
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
Next we have the C tensor:
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
Next we have `y` which is the ouput (y = ...)
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
So this is currently just a pointer ot the dst tensor data which is our `result tensor`:
```console
(lldb) p dst
(ggml_tensor *) 0x000000013e000fc0
(lldb) p dst->name
(char[64]) "result"
(lldb) p dst->ne
(int64_t[4])  ([0] = 160, [1] = 1, [2] = 1, [3] = 1)
```
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
So this is a pointer to the 128th element in the dst tensor data. Why 128?
```
After that we have:
```
            if (i2 > 0) { s0 = s; }
```
So if we are not at the first token we set `s0` to the value of `s`.
After that we hae a final loop which is iterating over all the dimensions of the inner
state (`d_inner`) (in our case 8):
```
            for (int i1 = 0; i1 < ir; ++i1) {

```

```c
                float dt_soft_plus = dt[i1] <= 20.0f ? log1pf(expf(dt[i1])) : dt[i1];
```
So here we are taking the first element of delta and checkinf if it is less than or
equal to 20.0, and if it is then we pass it to the `log1pf` function which computes the
natrual logarithm for the delta value for this inner state. And if the current delta
value is greater than 20.0 we just use the delta value as is.

Next we multiply the delta value with the current x value:
```c
                float x_dt = x[i1] * dt_soft_plus;
```
So we will have one value for each inner state dimension. This because this is "broadcasted".
This is where the input is "mixed" with the delta value making the delta input depenant.

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
