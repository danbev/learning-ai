## Flash Attention
This is a type of attention which uses information about the GPU hardware to
optimize the attention mechanism. It is I/0 aware and also exact which means that
it doesn't use any approximation techniques.

Specifically [GPU](./gpu.md) hardware and memory transfers can hurt performance
where data is being copied to and from the GPU's SRAM to its High Bandwidth
Memory (HBM). So the idea is to minimize the number of memory transfers between
the GPU's HBM and SRAM.

In attention it is possible, and very likely, that the matrices cannot fit into
the fast SRAM of the GPU and what happens is that they are instead loaded from
the large global memory (HBM) to the fast SRAM. Each thread would operate on
parts of the matrices and the intermediate results would be written back to the
global memory (because we can't fit it in SRAM).

Most operations in Transformers are bottlenecked not by the compute but by the
memory transfers between the GPU's SRAM and HBM.

The normal attention mechanism works something like this:
* Load the Q, K, and V matrices from the GPU's HBM to SRAM (using tiling)
* Compute the S = QKᵀ, and then write S to HBM. (transfer to global memory which is slow)
* Load S from HBM to SRAM to compute P = softmax(S), and then write P to HBM.
* Load P from HBM to SRAM compute O = PV, and then write O to HBM.

I'm simplifiying things a bit here as Q, K, and V cannot fit into SRAM as they
would most often be too large. So tiling is used for standard attention as well.

The difference compared to flash attention is that standard attention treats the
attention as three separate operations (kernels). Since these are separate kernels
the GPU has to "save its work" to global memory between each step.

Standard attention:
* Step 1 MatMul Kernel (QK^T)
Input: Reads blocks of Q and K from HBM to SRAM (using tiling)
Compute: Calculates a block of scores S
The issue: It writes that block of S (the NxN matrix) back to HBM. It has to,
because this kernel's job is just "Matrix Multiplication," and it must output
the result

* Step 2: Softmax Kernel
Input: Reads the full NxN matrix S from HBM (using tiling)
Compute: Calculates Softmax
The issue: It writes the full NxN probability matrix P back to HBM. So have
have a NxN matrix in HBM now.

* Step 3: MatMul Kernel (PV)
Input: Reads blocks of P and V from HBM
Compute: Calculates the output O
Output: Writes O to HBM.

What Flash Attentions proposes is that the loading and storing back and forth
be minimized. So instead of storing S back to HBM, just use it straight away
to compute P and then compute the softmax. But this imposes an issue with
regards to softmax.

In flash attention something like the following is done:
* Load a small block of Q, K, V into SRAM
* Compute the attention scores (QKᵀ) for that just this block in SRAM
* Compute the softmax (using "online softmax") for this block in SRAM
* Multiply by V to get the output for this block in SRAM
* Write only the accumulated output O back to HBM.

The huge NxN matrix is never fully created in HBM. It is computed on-the-fly in
SRAM and then discarded.
So how does this actually work, when we compute the softmax we need the entire
matrix S to compute the softmax normalization (the denominator needs the sum of
all the exponentiated values)?  
This is solved by using "Online Softmax" (see below) which allows us to compute
the softmax incrementally as we load blocks of the S matrix. We can calculate
the "local" sum of exponentials for the current block, and when you load the next
block, we can mathematically update the running sum and rescale the previous
results.

### Tiling
This involves restructuring algorithms to load block by block from GPU High
Bandwidth Memory (HBM) to the GPU SRAM to compute the attention. So the idea is
to load blocks of Q, K, and V from the HBM to the SRAM. The on-chip SRAM is defined
as variable `M` in the paper, which I think in my case would be 128 KB which is
the size of the L1 cache for each Streaming Multiprocessor (SM) on my card), and
I have 46 SMs on my card. So the total SRAM is 128 * 46 = 5888 KB (I'm not
totally sure about these numbers).

### Online softmax
The softmax function is defined as follows:
```
x = [x1, x2, ..., xn]

softmax(x₁) =                exp(x₁-max)
               ----------------------------------
               (exp(x₁-max) + exp(x₂-max) + ... + exp(xₙ-max))

max = global max
```
If we only see a small block of the row (a tile), we don't know the global max,
we only know the local max of the current block.

As we see new blocks (and potentially find a new "max" value that is larger than
the old one), we retroactively correct our previous partial results by
multiplying them by a scaling factor.

Lets say the first block sees a max of 10, the softmax for this block is
computed using 10 as the local max. If the next block has 20 as its max then the
first blocks computation is not correct. So for block 1 we should actually have
calculated `(x^(10-20))` instead of `(x^(10-10))`. We can rescale block 1 result
by multiplying it by `exp(10-20)`, and then add block 2 result to it and the
accumulate sum will be correct.

Flow in the kernel (TODO: add links to ggml-cuda functions):
1. Load Q_block from global memory to SRAM
2. Initialize registers:
   This creates accumulators that live in registers:
     running_max: stores the maximum value seen so far for each row
     running_sum: stores the sum of exponentials for each row
     running_output: stores the accumulated output for each row
3. The loop over K and V blocks:
   Load K_block and V_block from global memory to SRAM
   Compute scores: S_block = Q_block * K_blockᵀ in registers(?)
   Online softmax correction:
     Compare local max vs running_max
     Update running_max
     Rescale running_sum and running_output if running_max changed (using e^(old_max - new_max))
   Accumulate: add the new contributions to running_output.
4. Final write back (after loop has completed)
   Write running_output to global memory.

### Flash Attention Algorithm
Now, recall that the Q, K, and V matrices are of size N (sequence length) x D
(embedding dim) and these will be stored in the GPU's HBM initially (global
memory).

Lets say we have a sequence length of 4 and embedding dimension of 512 so imagine
something like this:
```
                     D
           0                      511
'Dan':   0 [0.1, 0.2, ..., 0.3, 0.4]
'loves': 1 [0.5, 0.6, ..., 0.7, 0.8]  N
'ice':   2 [0.9, 1.0, ..., 1.1, 1.2]
'cream': 3 [1.3, 1.4, ..., 1.5, 1.6]
```

Now, the first things that are done are actually setting the block sizes:
(I'm assuming M = 5888 KB as that is what I think my GPU has)
```
B_c = floor(M / (4*D)
B_c = floor(5888 / (4*512)
B_c = floor(5888 / 2048)
B_c = floor(2.87)
B_c = 2

B_r = min(B_c, D)
B_r = min(2, 512)
B_r = 2
```
Could c be for column and r for row?
```
T_r = floor(N/B_r)
T_r = floor(4/2)
T_r = 2

T_c = floor(N/B_c)
T_c = floor(4/2)
T_c = 2
```
The second step is to initialize the ouput matrix O in the GPU's HBM:
```
O = (0)NxD ε ℝᴺˣᴰ
           0             511
         0 [0, 0, ..., 0, 0]
         1 [0, 0, ..., 0, 0]
         2 [0, 0, ..., 0, 0]
         3 [0, 0, ..., 0, 0]
```

And then we initialize the m and ℓ vectors in the GPU's HBM:
```
m = (-inf)N ε ℝᴺ
            0       1     2     3
         0 [-inf, -inf, -inf, -inf]

ℓ = (0)N ε ℝᴺ
            0  1  2  3
         0 [0, 0, 0, 0]
```
These are used for the softmax calculations which we will discuss later but for
now just know that these vectors are stored and updated during the attention
processing.

To recap, the operations on the Q and K matrices is matrix multiplication:
(remember that the K matrix is transposed)
```
   +----------------+   +-------+
   |      Q         |   |  Key  |
   |                | X |       |
   |  (4, 512)      |   |(512,4)|
   +----------------+   |       |
                        |       |
                        |       |
                        +-------+

   +----------------+
   |      V         |
   |                |
   |  (4, 512)      |
   +----------------+
```

The third step in the algorithm to split Q into blocks size `T_r`:
```
Q₁ = 'Dan':   0 [0.1, 0.2, ..., 0.3, 0.4]   (B_r x D)
     'loves': 1 [0.5, 0.6, ..., 0.7, 0.8]

Q₂ = 'ice':   2 [0.9, 1.0, ..., 1.1, 1.2]   (B_r x D)
     'cream': 3 [1.3, 1.4, ..., 1.5, 1.6]
```

And also divide K into blocks of size `T_c` blocks:
```
K₁ = 'Dan':   0 [0.1, 0.2, ..., 0.3, 0.4]   (B_c x D)
     'loves': 1 [0.5, 0.6, ..., 0.7, 0.8]

K₂ = 'ice':   2 [0.9, 1.0, ..., 1.1, 1.2]   (B_c x D)
     'cream': 3 [1.3, 1.4, ..., 1.5, 1.6]

I'm just showing the transpose of K here to visualize it: 
Kᵗ:
            'Dan'   'loves'   'ice'   'cream'
              0        1        2        3
            [ 0.1      0.5      0.9      1.3]
            [ 0.2      0.6      1.0      1.4]
            [ ...      ...      ...      ...]
            [ 0.3      0.7      1.1      1.5]
            [ 0.4      0.8      1.2      1.6]
```

And also divide V into blocks of size `T_c` blocks:
```
           0                      511
V₁ = 'Dan':   0 [0.1, 0.2, ..., 0.3, 0.4]   (B_c x D)
     'loves': 1 [0.5, 0.6, ..., 0.7, 0.8]

V₂ = 'ice':   2 [0.9, 1.0, ..., 1.1, 1.2]   (B_c x D)
     'cream': 3 [1.3, 1.4, ..., 1.5, 1.6]
```

Next the Output matrix is also split into blocks of size `T_r`:
```
O₁ =        0             511
         0 [0, 0, ..., 0, 0]
         1 [0, 0, ..., 0, 0]
O₂ =     2 [0, 0, ..., 0, 0]
         3 [0, 0, ..., 0, 0]
```
And then the ℓ vector is split into blocks of size `T_r`:
```
ℓ₁ =    0 [0, 0]
ℓ₂ =    1 [0, 0]
```

And then the m vector is split into blocks of size `T_r`:
```
ℓ₁ =    0 [-inf, -inf]
ℓ₂ =    1 [-inf, -inf]
```

The softmax function is defined as follows for reference:
```
x = [x1, x2, ..., xn]

softmax(x₁) =                exp(x₁)
               ----------------------------------
               (exp(x₁) + exp(x₂) + ... + exp(xₙ))
```

```
for (int j = 0; j < T_c; j++) {
    load K_j from HBM to SM SRAM         [0.1, 0.2, ..., 0.3, 0.4]
                                         [0.5, 0.6, ..., 0.7, 0.8]

    load V_j from HBM to SM SRAM         [0.1, 0.2, ..., 0.3, 0.4]
                                         [0.5, 0.6, ..., 0.7, 0.8]
    for (int i = 0; i < Tr; i++) { 
      load Q_i from HBM to SM SRAMᵢ      [0.1, 0.2, ..., 0.3, 0.4]
                                         [0.5, 0.6, ..., 0.7, 0.8]

      load O_i from HBM to SM SRAMᵢ      [0,     0, ...,   0,   0]
                                         [0,     0, ...,   0,   0]

      load ℓ_i from HBM to SM SRAMᵢ      [0, 0]
      load m_i from HBM to SM SRAMᵢ      [0, 0]
      compute S_i_j = Q_iKᵗ:
         [0.1, 0.2, ..., 0.3, 0.4] x [0.1 0.5] = [x x]
         [0.5, 0.6, ..., 0.7, 0.8]   [0.2 0.6]   [x x]
                                     [... ...]
                                     [0.3 0.7]
                                     [0.4 0.8]
      _
      m_i_j = row_max(S_i_j) // [max_row_0 max_row1]
      _
      P_i_j = exp(S_i_j - m_i_j) // subtract max from each element in the matrix
      // and then exponentiate.
      _            _
      ℓ_i_j rowsum(P_i_j) // sum each row in the matrix producing a vector of
      // length 2 with the sum of each row.
                         _
      m_i_new = max(m_i, m_i_j) // m_i is the vector from HBM, size is 2 as we
      // have two rows and each contain the max of that row.
                                               _                  _
      ℓ_i_new = exp(m_i - m_i_new) * ℓ_i + exp(m_i_j - m_i_new) * ℓ_i_j
      // vector of size 2, with the denominator for the softmax.

      // Keep in mind that the following is updating the 0_i output matrix
      O_i = diag(ℓ_i_new)⁻¹ * (diag(ℓ_i) * exp(m_i - m_i_new) * O_i + P_i_j * V_j) // 0_i is written back to HBM
      // diag(ℓ_i_new) = [a 0]  // this is a diagonal matrix of [x x] vector.
      //                 [0 b]
      // The inverse operation takes the reciprocal of each diagonal element:
      //                 [1/a   0]
      //                 [0   1/b]
      // Each element is then multiplied by the current ℓ_i after multiplying
      // it by the  exponential of the difference between the current m_i and
      // the new m_i
      // (diag(ℓ_i) * exp(m_i - m_i_new)
      // This is creating a diagonal matrix of the ℓ_i vector (from HBM and
      // contains the softmax denominator from the last iteration). This is
      // them multiplied element-wise by the exponential of the difference
      // between the current m_i and the new m_i. This is then multiplied by
      // O_i which is the output matrix from the last iteration.
      // Next, P_i_j is multiplied by V_j and added to the previous result.
      // This results in the updated O_i, which now holds the attention output
      // for the current block. This output is a combination of the scaled
      // previous output and the current attention scores, normalized
      //  appropriately.

      // Recall that 0_i is part of the inner loop here and it will be updated
      // for each interation. So the first time through the look the initially
      // state of m_i will be all negative infinities, and ℓ_i will be all zeros.
      // The initial values of m_i (negative infinities) imply that no maximum
      // values have been encountered in the softmax computation yet. As a result,
      // m_new_i in this first iteration effectively becomes the row-wise max
      // of the current block (m~_i_j).
      // Similarly, the initial zeros in ℓ_i mean that the cumulative sum of 
      // exponentiated values (used in the softmax denominator) is starting from 
      // zero. Hence, ℓ_new_i accumulates these sums starting from this iteration.
      // This will cause the first iteration to calculate an incorrect softmax
      // value as it might not have the correct value for the denominator yet.
      // But  when the second (last) iteration comes it will have the correct value
      // and the softmax will be correct.
      ℓ_i = ℓ_new // written back to HBM
      m_i = m_new // written back to HBM
    }
}
```

### Recomputation (backward pass)
Another things is that is done is that the attention matrix is not stored during
the forward pass, that is the values that would be used for calculating the
gradients so that they can be used in the backward pass. Instead the attention
S = QKᵀ is recomputed during the backward pass. But the softmax normalization
from the forward pass is stored in HBMs and used in the backward pass so it
does not have to be recomputed. "Computation is cheap, and memory reading and
writing is expensive.". So Flash Attention actual performs more operations, 
FLOPS than standard attention but it is faster because it does not have to
read and write to memory as much.


### Results
Provides 2-4 times speedup over standard attention and the performance is
exact, that is there is no approximation involved.


### Flash Attention 2
TODO: 


### Debugging
Build with `-G -g` and DCMAKE_BUILD_TYPE=Debug, for example:
```console
cmake --fresh -S . -B build -DCMAKE_BUILD_TYPE=Debug\
    -DCMAKE_CUDA_FLAGS="-G -g" \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_SANITIZE_THREAD=OFF \
    -DLLAMA_LLGUIDANCE=ON \
    -DGGML_OPENMP=ON
cmake --build build -j8
```

Debugging session:
```console
(cuda-gdb) br fattn-mma-f16.cuh:1540  (ggml_cuda_flash_attn_ext_mma_f16_case, before kernel launch).)
(cuda-gdb) br fattn-common.cuh:984
(cuda-gdb) br fattn-mma-f16.cuh:1352
Note: breakpoint 2 also set at pc 0x7fff9529e040.
Breakpoint 3 at 0x1140: fattn-mma-f16.cuh:1352. (2 locations)
```

Just to understand how we end up in the kernel, when the operation GGML_OP_FLASH_ATTN_EXT
is executed this will be done by the CUDA backend in this case. The tensors below
have their buffers allocated on the GPU, not the host. So they exist in the GPUs
global memory (HBM) at this stage.

This should hit a break point when we run the program:
```console
Thread 1 "llama-completio" hit Breakpoint 3.76, ggml_cuda_flash_attn_ext_mma_f16_case<64, 64, 8, 1> (ctx=..., dst=0x555556df6980)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../fattn-mma-f16.cuh:1540
1540	    launch_fattn<DV, ncols1, ncols2>
```
And to see how we got here:
```console

(cuda-gdb) bt
#0  ggml_cuda_flash_attn_ext_mma_f16_case<64, 64, 8, 1> (ctx=..., dst=0x555556df6980)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/template-instances/../fattn-mma-f16.cuh:1540
#1  0x00007fffdaa22ab8 in ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<64, 64, 1> (ctx=..., dst=0x555556df6980)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/fattn.cu:16
#2  0x00007fffdaa216a8 in ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<64, 64> (ctx=..., dst=0x555556df6980)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/fattn.cu:78
#3  0x00007fffdaa20176 in ggml_cuda_flash_attn_ext_mma_f16 (ctx=..., dst=0x555556df6980)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/fattn.cu:91
#4  0x00007fffdaa21042 in ggml_cuda_flash_attn_ext (ctx=..., dst=0x555556df6980)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/fattn.cu:372
#5  0x00007fffdaa43140 in ggml_cuda_compute_forward (ctx=..., dst=0x555556df6980)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:2712
#6  0x00007fffdaa483f8 in evaluate_and_capture_cuda_graph (cuda_ctx=0x555556dc9680, cgraph=0x555556dc6e18,
    graph_evaluated_or_captured=@0x7fffffff6538: false, use_cuda_graph=@0x7fffffff652a: false,
    cuda_graph_update_required=@0x7fffffff652b: true) at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3677
#7  0x00007fffdaa48d14 in ggml_backend_cuda_graph_compute (backend=0x555556dc40b0, cgraph=0x555556dc6e18)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:3809
#8  0x00007ffff7eef4f1 in ggml_backend_graph_compute_async (backend=0x555556dc40b0, cgraph=0x555556dc6e18)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend.cpp:364
#9  0x00007ffff7ef4610 in ggml_backend_sched_compute_splits (sched=0x555556d82050)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend.cpp:1580
#10 0x00007ffff7ef55d5 in ggml_backend_sched_graph_compute_async (sched=0x555556d82050, graph=0x555556dd2620)
    at /home/danbev/work/ai/llama.cpp/ggml/src/ggml-backend.cpp:1803
#11 0x00007ffff7a7a612 in llama_context::graph_compute (this=0x555556dc4310, gf=0x555556dd2620, batched=true)
    at /home/danbev/work/ai/llama.cpp/src/llama-context.cpp:2070
#12 0x00007ffff7a759a5 in llama_context::process_ubatch (this=0x555556dc4310, ubatch=..., gtype=LLM_GRAPH_TYPE_DECODER,
    mctx=0x555556ef8260, ret=@0x7fffffffad44: GGML_STATUS_SUCCESS) at /home/danbev/work/ai/llama.cpp/src/llama-context.cpp:1094
#13 0x00007ffff7a77a2a in llama_context::decode (this=0x555556dc4310, batch_inp=...)
    at /home/danbev/work/ai/llama.cpp/src/llama-context.cpp:1532
#14 0x00007ffff7a7fae4 in llama_decode (ctx=0x555556dc4310, batch=...) at /home/danbev/work/ai/llama.cpp/src/llama-context.cpp:3438
#15 0x000055555566dd93 in main (argc=16, argv=0x7fffffffd648) at /home/danbev/work/ai/llama.cpp/tools/completion/completion.cpp:679
```
Now, we are still on the host as we can see. When we later switch to the device
the backtrace will only be for the frames on the device.

```c++
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    ...

    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true);
```
We can find `launch_fattn` in ggml/src/ggml-cuda/fattn-common.cuh and this is
still on the host side. Notice that the kernel is passed as a parameter to this
function (`fattn_kernel`):
```c++
template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const bool is_mla = DV == 512; // TODO better parameterization

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
```
Lets take a look at `dst`, `Q`, `K`, and `V`:
```console
(cuda-gdb) p *dst
$9 = {type = GGML_TYPE_F32, buffer = 0x555557312840, ne = {64, 14, 7, 1}, nb = {4, 256, 3584, 25088},
op = GGML_OP_FLASH_ATTN_EXT, op_params = {1040187392, 0, 0, 10, 0 <repeats 12 times>}, flags = 0,
src = {0x555558b79a20, 0x555558b79b90, 0x555558b79d00,
    0x555558b77a80, 0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>,
    0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>,
    0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>,
    0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>,
    0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>,
    0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>},
view_src = 0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>, view_offs = 0,
data = 0x7fff4d60ca00, name = "__fattn__-0", '\000' <repeats 52 times>,
extra = 0x0 <_INTERNAL_aedbb58d_7_norm_cu_9a3dffe9::min(unsigned int, int)>,
padding = "\000\000\000\000\000\000\000"}

(cuda-gdb) p Q->ne
$10 = {64, 7, 14, 1}

(cuda-gdb) p K->ne
$11 = {64, 256, 2, 1}

(cuda-gdb) p V->ne
$12 = {64, 256, 2, 1}
```
So dst is the GGML_OP_FLASH_ATTN_EXT which as created using:
```c++
        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
```

```c++
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;
```
```console
(cuda-gdb) p mask->ne
$16 = {256, 7, 1, 1}
```
And just to clarify these tensor buffers are all on the GPU, in global memory so
if we pass a pointer to them to a kernel it will work as expected, the kernel
can read write to these tensors in global memory. And trying to access them on
the host side would not work as expected.

```c++
    const int ntiles_x = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int ntiles_total = ntiles_x * (Q->ne[2] / ncols2) * Q->ne[3];
```

```console
(cuda-gdb) p ntiles_x    
$19 = 1
(cuda-gdb) p ntiles_total
$20 = 14
```

```c++
    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared);
    int parallel_blocks = max_blocks_per_sm;
```
```console
(cuda-gdb) p block_dim
$22 = {x = 32, y = 4, z = 1}

(cuda-gdb) p max_blocks_per_sm
$23 = 2

(cuda-gdb) p nsm
$24 = 46
```

The following block is setting the blocks_num for Stream-K mode:
```c++
    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_total + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks*tiles_nwaves);

        const int nblocks_stream_k = max_blocks;

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || tiles_efficiency_percent < 75;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;

        if (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(blocks_num.x) * ncols * (2 + DV/2)));
        }
```
```console
(cuda-gdb) p max_blocks
$25 = 92
```

Then we have the rest of the parameters being setup for the kernel launch:
```c++
    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
    );
```

The third argument is the function pointer to the kernel:
```console
(cuda-gdb) p fattn_kernel
$21 = (fattn_kernel_t) 0x7fffdbc1b329 <flash_attn_ext_f16<64, 64, 8, 1, false, false>>
```

In ggml/src/ggml-cuda/fattn-mma-f16.cuh we have the following function:
```c++
template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool mla>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2),
                  ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                        const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE))

    // Skip unused kernel variants for faster compilation:
    // Note that this uses template parameters so this is a compile time check.
    // This could use if constexpr for clarity but this works too.
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256)) {
        NO_DEVICE_CODE;
        return;
    }
```
This is the is "outer loop" which calculates coordinates, and tells the GPU
threads which specific tiles to work on. The actual attention computation is then
done in flash_attn_ext_f16_process_tile.
```c++
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / WARP_SIZE;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
                     nr_q_heads/nr_kv_heads
                             14/2
```
```console
(cuda-gdb) p ncols
$3 = 8
(cuda-gdb) p nbatch_fa
$4 = 128
(cuda-gdb) p nthreads
$5 = 128
(cuda-gdb) p nwarps
$6 = 4
(cuda-gdb) p gqa_ratio
$7 = 7
```

```c++
            const int iter_k = (ne11   + (nbatch_fa - 1)) / nbatch_fa;
```
This is the number of block we need to split K into.
```console
(cuda-gdb) p iter_k
$2 = 2
```
One thing to keep in mind is that the thread that the break point triggered on
might not have any work to do, so setting a breakpoint in the while loop migth
be a better option.

```c++
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
        const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
        const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

        const int head0 = zt * ncols2;

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02* head0);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
```
So this is iterating over the whole K blocks/chunks. And we have another
call to flash_attn_ext_f16_process_tile which handles the leftovers which we
will look at later.
And we can see that Q_f2 and K_h2 are pointer to global memory.

_wip_

The following function is what processes on tile of Q. So it handles one block
of Q and then iterates over all blocks of K and V to compute the attention for
that block of Q.
```c++
template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool mla, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale, // kq_scale from the operation tensor param
        const float slope,
        const float logit_softcap, // from the operation tenors param
        const uint3 ne01,
        const int ne02,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
        ...
```
First, `__force_inline__` is a directive in CUDA that is significantly stronger
than `__inline__` in C/C++. This instructs the nvcc to inline this function
into the calling code no matter what (unless it's recursive).
In GPU kernels, Register Pressure is critical. Function calls involve
pushing/popping the stack and moving data, which consumes registers and
instructions.


The shared memory allocation is done like this
```c++
    extern __shared__ half2 tile_Q[];
```
Next we define pointers to the other tiles in shared memory:
```c++
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
    half2 * tile_V    =           nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);
```

```c++
    // Load Q data into tile_Q, either temporarily (in a register) or permanently
    // (left in shared memory).
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.

    // This is the kq_scale from the operation parameters. This will be used
    // further down to scale the Query vectors as they are loaded from global
    // memory into shared memory. In the attention forumula we have:
    // Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
    // The scaling is the, / sqrt(d_k). Instead for first doing the matrix multiplication
    // (Q *K^T) and then dividing every element of the resulting matrix by sqrt(d_k),
    // we can scale the Q matrix first which is mathematically equivalent and more efficient.
    // Doing (Q / sqrt(d_k)) * K^T gives the same result as (Q * K^T) / sqrt(d_k).
    const half2 scale_h2 = make_half2(scale, scale);
    // So the scale is a float which is 32-bit precision. half/fp16 is 16-bit half-precision.
    // half2 is a special CUDA data type that packs two half-precision floating-point
    // Now, scale like we said comes from kq_scale and this is actually already the square root
    // this was done before calling build_attn:
    //      cur = build_attn(inp_attn,
    //              model.layers[il].wo, model.layers[il].bo,
    //              Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
    // So we calculate the reciprocal of the square root of the embedding dimension per head
    // before we call build_attn. This is then passed as kq_scale to the flash attention.
    // And recall we do this because dot products grow with the dimension and can
    // become very large. This can cause problems for softmax and also for the
    // vanishing/exploding gradient problem. So we want to preserve the relative
    // information between the dot prod scores, but scaled down.
    // The scale is duplicated in both halves of the half2 because this allows
    // us to later perform scale_h2 * make_half2(tmp.x, tmp.y) which will scale
    // tmp.x with the first half of scale_h2 and tmp.y with the second half of scale_h2
    // but this can be performed as one single operation/instruction.

    // range-based loop of using initializer list {32, 16, 8}
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start  = stride_k == WARP_SIZE ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            // uint3 is a built-in CUDA type that holds three unsigned integers
            // (x, y, z). Here .z is the number tokens in the sequence.
            if (jt*ncols1 + j < int(ne01.z)) {
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    // Q_f2 is in global memory and was passed in as a parameter.
                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    // float2 is a built-in CUDA type that holds two single-precision, so two 32-bit
                    // and is therefore 64-bits in total. Think of is a s struct like:
                    // struct float2 {
                    //     float x;
                    //     float y;
                    // };
                    // This where the scaling of Q actually happens, before we store
                    // it in shared memory.
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    // this is padding for out of bounds Q
                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    // sync the block so that we know that the shared memory loads are done
    __syncthreads();
```

```console
(cuda-gdb) p ne01
$8 = {x = 613566757, y = 3, z = 7}
```

Next we handle the case where the Query tile is stored in registers:
```c++
    // If the compiler has determined that Q can fit in registers do so using ldmatrix.
    // So it does not have to be re-read from shared memory later.
    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
            //               register        shared memory address
        }
    }

    __syncthreads();
```
In normal CUDA if we want to move data from shared memory into registers we can
just use normal assignment (=), but for tensor cores we need something else.
The hardware reads the rectangular tile from Shared Memory and automatically
distributes the correct bytes to the correct threads' registers to form a valid
Tensor Core Fragment.

Global to Shared: Load Q from VRAM to Shared Memory (done in previous block).
Barrier 1: Wait for load to finish.
Shared to Registers: Use ldmatrix to move Q into private storage (the block above).
Barrier 2: Wait for everyone to finish reading Q so we can recycle the Shared Memory.
Recycle: Use the now-empty Shared Memory to load K.
