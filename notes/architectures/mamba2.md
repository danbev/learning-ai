## Mamba 2
The issue with Mamba 1 was that it was still not as performant as transformers
and this was mostly due the selective scan could not be made efficient on 
tensor cores. The selective scan algorithm in Mamba-1 had inherent sequential
dependencies:
```
h[t] = A * h[t-1] + B[t] * x[t]  // Depends on previous state
y[t] = C[t] * h[t]               // Then compute output
```
Now, I was under the impression that a selective scan operation could be
optimized efficiently on a GPU, it is even in the book I've got "Programming
Massively Parallel Processors". But the book discusses GPU operations on general
GPU cores that process FP32 arithmetic. The tensor cores that are used for
transformers are optimized for matrix multiplications and convolutions and can
use 312 TFLOPS of FP16, whereas general GPU cores can only do 19 TFLOPS of FP32.
Both are parallel algorithms, but matrix multiplication can use the specialized,
ultra-fast tensor cores, while parallel scan cannot.

Mamba-1 was also limited to a small state dimension which is typically N=16
(d_state).
```
State update: h[t] = A·h[t-1] + B[t]·x[t]
              ↑        ↑
            [N×N]    [N×1]

Output: y[t] = C[t]·h[t]
               ↑     ↑
             [1×N] [N×1]
```
With N = 16 we have around 256 operation per time step.
With N = 64 we have around 4096 operation per time step.
With N =128 we have around 16384 operation per time step.

N=16 wasn't a hard limit in Mamba-1 and you can technically use N=64 in Mamba-1
but:
* It would be much slower (no tensor cores)
* The SSM layer would dominate computation time
* Other layers waiting for the SSM would be underutilized
* Overall throughput would suffer

So N=16 was chosen as the sweet spot where:
* State has enough capacity for most tasks
* SSM layer doesn't become the bottleneck
* GPU utilization stays reasonably balanced


Mamba-2 restructures the computation so it can be expressed as:
```
Y = M · X    // Where M is a structured matrix
```

This "duality" means the same operation can be computed either:
* Linear form   : Sequential recurrence (like SSM)
* Quadratic form: Matrix multiplication (like attention)


Mamba-2 introduces heads (like attention heads)

The SSD algorithm is significantly faster than the selective scan algorithm from
Mamba-1 for the same state dimension, and scales much better computationally to
larger state dimensions.

So what does this actually mean?
```
X = [x₀, x₁, x₂, x₃] = [1, 2, 3, 4]

Parameters that are selective (vary per token):
B = [b₀, b₁, b₂, b₃] = [0.1, 0.2, 0.15, 0.25] (input dependent)
C = [c₀, c₁, c₂, c₃] = [1.0, 0.8, 1.2, 0.9]   (input dependent)
A = 0.9 (simplified, this is usually a matrix)
```

Mamba-1:
```python
# Initialize state
h = 0  # hidden state (scalar for simplicity)

# Process sequentially - MUST do one at a time
for t in range(4):
    # Update state (depends on previous state!)
    h = A * h + B[t] * X[t]

    # Compute output
    Y[t] = C[t] * h
```
Step by step:
```
0 t=0: h₀ = 0.9 * 0     + 0.1  * 1 = 0.1    Y₀ = 1.0 * 0.1   = 0.10
1 t=1: h₁ = 0.9 * 0.1   + 0.2  * 2 = 0.49   Y₁ = 0.8 * 0.49  = 0.392
2 t=2: h₂ = 0.9 * 0.49  + 0.15 * 3 = 0.891  Y₂ = 1.2 * 0.891 = 1.069
3 t=3: h₃ = 0.9 * 0.891 + 0.25 * 4 = 1.802  Y₃ = 0.9 * 1.802 = 1.622
```

Mamba-2:
If we expand the state updates from above we can see that they are:
```
h₀ = B₀X₀                               // initial state h = 0
h₁ = A  · B₀X₀ + B₁X₁
h₂ = A² · B₀X₀ + A  · B₁X₁ + B₂X₂
h₃ = A³ · B₀X₀ + A² · B₁X₁ + A · B₂X₂ + B₃X₃
```
And the outputs are then:
```
Y₀ = C₀ · h₀ = C₀B₀X₀
Y₁ = C₁ · h₁ = C₁(A·B₀X₀ + B₁X₁)
Y₂ = C₂ · h₂ = C₂(A²·B₀X₀ + A·B₁X₁ + B₂X₂)
Y₃ = C₃ · h₃ = C₃(A³·B₀X₀ + A²·B₁X₁ + A·B₂X₂ + B₃X₃)
```

We can rearrange this into a matrix multiplication:
```
Y = M · X
```
where M is:
```
     X₀        X₁         X₂          X₃
  ┌                                     ┐
Y₀│ C₀B₀         0         0          0 │
Y₁│ C₁AB₀      C₁B₁        0          0 │
Y₂│ C₂A²B₀    C₂AB₁     C₂B₂          0 │
Y₃│ C₃A³B₀   C₃A²B₁    C₃AB₂       C₃B₃ │
  └                                     ┘
```
This is called a lower triangular semiseparable matrix. Notice that this is like
a masked attention matrix where tokens can only attend to previous tokens.

Filling in the numbers with our example:
```
M =
  ┌                                              ┐
  │  0.10      0         0          0            │
  │  0.072     0.16      0          0            │
  │  0.097     0.194     0.18       0            │
  │  0.117     0.261     0.203      0.225        │
  └                                              ┘
```
This then becomes:
```
Y = M · X =
  ┌                                              ┐  ┌   ┐   ┌       ┐
  │  0.10      0         0          0            │  │ 1 │   │ 0.10  │
  │  0.072     0.16      0          0            │  │ 2 │ = │ 0.392 │
  │  0.097     0.194     0.18       0            │  │ 3 │   │ 1.069 │
  │  0.117     0.261     0.203      0.225        │  │ 4 │   │ 1.622 │
  └                                              ┘  └   ┘   └       ┘
```
Notice that we get the same result as Mamba-1, but now we can compute it all
in parallel using matrix multiplication which is much more efficient on GPUs.

In the above example we showed A as a scalar for simplicity, but in reality A
is a matrix, a diagonal matrix.

In Mamba-1 A would be a matrix with different values:
have the same scalar values.
```
# State dimension N = 4 (just for example)
A = ┌                    ┐
    │ 0.9   0    0    0  │
    │ 0    0.85  0    0  │
    │ 0     0   0.92  0  │
    │ 0     0    0   0.88│
    └                    ┘

h = [h₀, h₁, h₂, h₃]  # N-dimensional state vector
```
But in Mamba-2 we can use the same A for all heads, so we can use a single
scalar A for all heads:
```
# Number of heads H = 4
A = ┌                   ┐
    │ 0.9  0    0    0  │
    │ 0   0.9   0    0  │
    │ 0    0   0.9   0  │
    │ 0    0    0   0.9 │
    └                   ┘

h = [h₀, h₁, h₂, h₃]  # N-dimensional state vector
```

The meaning of the values in A is that these values control how much each
state dimension remembers or forgets over time.

```
A = diag([0.9, 0.85, 0.92, 0.88])  # N=4

# At each time step:
h[0] = 0.9  * h[0] + B[t,0] * x[t]  # Dimension 0: slow decay
h[1] = 0.85 * h[1] + B[t,1] * x[t]  # Dimension 1: faster decay
h[2] = 0.92 * h[2] + B[t,2] * x[t]  # Dimension 2: very slow decay
h[3] = 0.88 * h[3] + B[t,3] * x[t]  # Dimension 3: faster decay
```
* Dimension 0 (a=0.9) : Retains 90% of previous value → "medium-term memory"
* Dimension 1 (a=0.85): Retains 85% of previous value → "shorter memory"
* Dimension 2 (a=0.92): Retains 92% of previous value → "longer memory"
* Dimension 3 (a=0.88): Retains 88% of previous value → "short memory"

With Mamba-2 we can use the same A value for all dimensions:
```
a = 0.9  # Single scalar value
A = 0.9 * I = diag([0.9, 0.9, 0.9, 0.9])  # All same!

# At each time step:
h[0] = 0.9 * h[0] + B[t,0] * x[t]  # All dimensions
h[1] = 0.9 * h[1] + B[t,1] * x[t]  # decay at the
h[2] = 0.9 * h[2] + B[t,2] * x[t]  # same rate
h[3] = 0.9 * h[3] + B[t,3] * x[t]  # (0.9)
```
When all dimensions share the same A value, they all decay at the same rate, but
these also means that we can factor it out.
And keep in mind that A is the matrix that determines how much of each state
get merged into the hidden state. In this example we have a inner state dimension
of 4, so each h[i] represent how much of the current token x[t] should we store
(merge) into the h[0] state and so on.
We can think of h as having N=4 memory slots:
```
h[0] = slot 0: stores some learned aspect of the history
h[1] = slot 1: stores another learned aspect
h[2] = slot 2: stores yet another aspect
h[3] = slot 3: stores another aspect
```

```
# Initialize: all memory slots empty
h = [0, 0, 0, 0]

# Token 1: "The"
x[0] = 1.0

B[0] = [0.8, 0.3, 0.5, 0.2]  # How much to store in each slot
# The model learned: "The" should go strongly into slot 0 and 2

h[0] = 0.9 * 0   + 0.8 * 1.0 = 0.8    # Slot 0: stores 0.8 from "The"
h[1] = 0.85 * 0  + 0.3 * 1.0 = 0.3    # Slot 1: stores 0.3 from "The"
h[2] = 0.92 * 0  + 0.5 * 1.0 = 0.5    # Slot 2: stores 0.5 from "The"
h[3] = 0.88 * 0  + 0.2 * 1.0 = 0.2    # Slot 3: stores 0.2 from "The"

# Token 2: "cat"
x[1] = 2.0

B[1] = [0.6, 0.9, 0.2, 0.4]  # How much "cat" goes into each slot
# The model learned: "cat" (subject) should go strongly into slot 1

h[0] = 0.9  * 0.8  + 0.6 * 2.0 = 0.72 + 1.2 = 1.92   # Some "The", more "cat"
h[1] = 0.85 * 0.3  + 0.9 * 2.0 = 0.26 + 1.8 = 2.06   # Little "The", lots "cat"
h[2] = 0.92 * 0.5  + 0.2 * 2.0 = 0.46 + 0.4 = 0.86   # Mostly "The", little "cat"
h[3] = 0.88 * 0.2  + 0.4 * 2.0 = 0.18 + 0.8 = 0.98   # Little "The", some "cat"

# Token 3: "sat"
x[2] = 3.0

B[2] = [0.2, 0.7, 0.3, 0.8]  # How much "sat" goes into each slot

h[0] = 0.9  * 1.92 + 0.2 * 3.0 = 1.73 + 0.6 = 2.33   # Fading memory
h[1] = 0.85 * 2.06 + 0.7 * 3.0 = 1.75 + 2.1 = 3.85   # "cat" and "sat"
h[2] = 0.92 * 0.86 + 0.3 * 3.0 = 0.79 + 0.9 = 1.69   # Long memory of "The"
h[3] = 0.88 * 0.98 + 0.8 * 3.0 = 0.86 + 2.4 = 3.26   # Mostly recent "sat"
```
So with Mamba-1 we can have different decay rates for each dimension, but in Mamba-2
we have the same decay rate for all dimensions. But this is per layer so it
different layer can have different decay rates and with multiple layers say 64
layers this gives 64 different timescales accross the model.

### repeat vs repeat_interleaved issue
The grouping in Mamba is done precisely to take advantage of GPU parallelization
and circumvent the memory and computational constraints associated with
processing the entire hidden dimension $D$ within a single, coherent SSM block.

By splitting D into G independent groups (4096\8 = 512), the large operation is
broken into G smaller, completely independent operations. These G operations can
be executed simultaneously across the thousands of cores of a GPU.

So the hidden dimension D might be 4096, and lets say we have 8 groups so that
gives us 512 dimensions per group. 
Now, B and C are not 4096 dimensional tensors. B for example is responsible for
mixing the current input x with the state update h_t-1.

```
D       = 4096         // Full hidden dimension
N       = 64           // State dimension per group
G       = 8            // Number of groups
D_group = D / G = 512  // Dimension/features per group
N_total = 512          // 64 x 8
```

And the input x is usually 4096 dimensional vector.

X is:
```console
x = [x₀, x₁, x₅₁₁, x₅₁₂, x₁₀₂₃, ...,  x₄₀₉₅]  // 4096-dim input vector
      group 1        group 2        group 8
```

The GPU executes 8 separate, parallel State Space Model (SSM) recurrences, one
for each group. Each of these groups only has access to a 512 chuck of the x
input vector.

The purpose of the B matrix is to project the x vector (4096-dim, D) into the
state dimension which is 64 dimensions (N). 

For an example of a real model:
```console
(gdb) p B->ne
$3 = {128, 8, 1, 1}
```
So that is 8 groups, and the 128 represents the state dimension and the internal
projection factor, in this case 2 * 64. This projection is done often at the
same time as the z/gate, B and C, and dt are projected together in a single
operation which results in a tensor. This tensor is the split into the different
matrices.

So after the extraction B will have the following shape:
```console
shape: [128, 8] (ggml format)
128 = N = 64 * num_heads (2 heads)
8   = D / group_size (4096 / 512)

  0 [0................127]
  1 [0................127]
  ...
  7 [0................127]
```
So this B matrix is storing weights for each of the 8 groups, and each group
has 128 dimensions because there are 2 heads. The hidden dimension is 4096 and
the state dimension is 64.

The B matrix is then exapanded by repeating it for each head.
```python
B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
             ↑  ↑              ↑                    ↑
         batch  |       group/head dim            state dim
              seq

B = B.repeat(1, 1, 2, 1)
Input B shape:  [batch, seq_len,  8, 128]
Output B shape: [batch, seq_len, 16, 128]
```
So we have the following layout for B before repeating:
```console
Group 0: +------+------+-- ... --+------+
         |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 1  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 2  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 3  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 4  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 5  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 6  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 7  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
```
And after repeating we have:
```console
Group 0: +------+------+-- ... --+------+
         |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 1  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 2  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 3  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 4  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 5  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 6  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 7  |  W1  |  W2  |   ...   |  W8  |   (Each W is 128-dim)
         +------+------+-- ... --+------+
Group 8  |  W1  |  W2  |   ...   |  W8  |   (Same data as Group 1)
         +------+------+-- ... --+------+
Group 9  |  W1  |  W2  |   ...   |  W8  |   (Same data as Group 2)
         +------+------+-- ... --+------+
Group 10 |  W1  |  W2  |   ...   |  W8  |   (Same data as Group 3)
         +------+------+-- ... --+------+
Group 11 |  W1  |  W2  |   ...   |  W8  |   (Same data as Group 4)
         +------+------+-- ... --+------+
Group 12 |  W1  |  W2  |   ...   |  W8  |   (Same data as Group 5)
         +------+------+-- ... --+------+
Group 13 |  W1  |  W2  |   ...   |  W8  |   (Same data as Group 6)
         +------+------+-- ... --+------+
Group 14 |  W1  |  W2  |   ...   |  W8  |   (Same data as Group 7)
         +------+------+-- ... --+------+
Group 15 |  W1  |  W2  |   ...   |  W8  |   (Same data as Group 8)
         +------+------+-- ... --+------+
```

Now, if we had use repeat_interleaved we would get:
```console
Group 0: +------+------+-- ... --+------+
         |  W1  |  W2  |   ...   |  W8  |
         +------+------+-- ... --+------+
Group 8  |  W1  |  W2  |   ...   |  W8  | (Same data as Group 0)
         +------+------+-- ... --+------+
Group 1  |  W1  |  W2  |   ...   |  W8  |
         +------+------+-- ... --+------+
Group 9  |  W1  |  W2  |   ...   |  W8  | (Same data as Group 1)
         +------+------+-- ... --+------+
Group 2  |  W1  |  W2  |   ...   |  W8  |
         +------+------+-- ... --+------+
Group 10 |  W1  |  W2  |   ...   |  W8  | (Same data as Group 2)
         +------+------+-- ... --+------+
Group 3  |  W1  |  W2  |   ...   |  W8  |
         +------+------+-- ... --+------+
Group 11 |  W1  |  W2  |   ...   |  W8  | (Same data as Group 3)
         +------+------+-- ... --+------+
Group 4  |  W1  |  W2  |   ...   |  W8  |
         +------+------+-- ... --+------+
Group 12 |  W1  |  W2  |   ...   |  W8  | (Same data as Group 4)
         +------+------+-- ... --+------+
Group 5  |  W1  |  W2  |   ...   |  W8  |
         +------+------+-- ... --+------+
Group 13 |  W1  |  W2  |   ...   |  W8  | (Same data as Group 5)
         +------+------+-- ... --+------+
Group 6  |  W1  |  W2  |   ...   |  W8  | 
         +------+------+-- ... --+------+
Group 14 |  W1  |  W2  |   ...   |  W8  | (Same data as Group 6)
         +------+------+-- ... --+------+
Group 7  |  W1  |  W2  |   ...   |  W8  |
         +------+------+-- ... --+------+
Group 15 |  W1  |  W2  |   ...   |  W8  | (Same data as Group 7)
         +------+------+-- ... --+------+
```
The repeated values would be interleaved and this can cause issues. The model
was trained on one specific format, either repeated or interleaved and if get
this part wrong the computation will still work but the model will not works as
expected the values it is using could be the incorrect values for the groups.
