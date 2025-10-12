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
