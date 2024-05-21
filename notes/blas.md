## Basic Linear Algebra Subprograms (BLAS)
Is both a standard and a library for linear algebra operations. The standard
defines an API of operations like vector and matrix multiplication, vector
addition, and dot products.

The subprograms/subroutines are split into three categories:
* Level 1: Vector-vector operations (e.g., dot product)
* Level 2: Matrix-vector operations (e.g., matrix-vector multiplication)
* Level 3: Matrix-matrix operations (e.g., matrix-matrix multiplication)

### Implementations

#### CPU
* OpenBLAS
* Intel MKL (Intel's Math Kernel Library)
* ATLAS (Automatically Tuned Linear Algebra Software)

#### GPU
* cuBLAS (NVIDIA)
* clBLAS (OpenCL)
* rocBLAS (AMD)


### General Matrix Multiply (GEMM)
Is a common operation in BLAS and performs matrix multiplication. Now, I was
thinkig that this would be just `AB` instead it is defined as:
```
C = alpha * A * B + beta * C
```
Where `A`, `B`, the input matrices that we want to multiply and and `C` is the
resulting output matrix. `alpha` and `beta` are scalars and if we set them to
1 and 0 respectively, we get the standard matrix multiplication and initial
values of C are ignored.

But C might also be a non-zero matrix in which case beta will be applied before
the addition of the result of the multiplication.

Now, in the context of a neural network A might be the weights and B the
incoming activations. Normally alpha and beta would be set to 1 and 0, but if
there is a skip connection (residual connection) then beta would be set to 1
allowing the new activations to be added to the previous activations.


Example can be found in [gemm.c](../fundamentals/blas/openblas/src/gemm.c).

### saxpy
Single (float) precision A * X + Y. This is a Level 1 operation and is a
vector-vector operation. It is defined as:
```
Example can be found in [vector_add.c](../fundamentals/blas/openblas/src/vector_add.c).
