## GGML Exploration Project
The sole purpose of this directory is to explore the GGML C++ library and have
an example that I can use for exploration and understanding.

### Configuration
This project uses a git submodule to include the GGML library. To initialize the
submodule run:
```console
$ git submodule init
```

To update the submodule run:
```console
$ make update-ggml
```

### Building
```console
$ make tensor
```

### Running
The LD_LIBRARY_PATH environment variable must be set to the location of the
GGML library. For example:
```console
$ export LD_LIBRARY_PATH=ggml/build/src
```

```console
$ ./tensor
GGML tensor example
ctx mem size: 16777216
ctx mem used: 0
x tensor type: f32
x tensor backend: 0 
x tensor dimensions: 1
x tensor data: 0x7f0b48022190
x tensor operation: NONE, none
x tensor grad: (nil)
x tensor src: 0x7f0b480220d8
x tensor name: 
x tensor is_param: 0
updated tensor data: 18.000000
updated tensor name: updated
matrix ne[0]: 3
matrix ne[1]: 2
matrix nb[0]: 4
matrix nb[1]: 12
matrix name: 
```

### Graph example
The graph example show a very basic example of creating a computation graph
with two one dimensional tensors and adding them together. The graph for this
looks like this:

![image](./add.dot.png)

In the image above we can see two leafs named "a" and "b", and a node named
"c".
"a" and "b" are constants with the values 3 and 2 respectively. If we look at
the center row of "a"'s output we see:
```
CONST 0 [1, 1]
```
So this is a constant and 0 is the index of this leaf in the graph. The `[1, 1]`
is `ne`, number of elements array where nb[0] is the number of bytes to move to
get to the next element in a row. 


### Multidiminensional tensor example
When reading about matrix multiplication we often seen it in the format that
we create a matrix like 3x2 which means 3 rows and 2 columns. When working
with GGML, and I think this is common with graphics libraries in general, that
one first specifies the x-axis, that is the horizontal axis/number of columns.
If we have multiple dimensions then we have another value that specifies the
size of the y-axis, that is the vertical axis/number of rows. So think of this
as building a matrix from the bottom up and specifyig one dimension at a time.

So if we want to create a 3x2 matrix in GGML we do the following:
```c
  struct ggml_tensor* matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
```
This is because we are first specifying the number of elements in the first
dimension (x-axis) and then the number of elements in the second dimension
(y-axis).

Which can be visualized like this:
```
    +---+---+
    | 0 | 1 |
    +---+---+
    | 2 | 3 |
    +---+---+
    | 4 | 5 |
    +---+---+ 

ne[0] = 2 
ne[1] = 3

nb[0] = 4 (size of each element, moving this number will move to the next column)
nb[1] = 8 (stride, moving this number will move to the next row)

Memory layout:
0000 0001    0010 0011    0100 0101
  0    1      2     3       4    5
    row 1      row 2        row 3
             ↑
             8 (ne[1])
```  

An example of this can be found in [matrix-mul.c](./src/matrix-mul.c) and
the compute graphs looks like this:

![image](./mul.dot.png)
