## WebAssembly System Interface for Neural Networks (wasi-nn)

### WebAssembly Interface Types (wit)
The types and function are defined in [wasi-nn.wit]. 

A basic Tensor type is defined as
```wit
interface tensor {
    type tensor-dimensions = list<u32>;
```
So recall that at a tensor could be a single list of number also called a
vector in which case the tensor-dimensions be a list of length 1. If we have a
matrix then it would be a list of length 2, and so on.

The tensor stores it data in the tensor-data type which is defined as:
```wit
type tensor-data = list<u8>;
```
And the length of this list depends on the type of the tensor which can be one
of:
```wit
enum tensor-type {
        FP16,
        FP32,
        FP64,
        BF16,
        U8,
        I32,
        I64
    }
```
So a 2x2 matrix of FP16 would have a tensor-data of length 8 (2x2 + 2x2).
To create a Tensor we use the construtor function:
```wit
   resource tensor {
        constructor(dimensions: tensor-dimensions, ty: tensor-type, data: tensor-data);

        dimensions: func() -> tensor-dimensions;

        ty: func() -> tensor-type;

        data: func() -> tensor-data;
    }
```
A `resource` in the Web Assembly Component model can be thought of as an object
which encapsulates some state and provides methods to interact with that state.
But in WASM these object/resources are designed to work efficiently within and
accross the boundries of WASM modules and the host environment. This allows a
tensor to be created in the wasm module and passed to the host environment for
example.


There is also a data structure named `graph`:
```wit 
interface graph {
    use tensor.{tensor};
    use errors.{error};
    use inference.{graph-execution-context};
```
Notice that graph is using/importing the tensor type. Likewise it is using
the errors.error type which is defined later in the wasi-nn.wit.
A graph can be used to initialize an execution context:
```wit
    resource graph {
        init-execution-context: func() -> result<graph-execution-context, error>;
    }
```
The `result` type is a way to handle errors in the wasm module and is very
simliar to how Rust handles errors.
The encoding of the graph (or model might be a better term) can be one of the
following:
```wit
 enum graph-encoding {
        openvino,
        onnx,
        tensorflow,
        pytorch,
        tensorflowlite,
        autodetect,
    }
```
Notice that `GGML` is not listed here :(. 
We can also specify where the model will be run:
```wit
enum execution-target {
        cpu,
        gpu,
        tpu
    }
```
Next we have a `graph-builder`:
```wit
     type graph-builder = list<u8>;
```
So this is just a list of bytes. So lets see how this is uesd to understand this
better:
```wit
load: func(builder: list<graph-builder>,
           encoding: graph-encoding,
           target: execution-target) -> result<graph, error>;
```
So we can pass an list of bytes to the load function and it will return a graph
object. So the actual loading would be done by a specific backend inference
engine.
It is also possible to load a graph using a name:
```wit
load-by-name: func(name: string) -> result<graph, error>;
```
Exactly how this would work is up to each backend impl I'm guessing.

Next we have the `inference` type:
```wit
interface inference {
    use errors.{error};
    use tensor.{tensor, tensor-data};

    resource graph-execution-context {
        set-input: func(name: string, tensor: tensor) -> result<_, error>;

        compute: func() -> result<_, error>;

        get-output: func(name: string) -> result<tensor, error>;
    }
```
One would first call `set-input`, followed by `compute` which runs the
inference, and then `get-output` to get the result.
Recall that the graph-execution-context is created by calling
init-execution-context.

Finally we have the errors:
```wit
interface errors {
    enum error-code {
        invalid-argument,
        invalid-encoding,
        timeout,
        runtime-error,
        unsupported-operation,
        too-large,
        not-found,
        security,
        unknown
    }

    resource error {
        constructor(code: error-code, data: string);
        code: func() -> error-code;
        /// Errors can propagated with backend specific status through a string value.
        data: func() -> string;
    }
}
```

[wasi-nn.wit]: https://github.com/WebAssembly/wasi-nn/blob/main/wit/wasi-nn.wit
