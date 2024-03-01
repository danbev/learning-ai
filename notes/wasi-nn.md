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
the errors.error type which is defined later in the wasi-nn.wit. And also the
infrence.graph-execution-context type.

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
So this is just a list of bytes and I think that it would be possible for this
to contain a path to a model file instead of the actual binary model data
itself.

So lets see how this is used to understand this better:
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
inference, and then `get-output` to get the result. This interface has recently
changed and set-input and get-output used to take integer indices instead of
names.

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


### wit-bindgen
The code that is generated from the wit file is done using the `wit-bindgen`
command and the output is a rust file named ml.rs (because the world is `ml` in
wasi-nn.wit).

Now, keep in mind that what is produced by wit-bindgen is the binding of types
to integrate with the host environment and other wasm modules.

For the `compute` function which is part of the Interface resource the following
will get generate (shortened a little for readability):
```rust
impl GraphExecutionContext {                                              
          pub fn compute(&self,) -> Result<(),Error>{                             
                                                                                  
            #[allow(unused_imports)]                                              
            use wit_bindgen::rt::{alloc, vec::Vec, string::String};               
            unsafe {                                                              
              #[repr(align(4))]                                                                             
              struct RetArea([u8; 8]);                                                                      
              let mut ret_area = ::core::mem::MaybeUninit::<RetArea>::uninit();                             
              let ptr0 = ret_area.as_mut_ptr() as i32;                                                      

              #[cfg(target_arch = "wasm32")]                                                                
              #[link(wasm_import_module = "wasi:nn/inference")]                   
              extern "C" {                                                        
                #[link_name = "[method]graph-execution-context.compute"]          
                fn wit_import(_: i32, _: i32, );                                  
              }                                                                                             
                                                                                                            
              #[cfg(not(target_arch = "wasm32"))]                                                           
              fn wit_import(_: i32, _: i32, ){ unreachable!() }                                             
              wit_import((self).handle() as i32, ptr0);                                                     
              let l1 = i32::from(*((ptr0 + 0) as *const u8));                                               
              match l1 {                                                                                    
                0 => {                                                            
                  let e = ();                                                     
                  Ok(e)                                                           
                }                                                                 
                1 => {                                                                               
                  let e = {                                                       
                    let l2 = *((ptr0 + 4) as *const i32);                         
                                                                                  
                    super::super::super::wasi::nn::errors::Error::from_handle(l2 as u32)
                  };                                                              
                  Err(e)                                                          
                }                                                                 
                _ => wit_bindgen::rt::invalid_enum_discriminant(),                
              }                                                                   
            }                                                                     
          } fn compute                                                            
        } impl GraphExecutionContext       
```
Notice that there is a function declared name `wit_import` which takes two i32
values. Now, I've been using wasmedge's wasi-nn crate and a branch which has
support for GGML. But this crate uses the older .witx types for wasi-nn and not
the newer which was the cause of the issue above. I'm trying to update these
and struggling a little understanding how this is done. I was confused about
what to implement in the wasmedge wasi-nn crate, was it the compute function
itself or just a wrapper around it providing a higher level interface. It is the
latter, so for the compute function it would take the Rust input types and
then call the compute funtion imported from src/generated.rs.
And also the current version that uses .witx is different in that one would have
so pass pointers into the generated functions. With the newer .wit files these
functions take objects as input and return objects as output. So will not be
possible to directly convert the code but it might be possible to simplify it
knowing this.


### wasmtime Llama.cpp backend
The are notes taken during and attempt to add a Llama.cpp backend to wasmtime.

The first thing to notes is that the wasn-nn.wit file that is used in
wasmtime is somewhat old, at least it is not the version used in the upstream
[wasi-nn repo]. The difference can be seen in a few places, for example instead
of resources in the newer version there are records in the older version.

The Tensor type does not have a constructor as in the newer version so we have
to create them using a normal struct syntax:
```console
let options_tensor = tensor::Tensor {                                      
    dimensions: vec![1_u32],                                               
    tensor_type: tensor::TensorType::U8,                                   
    data: options.to_string().as_bytes().to_vec(),                         
};
```

The graph-builder types are the same:
```wit
type graph-builder = list<u8>;
type graph-builder = list<u8>;
```
So this is just a list of bytes and this type if used a a parameter to the
`load` function:
```wit
load: func(builder: list<graph-builder>, encoding: graph-encoding, target: execution-target) -> result<graph, error>;
load: func(builder: list<graph-builder>, encoding: graph-encoding, target: execution-target) -> result<graph, error>;
```
Notice that builder above is a list of graph-builder. In the case of Llama.cpp
I think there will only be one element in the `builder` list and it will be a
path to a GGUF model file. Actually, there might be more than one, for example
the llava models usually have two model files. TODO: check this.

There is also a `load-by-name`:
```wit
load-by-name: func(name: string) -> result<graph, error>;
```
I think this is more intended for caching of models but migth be something that
the llama implementation could use as well.

If we take a look at the `inference` type we see that the
`graph-execution-context` looks like this in the older version:
```wit
interface inference {
  type graph-execution-context = u32;
  init-execution-context: func(graph: graph) -> result<graph-execution-context, error>;
  set-input: func(ctx: graph-execution-context, index: u32, tensor: tensor) -> result<_, error>;
  compute: func(ctx: graph-execution-context) -> result<_, error>;
  get-output: func(ctx: graph-execution-context, index: u32) -> result<tensor-data, error>;
}
```
And in the newer version it looks like this:
```wit
interface inference {
     resource graph-execution-context {
        set-input: func(name: string, tensor: tensor) -> result<_, error>;
        compute: func() -> result<_, error>;
        get-output: func(name: string) -> result<tensor, error>;
    }
}
```
So this is very different. I think the graph-execution-context in the older
version is a pointer since it is using u32 (unsigned 32 bit integer) which
matches the pointer size on WASM32.

So we will need to load the model from a file for which we can use:
```c++
LLAMA_API struct llama_model* llama_load_model_from_file(
    const char * path_model, struct llama_model_params params);
```
The llama_model_params can be populated using:
```c++
llama_model_params model_params = llama_model_default_params();
```

And the llama context can be created using:
```c++
LLAMA_API struct llama_context* llama_new_context_with_model(                 
   struct llama_model * model, struct llama_context_params params);
```
And the llama_context_params can be populated using:
```c++
llama_context_params ctx_params = llama_context_default_params();
```

How do we set these configuration parameters for the model and context?  
I think the set-input function will have to do this by using tensors.
```rust
let options = json!({                                                      
              "stream-stdout": true,                                                 
              "enable-log": true,                                                    
              "ctx-size": 1024,                                                      
              "n-predict": 512,                                                      
              "n-gpu-layers": 25                                                     
          });                                                                        
          println!("Options: {}", options);                                          
                                                                                     
          let options_tensor = tensor::Tensor {                                      
              dimensions: vec![1_u32],                                               
              tensor_type: tensor::TensorType::U8,                                   
              data: options.to_string().as_bytes().to_vec(),                         
          };                                                                         
          inference::set_input(context, 1, &options_tensor).unwrap();
```
Now, set_input will take a tensor and the data it contains a string so we should
be able to parse that into json and the extract configuration values.

So perhaps using `index 1` for model configuration parameters. And the following
index would then be the prompt as a tensor which would be tokenized by calling
llama_tokenize. The llama_model, and the llama_model_params would need to be
stored somewhere, as well as the tokenized prompt as it will be needed later in
the compute function.

On the wasmtime side we have the following structs and traits that need to be
implemented to add a new backend:

In src/lib.rs we have graph defined as:
```rust
 pub struct Graph(Arc<dyn backend::BackendGraph>);
```
So here we have struct name Graph with a single field which is an Atomic
Reference Counted (Arc)/smart pointer. This is a dynamic dispatch (dyn) so the
actual type of the backend will be determined at runtime.

The `backend::BackendGraph` is defined in the backend module and is a trait:
```rust
  pub trait BackendGraph: Send + Sync {
      fn init_execution_context(&self) -> Result<ExecutionContext, BackendError>;
  }
```
And the `ExecutionContext` is defined as:
```rust
pub struct ExecutionContext(Box<dyn backend::BackendExecutionContext>);
```
And the `backend::BackendExecutionContext` is defined as:
```rust
  pub trait BackendExecutionContext: Send + Sync {                                
      fn set_input(&mut self, index: u32, tensor: &Tensor) -> Result<(), BackendError>;
      fn compute(&mut self) -> Result<(), BackendError>;                          
      fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError>;
  } 
```
We also have a Backend struct:
```rust
pub struct Backend(Box<dyn backend::BackendInner>);
```
So we need to implement the following trait:
```rust
  /// A [Backend] contains the necessary state to load [Graph]s.                     
  pub trait BackendInner: Send + Sync {                                              
      fn encoding(&self) -> GraphEncoding;                                           
      fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError>;
      fn as_dir_loadable<'a>(&'a mut self) -> Option<&'a mut dyn BackendFromDir>;    
  }
```

[wasn-nn.wit]: https://github.com/WebAssembly/wasi-nn/blob/e2310b860db2ff1719c9d69816099b87e85fabdb/wit/wasi-nn.wit
[wasi-nn-repo]: https://raw.githubusercontent.com/WebAssembly/wasi-nn/main/wit/wasi-nn.wit


