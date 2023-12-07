## WASM for AI/ML
The trained models are often deployed to various devices and using wasm makes
them portable. The usage of WASI is required as ML usually requires special
hardware like GPUs and TPUs (also AVX and SSE for CPUs) and WASI allows the ML
code to access that hardware.

wasi-nn is focused on inference and not training, at least at this stage.

### WebAssembly Component for LLM
This is just an idea at this stage, but the motivation is to be able to run LLM
locally on a server, on the command line, embedded as a plug-in in users code,
or in a browser, in a secure manner.

The idea is to create a wasm component that contains one component that is the
inference engine, another component that holds/loads the model, and a third
component could be the end users component which would optionally have a
configuration function to configure the inference engine.

The end user would compose, which is a command/process to combine multiple
component, it's component using the other components. So in theory we could
provide different inference engines, different models and the user can choose
the ones most suitable to their task.

This would be somewhat similar to what we did for the trustification policy
engine [static executor](https://github.com/seedwing-io/seedwing-policy/pull/237)
and we could get some inspiration/code from:
```
"The idea here is that we can take a policy, perhaps created in the
playground, and include it in a static component. This component
will import the policy engine and call the eval function in the policy
engine component module. This way a user can have a self contained
.wasm which can be executed by any wasm runtime that supports the
webassembly component model."
```
This would allow an end user to create a wasm component that imports the
inference engine, a model, and then calls the inference engine with some
input data and get the output.

The inference engine could be based on llama.cpp and we could use the bindings
from llm-chain-llama or use the llm-chain-llama crate. We might also look at
at using wasi-nn.

Doing this would allow for running this component on any wasm runtime that has
support for the web assembly component model. So it could work in the browser,
on the server, on the edge, or embedded into an application as a sort of
plugin.

I'd like to investigate this further to see exactly what is out there in regards
to this, like how the existing wasm example work and what they are using.
Perhaps taking advantage of the web assembly component model is not something
that has been done yet and might be a good idea to explore.
