## Llama.cpp Agent
This is an exploration into AI Agents for llama.cpp. The goals is to gain some
hands-on experience with AI agents and understand how they can be built.

### Overview

The idea is to enable agent to work with llama.cpp and be able to run the
locally. The tools that an Agent uses will be defined using the Web Assembly
Component Model and an interface defined in Web Assembly Interface Types (WIT).

The motivation for choosing this is that using the Web Assembly Component Model,
we can define an interface for the tools that an agent uses and then implement
the tool in any language that supports Web Assembly Component Model. This
includes Rust, Python, JavaScript.

Another motivation for using this is that using WASM we can run the agent in a
sandboxed environment and hence the agent can be trusted to run on the
user's machine or perhaps in a server environment. Perhaps this could enable
agents to be deployed in server environments but in a safe way for companies
that want to offer this feature as a service.

So the idea is that a tools interface be defined in WIT and then tool
implementors would use wit-bindgen (or similar tools) to generate the interface
in their language of choice. These would then be compiled to WASM and the agent
would would load and use the WASM tools need to accomplish its tasks.

## Tools
Tools are what agents use to accomplish their tasks. These tools are defined
using the Web Interface Types (WIT) and implemented as Web Assembly Component
Models.

_this is very much a work in progress and exploration at this point_

### Building Tools
The following shows an example of building the Echo tool which just
returns/echos the input it recieves:
```console
$ make echo-tool
```
This will produce a `tools/echo/target/wasm32-wasip1/debug/echo_tool.wasm` which
is a normal wasm (not a web assembly component model module that is).

Then we create a component model module from the wasm file:
```console
$ make echo-component
```
This will produce `components/echo-tool-component.wasm` which is a web assembly
component model module.

This can then be used with the `tool-runner` which is really just for testing
the component standalone:
```console
$ make run-echo-tool
cd tool-runner && cargo build
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.08s
cd tools/echo && cargo build --target wasm32-wasip1
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.01s
wasm-tools component new tools/echo/target/wasm32-wasip1/debug/echo_tool.wasm \
    --adapt wit-lib/wasi_snapshot_preview1.reactor.wasm \
    -o components/echo-tool-component.wasm
cd tool-runner && cargo run -- -c ../components/echo-tool-component.wasm --value "Hello"
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.07s
     Running `target/debug/tool-runner -c ../components/echo-tool-component.wasm --value Hello`
Component path: "../components/echo-tool-component.wasm"
Tool metadata:
  Name: Echo
  Description: Echos the passed in value
  Version: 0.1.0
  Parameters:
    - value: Value to be echoed (string)

Executing tool...
[Success] Tool output: Hello
```
There is also a print tool but this was mainly to make sure that wasi is working
and that it is possible to print to the console from the wasm module.

### Agent

### Running the agent
```console
$ make run-agent
```

#### Download model
There is no particular model that is needed for this agent, however is needs
to be an instruction trained model.
```console
$ make download-phi-mini-instruct 
```

### Setup/Configuration
```console
$ rustup target add wasm32-wasip1
```

```console
$ cargo install wasm-tools
```

```console
$ cargo install wac-cli
```
