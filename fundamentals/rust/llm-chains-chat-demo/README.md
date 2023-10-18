## llm-chain chat demo
This example is intended to mimic the python chat demo in
[](../../../langchain/src/trust-chat.py). This example differs in that it does
not use an in-memory vector database but instead an external Qdrant database
which needs to be running.

__work in progress__

### Running
Currently this example is using OpenAI's API and the following environment
variable needs to be set
```console
$ export OPENAI_API_KEY=<your key>
```
The Qdrant database needs to be started:
```console
$ cd ../../../vector-databases/qdrant/rust
$ make start-qdrant-server 
```
Then we can run the OpenAI example using:
```console
cargo r --bin openai
```
