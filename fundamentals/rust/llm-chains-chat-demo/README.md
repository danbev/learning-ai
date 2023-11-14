## llm-chain chat demo
This example is intended to mimic the python chat demo in
[trust-chat.py](../../../langchain/src/trust-chat.py). This example differs in
that it does not use an in-memory vector database but instead an external Qdrant
database which needs to be running.

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

Currently, the progress made is that we can run the example and it will
retrieve data from the vector database related to the query "Can you show me a
summary of RHSA-2020:5566?", This will then be used as the context for a new
request to the LLM answer the query.

Then we can run the OpenAI example using:
```console
$ cargo r -q --bin openai
Query: Can you show me a summary of RHSA-2020:5566?
Result: Assistant: Sure! RHSA-2020:5566 is a Red Hat Security Advisory that provides a security update for the openssl package. The vulnerability addressed in this advisory is classified as "Important" and has a CVSS base score of 6.5. 

The update fixes a flaw in the OpenSSL Diffie-Hellman (DH) key exchange implementation. This flaw could allow an attacker to downgrade the security of the DH key exchange to the weakest commonly supported level, known as "export-grade." By doing so, the attacker could potentially intercept and decrypt TLS connections between vulnerable clients and servers.

It is recommended to install the updated openssl packages as soon as possible to mitigate this vulnerability. You can find further details and instructions on how to update your systems in the Red Hat Security Advisory at the following URL: [RHSA-2020:5566](https://access.redhat.com/errata/RHSA-2020:5566).
```

The is also an example that uses Llama:
```console
$ cargo r -q --bin llama
```
