## Retrieval Augmented Generation (RAG)
The principal behing RAG is to use external data sources to provide an LLM with
context.

For example, say we want to generate a summary of some information that was not
available at the time the LLM model we are using was trained. What we can do is
take these documents and create embeddings for them and then use those
embeddings can be stored in a vector store. This is the retrieval part of RAG,
sometimes refered to as the retriever (model).

We can then take the query/prompt that is intended for our LLM and perform a
search against the vector store for the closest documents. These documents can
then be passed as context, included in the request, for our LLM. The LLM is
responsible for the generation part of RAG.
