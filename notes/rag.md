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

The benefit of this approach is that we can use the LLM to operate on data that
it was not trained on, so it has no knowledge of it. And also training a model
is expensive and time consuming, using this model we can update the vector store
as needed and use the same LLM model.
This should also minimize the risk of the LLM hallucinating.

By also being able to provide the sources of the documents we can get a higher
degree of confidence in the generated text as we can check the references which
is not possible by just using an LLM. This is also useful for legal reasons.
