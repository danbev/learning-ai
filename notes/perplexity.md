## Perplexity AI
This is a [service](https://www.perplexity.ai) that allows web search to be
integrated with an LLM.

This is pretty interesting as it provides the LLM with a way to access updated
information (information that is was not trained on), and use it to generate
responses, but it will `only` use the information that it has gathered and 
provides references to the sources. So it does not make things up (hallucinate).

### API Reference
The API that Perplexity provides is compatible with OpenAI's Client API so
you can any client library that is compatible with OpenAI's API.
For example there is an [example](../fundamentals/rust/perplexity-ai-example)
which uses the Rust llm-chain-openai crate to interact with the Perplexity AI.

* [API Reference](https://docs.perplexity.ai/reference/post_chat_completions)

While looking at the the API they provide I was not able to find any ability to
have get any of the resource references that the LLM used to generate the
response. This does not seem to be available in the REST API.
