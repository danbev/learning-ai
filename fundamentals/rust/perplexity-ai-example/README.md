## Perplixity AI Example
This is an example of using llm-chain to interact with the Perplexity AI API.
The API that Perplexity provides is compatible with OpenAI's client API so
you can use the same client library to interact with it which in this case is
the Rust llm-chain-openai crate.

### Setup
To try this example out you will need to have a Perplexity API key. This
currently requires a paid account. You can get one by signing up at
You can get one by signing up at [perplexity.io](https://perplexity.ai) and
then generating an API key.

### Running the example
Set the environment variable `PERPLEXITY_API_KEY` to your API key, and specify
a different base URL for the API (the default is OpenAI's URL):
```bash
$ export OPENAI_API_KEY="YOUR_PERPLEXITY_API_KEY_HERE"
$ export OPENAI_API_BASE_URL=https://api.perplexity.ai
```
The OPENAI_API_BASE_URL required a change in llm-chain-openai that I've
currently made locally. I will submit a [PR] to the original repo to include
this change.

Then run the example:
```bash
$ cargo r -q
Query: Can you give me a summary of RHSA-2020:5566?

Perplixity AI:
Assistant: RHSA-2020:5566 is a security advisory for Red Hat Enterprise Linux 7, which addresses a vulnerability in the openssl package. The vulnerability, identified as CVE-2020-1971, is a NULL pointer dereference flaw in openssl. A remote attacker could potentially exploit this vulnerability to cause a denial of service, impacting the system's availability. The issue can be mitigated by updating the affected packages.
```

[PR]: https://github.com/sobelio/llm-chain/pull/267
