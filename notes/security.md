## Language Model Security

There are various security concerns when using language models and how they are
used today. Things like prompt injection, poisoning of training data, DoS
attacks, supply chain attacks (like using thirdparty datasets or pre-trained
models, and more.

### Model Collapse
This is something that occurs when more and more content on the internet is
genereated by AI, and then new models are trained this new content. The new
models become too dependent on patterns that are present in the generated data,
and they are replicating patterns they have already seen.
So probable events are overestimated and improbable events are underestimated
and as this happens over generations of models this problem amplifies.

### Resources
The repo as a list of security related resources:
https://github.com/chawins/llm-sp
