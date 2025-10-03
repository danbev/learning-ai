## Low-Level Guidance (llguidance)
This is somewhat similar to grammars in llama.cpp but is more general. Both
operate prior to the sampling process and provide a way to modify the logits of
the LLM (the logits of the models vocabulary that have been predicted). The
grammars/llguidance then have the chance to modify the logits before the sampling
process takes place, effectivly being able to restrict/filter what the samplers
can choose from.

Github: https://github.com/guidance-ai/llguidance

### Background
When we use an LLM we sometimes want the output to follow specific rules or 
have a specific structure (like JSON as an example).

First there needs to be some way of defining the rules and for this a context-free
grammar is used. This is what defines the structure of the output.

The components involved in this process are:
* The context-free grammar (CFG) that defines the structure of the output.
* The tokenizer that the LLM uses.
* Token prefix which is the sequence of tokens already generated.
* Token mask which is a set of allowed next tokens that will keep the output valid.

This is something that happens before sampling and the purpose is to limit the
the tokens that the sampling process can choose from.
So llguidance looks at the the current prefix, the tokens generated thus far. 

1) The LLM generates logits for ALL tokens in its vocabulary for the next position
2) llguidance computes a mask of which tokens are grammatically valid in this position
3) The mask is applied to the logits, setting the probability of invalid tokens
to negative infinity.
4) The modified logits are then passed to the sampling process.


