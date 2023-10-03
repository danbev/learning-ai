## Prompt Engineering

### Zero-shot prompting
In this case we don't pass any examples to the model. We just pass the prompt
and the model will generate the rest of the text. This is the simplest way to
use a language model.

This might not always for well depending on the questions asked, and that is
why other methods discusssed below have been developed.


### Few-shot prompting
In this case we pass a few examples to the model. The examples are in the form
of a question and answer. The model will then generate the rest of the text
based on the examples.
This works well for questions and questions like classification but might not
work when asking questions that involve resoning.


### Chain-of-Thought (CoT) prompting
This is where we provide an example of a question and and an anwser wehere the
answer that we provide contains an example of how we would reason about the
question to solve it. 


### Self-Consistency prompting
This is similar to chain-of-thought prompting where we send multiple CoT
examples to the model. We then sample from the answers to select the most
consistent answer. So we would choose the answer that corresponds to the most
similar answers.

### Generate Knowledge prompting
This is where we provide a question and an answer in the format:
```
Input: ...
Knowledge:...
Input: ...
Knowledge:...
...
Input: <our actual target question goes here>
Knowledge: 
```
So we are getting the LLM to complete the knowledge part of the question format
us.

### Tree of Thoughs (ToT) prompting
This build upon an CoT which involves generating one solution at a time. Tree of
Thought Prompting enables the generation and evaluation of multiple solutions,
thereby navigating through the solution space more efficiently by pruning
non-effective paths and focusing on promising ones.

All problems are framed as a search of a tree where each node represents a
partial solution. The root is the input.
TODO: provide an example of a tree of thoughts.
