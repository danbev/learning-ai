## Sampling strategies

### Temperature
The temperature parameter is used to control the randomness of the output. The
higher the temperature the more random the output will be.

In the above section the LLM the predicted next word does not have to be the
most probable word. For example if we want the LLM to be more creative
we can tell it to not always choose the most probable word but instead choose
another one to create something different. This is what the temparature
parameters controls. So for question answering involving facts we would want the
LLM to be always choose the most probable word but for creative writing we
would want it to be more creative and thus have a highter temperature value.

### Top_p
This is a hyperparameter that uses a technique called nucleus sampling. This
works as follows, we start by sorting the words in our very limited
vocabulary:
```
"apple": 0.5
"banana": 0.2
"strawberry": 0.15
"watermelon": 0.1
"mango": 0.05
```
Next, we add these probablities until we reach, or exceed the specified top_p
value. For example if top_p = 0.8:
```
"apple": 0.5 + "banana": 0.2 + "strawberry": 0.15 = 0.85
```
This means our neucleus will then be:
```
["apple", "banana", "strawberry"]
```
And then a random sample is choses from that neucleus. So without this the
most probable word would always be chosen but with this we can get some
variation in the output.

### Top_k
This is a hyperparameter that uses a technique called top-k sampling.
Lets say we have the following vocabulary:
```
"apple": 0.4
"banana": 0.25
"cherry": 0.15
"date": 0.1
"elderberry": 0.05
"fig": 0.02
"grape": 0.015
"honeydew": 0.01
"kiwi": 0.005
"lemon": 0.005
```
And say we set top_k = 4. This means that we will only consider the 4 top most
probable words for the next selection and then sample from them.

Unlike Nucleus Sampling (top_p), which considers a variable number of tokens
based on a probability threshold, top_k sampling always considers a fixed number
of tokens (the top k most probable ones).

The advantage of top_k is that it's straightforward and computationally
efficient. However, it may not capture as much diversity as top_p if the top few
tokens have significantly higher probabilities than the rest.

### Locally Typical Sampling
This is introduced in the paper: https://arxiv.org/pdf/2202.00666.pdf.
Example, we have the following vocabulary:
```
"sunny"
"cloudy"
"rainy"
"windy"
```
The parameter of this in llm-chain is called `TypicalP`.

TODO: Explain this better


### Mirostat
This is introduced in the paper: https://openreview.net/pdf?id=W1G1JZEIy5_ and
is also related to sampling of generated tokens, so in the same "category" as
top_p and top_k I think. It is about controlling perplexity, which is a measure
of how well a language model predicts a sample of text (lower is better).

It sounds like MIROSTAT also uses top_k sampling but unlike traditional top-k
sampling, MIROSTAT adaptively adjusts the value of k to control the perplexity.
So there is still a top_k sampling performed but the value of k is not fixed but
instead is adjusted to control the perplexity.


### Repetition penalty
This is a hyperparameter that penalizes the probability of tokens that have
already been generated. This is useful for text generation tasks where we want
to avoid repetition.
The probability of each token is modified like this:
```
modified probability = original probabilty^repetition_penalty
```
Here the original probablity is the probability of the token given to the token
by the LLM. The repetition penalty is a value between 1 and infinity. A value
of 1 will not do anything as we in this case are just raising the probability
to the power of 1 which is just the original probability.

Now, a value greater than one will actually make the probability, a value
between 0-1, smaller which might seem counter intuitive but it makes sense if we
think about it:
```
0.20^1.2  = 0.16
0.20^-1.2 = 0.25
```

* A Repetition Penalty greater than 1 discourages repetition by reducing the
probabilities of already-appeared tokens.

* A Repetition Penalty less than 1 encourages repetition by increasing the
probabilities of already-appeared tokens.


### Frequency penalty
This is a hyperparameter that is simlar to the repetition penalty but instead
controls the likelihood of generating tokens that are inherently frequent in the
language model's training data, whereas repetition penalty is about the tokens
in the generated text.

The following is applied to each token in the vocabulary:
```
modified probability = original probabilty^frequency_penalty
```
Frequency Penalty is applied universally to all tokens, while Repetition Penalty
is applied only to tokens that have already appeared in the generated text.

Frequency Penalty is useful when you want to encourage or discourage the use of
common words. Repetition Penalty is useful when you want to avoid repetitive
text in longer sequences.

#### Presence penalty
This hyperparameter influences the models output and controls the presence or
absence of new token in the generated text.

If this value is 0 then there is no effect at all for this hyperparameter.
If this value is > 0, the model is discouraged from introducing new tokens that
are not strongly supported by the context (the model is more conservative).
If this value is < 0 the model is encouraged to introduce new tokens, even if
they are not strongly supported by the context (the model is more creative).

When the model is generating text, it computes a probability distribution over
the vocabulary for the next token to be generated, given the current context
(i.e., the tokens that have been generated so far and the initial prompt).

The PresencePenalty value is used to adjust these probabilities before the next
token is sampled. Specifically, it modifies the "logits" (unnormalized log
probabilities) associated with each token in the vocabulary. The adjustment can
either increase or decrease the likelihood of each token being selected as the
next token in the sequence, depending on the value of PresencePenalty

Somewhat simplified process:
* The model computes a probability distribution over the vocabulary for each
token in the vocabulary based on the current context.
* The PresencePenalty value is used to adjust these probabilities by adding or
subtracting the value from the logit (unnormalized log probabilities) associated
with each token.
* Adjusted/Rescale the logits so thy sum to 1 after possibly adding/subtracting
the PresencePenalty value.
* Sample the next token from the adjusted probability distribution. 
* Add the token to the current context and repeat the process.
