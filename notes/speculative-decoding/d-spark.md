### D-Spark
This was part of Deepseek V4 series and is a speculative decoding implementation.

Paper:  
https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf

The speculative decoding uses a draft model. So this is not anything new, we have
seen multiple variants of speculative decoding with draft models in this directory
before. So what is different with d-spark?  

D-Spark is similar to [d-flash](./d-flash.md) in that it does a block of decoding
in parallel, compared to Eagle which does autoregressive decoding (sequential
one after the other).

The is an issue with D-Flash which is inherent to the parallel processing which
is when we do parallel decoding we are predicting the complete "sentence" in one
go. There is no way for any token to know about the token prior to it or after
it as they get generated at the same time. 
A token generated at position 3 has no idea what token was just generated at
position 2. This is called lack of "inter-token dependencies" and creates an issue
called 'multi-modal collision'. This has nothing to do with multi-model models
but it about the 'mode' or 'path' that the sequence of tokens could follow.
A multi-modal collision happens because the parallel drafter is evaluating all
of these possible paths simultaneously without coordinating between the token
positions.

Because the model predicts each position independently and averages out all the
possible past tokens rather than locking into a single specific path, it outputs
the incoherent, mixed-up combination "of problem" . The different valid paths
(modes) literally collide with one another. This results in what the paper calls
"inconsistent suffix combinations," which causes the quality of the draft tokens
to rapidly decay the further down the sequence you go (called suffix decay I
think).

### Parallel block and sequential block
So lets say we have an initial prompt, this will be processed by the target
model and it will produce predicted next token.
```
Target model:

prompt -> target model -> predicted next token (anchor token)

Parallel block:
                                                        -> base_logit_0, hidden_state_0
[anchor token + mask + mask + mask] -> parallel block   -> base_logit_1, hidden_state_1
                                                        -> base_logit_2, hidden_state_2
                                                        -> base_logit_3, hidden_state_3
Notice that the parallel block produces logits (raw scores for the whole language
vocabulary, and the hidden states (the last layer of the tranformer for each
token). The logits will be used in the sequential block, next in the diagram,
and the hidden states are used after the sequential block.


Sequential block:

anchor token
    ↓
sequential block  ← base_logit_0
    ↓
transitioned logit_0  ➔  [ Sample Draft Token 0 ]  (Saved for final output)
     +-----------------------------+
     ↓             
sequential block  ← base_logit_1
     ↓             
transitioned logit_1  ➔  [ Sample Draft Token 1 ]  (Saved for final output)
     +-----------------------------+
     ↓             
sequential block  ← base_logit_2
     ↓             
transitioned logit_2  ➔  [ Sample Draft Token 2 ]  (Saved for final output)
     +-----------------------------+
     ↓             
sequential block  ← base_logit_3
     ↓             
transitioned logit_3  ➔  [ Sample Draft Token 3 ]  (Saved for final output)

Output:
Tokens: [Draft Token 0, Draft Token 1, Draft Token 2, Draft Token 3]

[Draft Token 0, Draft Token 1, Draft Token 2, Draft Token 3]
     ↓               ↓             ↓            ↓
                    Confidence Head
     ↓               ↓             ↓            ↓
[Conf Score 0,  Conf Score 1,  Conf Score 2,  Conf Score 3 ]


Tokens: [Draft Token 0, Draft Token 1, Draft Token 2, Draft Token 3]
Scores: [Conf Score 0,  Conf Score 1,  Conf Score 2,  Conf Score 3 ]
```
What D-Spark does is it adds a sequential layer after the parallel block. So the
parallel block will output logits and also the hidden states for the predicted
tokens. And recall that the hidden states could just be the last layer of the
transformer, and the logits are the output of running the last hidden state
through the lm-head projection, going from the internal hidden space to the
token vocabulary score.

### Confidence head
It takes two pieces of information, the draft token and the conf score and
stitches them together into a single vector, multiplies that vector by its
learned matrix, and then passes the result through a sigmoid function to squash
the final number into a probability between 0 and 1
This matrix has learned how to accurately score the quality of its own guesses.
While this learned matrix is great at ranking which tokens are good and which
are bad, its absolute probability numbers are usually too high

### Transisition bias
So a transisition in probablitiy and statistics simply describes the likeihood
of moving from one specific state to another. An example of this is a Markov chain
where the next state depends only on the current state. In DSpark the default
sequential block is a Markov head, which looks at the token we just generated,
which is the anchor token for from the target model, or a logits vector from
the paralllel block. It calculates the probability of transistioning to the
next logical token.

Recall that the parallel block produces "base logits" (raw vocabulary scores)
independently, meaning those scores have no idea what the preceding tokens are.
The sequential stage supplements these independent base logits by mathematically
adding the transition bias directly to them.

Lets say the parallel block produces logits for "of course" and "no problem" and
that it give both "course", and "problem" a high score. And lets say that the
sequential block choose "of" for position 1. The Markov head will look up 'of'
in a small embedding table and generates a transistion bias vector, which is a
set of adjustment numbers, and adds it to the base logits. It literally adds a
positive number to the score for "course" (boosting it) and adds a negative
number to the score for "problem" (suppressing it.

_wip_
