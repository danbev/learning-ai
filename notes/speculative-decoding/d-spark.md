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

What D-Spark does is it adds a sequential layer after the parallel block.
_wip_
