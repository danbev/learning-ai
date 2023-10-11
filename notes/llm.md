## Large Language Models
Are language models with many parameters, like over 100M, and are pre-trained
on large collections of text (corpra).

This page tries to create an overview of the various large language models and
which companies/orginizations are behind them.

### OpenAI
* GPT-2
* GPT-3
* GPT-4

### Google
* BERT
* T5
* PaLM
* PaLM2

### Facebook
* Llama
* Llama2 (Almost opensource I think)
* RoBERTa
* XLM-R
* XLM-RoBERTa

### EleutherAI
* GPT-Neo
* GPT-NeoX
* GPT-J

### Hugging Face/BigScience
* Bloom

### MosaicML
[MosaicML](https://www.mosaicml.com/) offers models that are completely free
to use even for commercial purposes. 
* MPT-7B (MosaicML Pretrained Transformer) ChatGPT style decoder only.

### Technology Innovation Institute (TII)
* Falcon opensource

## AI Chat applications
* ChatGTP (from OpenAI)
* Bard (from Microsoft)


### Using an OpenSource LLM
The examples that can be found in this repo have been using OpenAI which means
that it has a cost associated with it. So I've been looking into using an
opensource LLM instead. Above we have a couple of alternatives but one thing
to keep in mind is that while these can be run standalone it may not be very
performant to do so. Running one locally would require a GPU or the performance
of it would be very poor. There are cloud providers that offer the ability to
run these models in the cloud and then you can use the API to interact with
like [Replicate](https://replicate.com/) but they also cost money which is what
I'm trying to avoid in this case.

I'm going to investigate using llama.cpp as the inference engine and see if I
can get a chat ui to run against it, all locally.


### LLM parameters

#### Context size
An LLM has a context size which specifies how much context the LLM can handle.
So what does that mean? Well, lets say we want to generate a sentence, we would
pass the start of the sentence to the LLM and it would generate the next word.
For example:
```
Somewhere over the...
```
We can imaging the context that an LLM as like this:
```
[                   Context                              ]    [Next word prediction]
+----+   +----+    +----+    +---------+  +----+    +----+    +----+
|    |   |    |    |    |    |Somewhere|  |over|    |the |    |    |
+----+   +----+    +----+    +---------+  +----+    +----+    +----+
```
The LLM will use the words (not really words but tokens but we'll use words
for simplicity) in the context to predict the next word.

```
[                   Context                              ]    [Next word prediction]
+----+   +----+    +----+    +---------+  +----+    +----+    +-------+
|    |   |    |    |    |    |Somewhere|  |over|    |the |    |rainbow|
+----+   +----+    +----+    +---------+  +----+    +----+    +-------+
```

```
[                   Context                                 ]    [Next word prediction]
+----+   +----+    +---------+  +----+    +----+    +-------+    +-----+
|    |   |    |    |Somewhere|  |over|    |the |    |rainbow|    |Way  |
+----+   +----+    +---------+  +----+    +----+    +-------+    +-----+

+----+   +---------+  +----+    +----+    +-------+  +-----+     +----+
|    |   |Somewhere|  |over|    |the |    |rainbow|  |Way  |     |up  |
+----+   +---------+  +----+    +----+    +-------+  +-----+     +----+

+---------+  +----+    +----+    +-------+  +-----+   +----+     +----+
|Somewhere|  |over|    |the |    |rainbow|  |Way  |   |up  |     |high|
+---------+  +----+    +----+    +-------+  +-----+   +----+     +----+

+----+    +----+    +-------+  +-----+   +----+   +----+         +---------+
|over|    |the |    |rainbow|  |Way  |   |up  |   |high|         |There's  |
+----+    +----+    +-------+  +-----+   +----+   +----+         +---------+
```
Notice that Somewhere is no longer in the context and hence not be used to
predict the next word.


#### Temperature
The temperature parameter is used to control the randomness of the output. The
higher the temperature the more random the output will be.

In the above section the LLM the predicted next word does not have to be the
most probable word. For example if we want the LLM to be more creative
we can tell it to not always choose the most probable word but instead choose
another one to create some thing different. This is what the temparature
parameters controls. So for question answering involving facts we would want the
LLM to be always choose the most probable word but for creative writing we
would want it to be more creative and thus have a highter temperature value.

#### Top_p
This is a hyperparameter that uses a technique called nucleus sampling. This
works like follows, we start by sorting the words in our very limited
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

#### Top_k
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
probable words for the next selection and then sample from the.

Unlike Nucleus Sampling (top_p), which considers a variable number of tokens
based on a probability threshold, top_k sampling always considers a fixed number
of tokens (the top k most probable ones).

The advantage of top_k is that it's straightforward and computationally
efficient. However, it may not capture as much diversity as top_p if the top few
tokens have significantly higher probabilities than the rest.

#### Repetition penalty
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

