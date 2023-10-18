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


### LLM hyperparameters
These are configuration options for the llm that influences how inference is
done. This section will list the ones that I've come accross and what they do.

But to fully understand where these come into play it might be worth backing
up a little and think about how inference works. Lets say we have the following
query "What is LoRA?". This string is passed to the llm and the first thing that
will happen is that it will be first be tokenized and then mapped (encoded) into
an integer id according to the vocabulary that the llm was trained on.
So if we run llama.cpp with the following command:
```
./main -m models/llama-2-13b-chat.Q4_0.gguf --prompt "What is LoRA?</s>"
```
What is the vocabulary that the llm was trained on? Well, that is defined in
the model file and once the model has been loaded we can inspect it:
```console
(gdb) br
Breakpoint 2 at 0x4e6b64: file common/common.cpp, line 813.

(gdb) p model->name
$28 = "LLaMA v2"

(gdb) p model->vocab->token_to_id
$25 = std::unordered_map with 32000 elements = {["왕"] = 31996, ["还"] = 31994, 
  ["ὀ"] = 31990, ["역"] = 31987, ["ɯ"] = 31983, ["합"] = 31980, ["才"] = 31979, 
  ["명"] = 31976, ["遠"] = 31971, ["故"] = 31969, ["丁"] = 31968, ["ญ"] = 31967, 
  ["음"] = 31966, ["ษ"] = 31964, ["败"] = 31955, ["茶"] = 31954, ["洲"] = 31952, 
  ["월"] = 31950, ["ḏ"] = 31946, ["방"] = 31945, ["效"] = 31944, ["导"] = 31943, 
  ["중"] = 31941, ["내"] = 31940, ["ほ"] = 31938, ["Ġ"] = 31937, ["瀬"] = 31933, 
  ["助"] = 31931, ["ˇ"] = 31929, ["現"] = 31928, ["居"] = 31924, ["პ"] = 31919, 
  ["ව"] = 31918, ["客"] = 31915, ["ശ"] = 31913, ["昭"] = 31912, ["员"] = 31911, 
  ["反"] = 31908, ["과"] = 31906, ["操"] = 31904, ["米"] = 31902, ["构"] = 31901, 
  ["ྱ"] = 31896, ["식"] = 31895, ["运"] = 31894, ["种"] = 31893, ["ҡ"] = 31892, 
  ["̍"] = 31891, ["ɵ"] = 31890, ["ദ"] = 31889, ["貴"] = 31887, ["達"] = 31883, 
  ["‭"] = 31881, ["頭"] = 31876, ["孝"] = 31875, ["ự"] = 31874, ["Ÿ"] = 31872, 
  ["論"] = 31871, ["Ħ"] = 31870, ["红"] = 31869, ["庄"] = 31868, ["ὺ"] = 31866, 
  ["ো"] = 31864, ["ರ"] = 31861, ["ਿ"] = 31857, ["터"] = 31856, ["测"] = 31851, 
  ["溪"] = 31850, ["ក"] = 31849, ["麻"] = 31846, ["希"] = 31841, ["❯"] = 31840, 
  ["望"] = 31839, ["非"] = 31838, ["索"] = 31836, ["确"] = 31835, ["む"] = 31834, 
  ["ந"] = 31833, ["ϊ"] = 31832, ["塔"] = 31831, ["ც"] = 31828, ["Ξ"] = 31827, 
  ["만"] = 31826, ["학"] = 31822, ["样"] = 31819, ["զ"] = 31816, ["衛"] = 31815, 
  ["尔"] = 31814, ["話"] = 31812, ["എ"] = 31808, ["ӏ"] = 31806, ["ḷ"] = 31802, 
  ["ြ"] = 31801, ["ܝ"] = 31800, ["达"] = 31798, ["ณ"] = 31796, ["˜"] = 31793, 
  ["개"] = 31789, ["隆"] = 31788, ["変"] = 31786...}

(gdb) p model->vocab->id_to_token[31996]
$61 = {text = "왕", score = -31737, type = LLAMA_TOKEN_TYPE_NORMAL}
```
Notice that the vocabulary has the size of 32000. This means that the llm can
handle 32000 different tokens.

The context has a logits with a reserved capacity of 32000:
```console
(gdb) p ctx->logits
$73 = std::vector of length 0, capacity 32000
```

#### Context size
An LLM has a context size which specifies how much context the LLM can handle.
So what does that mean? Well, lets say we want to generate a sentence, we would
pass the start of the sentence to the LLM and it would generate the next word.
For example:
```
"Somewhere over the..."
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

#### Locally Typical Sampling
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


#### Mirostat
This is introduced in the paper: https://openreview.net/pdf?id=W1G1JZEIy5_ and
is also related sampling of generated tokens, so in the same "category" as
top_p and top_k I think. It is about controlling perplexity, which is a measure
of how well a language model predicts a sample of text.

It sounds like MIROSTAT also uses top_k sampling but unlike traditional top-k
sampling, MIROSTAT adaptively adjusts the value of k to control the perplexity.
So there is still a top_k sampling performed but the value of k is not fixed but
instead is adjusted to control the perplexity.


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


#### Frequency penalty
This is a hyperparameter that is simlar to the repetition penalty but instead
controls the likelihood of generating tokens that are inherently frequent in the
language model's training data, whereas repetition penalty is about the the
tokens in the generated text.

The following is applied to each token in the vocabulary:
```
modified probability = original probabilty^frequency_penalty
```
Frequency Penalty is applied universally to all tokens, while Repetition Penalty
is applied only to tokens that have already appeared in the generated text.

Frequency Penalty is useful when you want to encourage or discourage the use of
common words. Repetition Penalty is useful when you want to avoid repetitive
text in longer sequences.

##### Presence penalty
This hyperparameter influences the models output and controlls the presence or
absence of new token in the generated text.

If this value i 0 then there is no effect at all for this hyperparameter.
If this value is > 0 the model i discouraged from introducing new tokens that
are not strongly supported by the context (the model is more conservative).
If this value is < 0 the model is encouraged to introduce new tokens, eveni if
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

#### Perplexity
This is a metric that is used to evaluate language models. It is a measure of
how well a probability model predicts a sample. It may be used to compare
probability models.

A low perplexity indicates the probability distribution is good at predicting
the sample.
