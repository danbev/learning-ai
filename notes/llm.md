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

#### Zephyr-7B
Is a fine-tuned version of Mistral 7B and has similar performance to Chat Llama
70B but with a smaller size (7B vs 70B).

TODO: explain the finetuning process that uses Direct Preference Optimization).

### Mistral AI
* Mistral 7B (https://mistral.ai/product/)

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
will happen is that it will be first be tokenized and then mapped (indexed) into
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



#### Perplexity
This is a metric that is used to evaluate language models. It is a measure of
how well a probability model predicts a sample. It may be used to compare
probability models.

A low perplexity indicates the probability distribution is good at predicting
the sample.

### Cost
This section will try to give an approximate cost of training LLMs and the
option of renting vs buying GPUs.

```
NVidia A100: about $1-$2 per hour
 10b Model: $150K to train
100b Model: $1.5M to train
```

### Feed Forward Neural Network
This is a type of neural network where the data flows in one direction only,
from input nodes, through the hidden nodes (if any) and to the output nodes.
They typically have input layers, hidden layers, and output layers.

### Speculative Decoding
Lets say we have the following sequence of tokens (but shown as words here for
clarity):
```
 ["Somewhere", "over", "the", "?", "?", "?", "?"]
```
The LLM will generate the next word based on the context, and when it has
generated the next token, then will do the same process again:
```
 ["Somewhere", "over", "the", "rainbow, "?", "?", "?"]
```
This is a serial process and can be slow. The generation of the next token is
called "decoding" and the LLM will decode the next token based on the context.
A GPU is very good at performing tasks in parallel but in this case we are
stuck with a serial process. So how can we make this process faster? Well, we
can use speculative decoding.

The ideas is that we, or the LLM rather, guesses some of the future tokens.
```
 ["Somewhere", "over", "the", "?", "?", "?", "?"]
```
Lets say we guess "rainbow". Now that gives us:
```
 ["Somewhere", "over", "the", "rainbow"]

```
And this would be like decoding/predicting the next token that comes after
"rainbow" even though we don't know what that is yet.
So, we would then take our actual sequence and the one with the guess and
decode/predict them in parallel:
```
 ["Somewhere", "over", "the"]
 ["Somewhere", "over", "the", "rainbow"]
```

```
 ["Somewhere", "over", "the"] -> "rainbow"
 ["Somewhere", "over", "the", "rainbow"] -> "Way"
```
Now, since we have predicted the original sequence we can compare the token
that the model predicted with our guess, and if they are the same we guessed
correctly and we also know then that the prediction of our guess was correct
as well, "Way" above. Our new sequence will then be:
```
 ["Somewhere", "over", "the", "rainbow", "Way"]
```
So instead of producing a single token, we have produced two tokens in parallel.

If our guess in incorrect:
```
 ["Somewhere", "over", "the"]
 ["Somewhere", "over", "the", "desert"]
```
The predicted output will be:
```
 ["Somewhere", "over", "the"] -> "rainbow"
 ["Somewhere", "over", "the", "desert"] -> "far"
```
Now, if we compare the actual predition of our original sequence we can see that
our guess was incorrect, so we can't use the second token, but we still have the
next token (simliar to if we had only decoded the original sequence). And then
this continues with guessing.

The above is the gist of it, we try to guess `n` future tokens and then decode
them in parallel.

That sounds good, but how do we guess?  
The guessed token needs to come from the vocabulary of the LLM. If we take llama
as an example it has 32000 tokens in its vocabulary. So just guessing would
mean that we would have to guess from 32000 tokens. 1/32000 = 0.00003125 chance
of guessing correctly. So we need to do better than that and would like
somewhere like 50/50 chance of guessing correctly. 

There are various ways to do this, one is to use n-grams of the prompt/context
so far to guess the next token. Another way is to use look ahead where we just
have blank tokens and get the model to predict the blanks (I'm not 100% sure
how this works). But LLMs are pretty good at predicting the next token so why
not use one to predict the next token? This is called a helper model or a
draft model. So we use a small LLM to predict the next token and then use that
as our guess for the larger LLM model that we are using.
In llama.cpp the main application has the following options for the draft model:

```console
$ ./llama.cpp/llama-cli -h | grep draft
  --draft N             number of tokens to draft for speculative decoding (default: 8)
  -ngld N, --n-gpu-layers-draft N
                        number of layers to store in VRAM for the draft model
  -md FNAME, --model-draft FNAME
                        draft model for speculative decoding
```

### Visualization
https://bbycroft.net/llm
