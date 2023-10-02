## Modular Reasoning, Knowledge and Language (MRKL)
[Modular Reasoning, Knowledge and Language] (MRKL, pronouned as "mircle").

Although Language Models include Language in the name and hence we think about
text the Large Language Models (LLM) are not really about text per say. They
are multimodel in that they can take in any kind of input and produce any kind
of output. The input can be text, images, audio, video, etc. The output can be
text, images, audio, video, etc.

I liked the following quote from the paper:
```
While LMs indeed model syntax, and other linguistic elements, their most
striking feature is that they model the world, as described by the data on which
they were trained.
```
And the paper goes on to say that perhaps they, LLMs, should be called language
and knowledge models.

LLMs are trained with a corpus of data which has been gathered from different
resources. Events that happen after the training of the model are not part of
the models "knowledge". This can cause the LLM to hallucinate, which means it
makes stuff up that is not true, it calculating the probablitiy of what would
come next based on the data (weights) it was trained on.
It also does not have access to private data.

```
                 +----------------+
                 |  Input Text    |
                 +----------------+
                         |
                         |
                         ↓
                 +----------------+
                 |      LLM       |
                 +----------------+
                         |
                         ↓
                 +----------------+
                 | Export Router  |
                 +----------------+
                         |
                         ↓
        +-------+ +-------+ +-------+
        |Expert1| |Expert2| |Expert3| 
        +-------+ +-------+ +-------+ ...
                         |
                         |
                         ↓
                 +----------------+
                 |      LLM       |
                 +----------------+
                         |
                         |
                         ↓
                 +----------------+
                 |  Output Text   |
                 +----------------+

```
The `Export` modules/boxes can be thought of as the Tools in LangChain. These
experts module can be tools like calculator/math function, web search modules,
custom code modules, or "standard" LLM query modules or specialized LLMs.

An LLM is used to extract the correct arguments from the input text which are
then passed to the appropriate expert module. 

[modular reasoning, knowledge and language]: https://arxiv.org/pdf/2205.00445.pdf
