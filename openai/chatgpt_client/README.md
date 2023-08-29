## ChatGPT Example in Rust

### Building
This examples requires a OPENAI_API_KEY environment variable to be set as
an environment variable. This can be done in the terminal environment or as
as env command when running the example:

```console
$ export OPENAI_API_KEY=...
```

### Running
```console
$ env OPENAI_API_KEY=... cargo run
You: Explain positional encoding
Payload Payload { model: "text-davinci-003", prompt: "Explain positional encoding", max_tokens: 150 }
ChatGPT: Positional encoding is a method of translating the position of words in a sentence to numerical values. These values are then used to represent those words in a mathematical representation. The purpose of using this encoding is to create a mapping between the words in the sentence and the numerical representation, allowing the sentence to be interpreted by machines. For example, if a sentence has five words, each of them will be converted to a number, and those numbers will be used to represent the given words. The exact values used to convert words to numbers depend on the algorithm used. In general, a larger number indicates that the word is more important in the context of the sentence.
```

