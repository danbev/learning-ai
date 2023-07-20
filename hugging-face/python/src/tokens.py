import pandas as pd

# Tokens section
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
# A very simple way of tokenizing a string:
print(f'char tokens: {tokenized_text}')
# But for these tokens to be used in a model they need to be number/integers
# so we need to convert them to integers.
token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print(f'token2idx: {token2idx}')

input_ids = [token2idx[token] for token in tokenized_text]
print(f'Tokens: {input_ids}')

print(f"Index of \'T\': {token2idx['T']}")


categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]})
print(categorical_df)

import torch
import torch.nn.functional as F

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
print(one_hot_encodings)
print(one_hot_encodings.shape)

print(f"Token: {tokenized_text[0]}")
print(f"Tensor index: {input_ids[0]}")
print(f"One-hot: {one_hot_encodings[0]}")

# Word tokens examples
tokenized_text = text.split()
print(tokenized_text)
# Notice that the last element of the list contains a period. 

# Wordpiece tokens examples which is used by BERT and DistilBERT tokenizers.
from transformers import AutoTokenizer
# From chatgpt 4:
"""
In the context of Hugging Face models, a checkpoint typically refers to the
saved state of a trained model. It includes both the model architecture and the
learned parameters (or weights).

While training large models, it is common practice to periodically save
checkpoints. If the training process gets interrupted for any reason, you can
resume from the last saved checkpoint, rather than starting from scratch. This
can save a lot of computational resources and time.

In Hugging Face's Model Hub, a "checkpoint" often refers to a specific model
that has been pre-trained and shared publicly. For example, "bert-base-uncased"
and "gpt-2" are names of checkpoints. You can load these checkpoints directly
using the from_pretrained method, which allows you to leverage the power of
pre-trained models for your specific tasks with additional fine-tuning.
"""
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
encoded_text = tokenizer(text)
print(encoded_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens))
print(f'vocab size: {tokenizer.vocab_size}')
# print max length of tokenizer
print(f'max length: {tokenizer.model_max_length}')
# print input names of tokenizer
print(f'input names: {tokenizer.model_input_names}')
