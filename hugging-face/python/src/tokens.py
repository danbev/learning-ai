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
