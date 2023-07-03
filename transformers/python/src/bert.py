from transformers import BertModel, BertConfig, BertTokenizer
from bertviz import head_view
import torch
import pandas as pd
import numpy
from IPython.display import display

model_name = 'bert-base-uncased'

model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

print(f'Number of encoders: {len(model.encoder.layer)}')
print(f'First encoders: {model.encoder.layer[0]}')

config = BertConfig()
print(config)

text = "My friend told me about this class and I love it so far! She was right."
#text = "I love icecream"
print(f'text: {text}')
# embedding vector?
tokens = tokenizer.encode(text)
print(f'tokens: {tokens}')

# Changes the shape of tokens from (20,) to (1, 20)
inputs = torch.tensor(tokens).unsqueeze(0)
print(inputs)

attention = model(inputs, output_attentions=True)[2]
final_attention = attention[-1].mean(1)[0]
print(final_attention)

attention_df = pd.DataFrame(final_attention.detach()).applymap(float).round(3)
attention_df.columns = tokenizer.convert_ids_to_tokens(tokens)
attention_df.index = tokenizer.convert_ids_to_tokens(tokens)

import sys
numpy.set_printoptions(threshold=sys.maxsize)
print(attention_df.to_string())

tokens_as_list = tokenizer.convert_ids_to_tokens(inputs[0])
display(head_view(attention, tokens_as_list))
