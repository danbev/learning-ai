from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Reacll that bert-base-uncased has 12 encoder layers.
model = BertModel.from_pretrained('bert-base-uncased')

named_parameters = list(model.named_parameters())
print(f'Bert base model has {len(named_parameters)} named parameters.')

def print_layer(layer):
    for (name, values) in layer:
        tup = tuple(values.size())
        print(name, str(tup)) # The first value is the number of tokens in this layer

print(f'Embedding layer:')
print_layer(named_parameters[0:5])
# The first value is the number of tokens in this layer
# So if we have (30522, 768) that means we have 30522 tokens in this layer
# and each token has a 768 dimensions.

print(f'\nEncoder layer 1:')
print_layer(named_parameters[5:21])

print(f'\nOutput layer 1:')
print_layer(named_parameters[-2:])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(f'Token.vocab_size: {tokenizer.vocab_size}')
if 'daniel' in tokenizer.vocab:
    print("daniel was in the vocabulary of Bert base uncased")
else:
    print("daniel was not in the vocabulary of Bert base uncased")

tokens = tokenizer.encode("Dan loves icecream")
print(f'tokens: {tokens}')

response = model(torch.tensor(tokens).unsqueeze(0))

print(f'\nOutput of Encoder 12:\n{response.last_hidden_state}')
print(response.pooler_output.shape)

#print(response.last_hidden_state[0][0])
#print(response.last_hidden_state[0][0].tolist())

print(model.pooler)

print(f'last_hidden_state:\n{response.last_hidden_state}')
print(f'last_hidden_state shape:{response.last_hidden_state.shape}')
cls_embedding = response.last_hidden_state[: ,0 ,:].unsqueeze(0)
# All of the first dimension, the first token, all of the dimensions last dim.
print(cls_embedding.shape)
# batch_size, nr_tokens which is cls token in this case, size of the last layer

pooled = model.pooler(cls_embedding)
print(pooled.shape)

          

