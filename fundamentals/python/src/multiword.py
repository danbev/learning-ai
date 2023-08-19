import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('src/names.txt', 'r').read().splitlines()
print(f'words: {words[:8]}')
print(f'Number of words: {len(words)}')

chars = sorted(list(set(''.join(words))))
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}
#print(f'itos: {itos}')

# Build the dataset
# Block size is the context length, as in how many characters to use to predict
# the next character.
block_size = 103
print(f'block_size: {block_size}')
X = []
Y = []
for word in words[:5]:
    context = [0] * block_size
    print(f'{word=}, context={context}')
    for ch in word + '.':
        ix = stoi[ch] # map character to integer
        # X will represents the input for example 3 tensors that will represent
        # the word for which we want to predict the 4th character
        X.append(context) 
        Y.append(ix) # the labels
        print(''.join([itos[i] for i in context]), '---->', itos[ix])
        # context[1:] is a slicing operation which creates a new list which will
        # contain all the elements of the constext list except the first one.
        # [1, 2, 3, 4][1:] -> [2, 3, 4]
        # The +[ix] will join a new list with the integer ix to the end of the
        # context list.
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)
print(X)
print(Y)
