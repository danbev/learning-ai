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
block_size = 3
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
# The inputs 32x3 are the 32 words and the 3 characters that we are using to
# predict the next character.
print(f'{X.shape=}, {X.dtype=} (Inputs)')
print(f'{Y.shape=}, {Y.dtype=} (Labels)')
print(X)
print(Y)
# We are going to take the X's and predicts the Y values for those inputs.
C = torch.randn(27, 2) # 27 characters (rows), 2 dimensions embedding (columns)
print(f'{C=}')

# Manual embedding of the integer 5:
# We have 27 possible characters.
one_hot = F.one_hot(torch.tensor(5), num_classes=27).float()
print(f'{one_hot=}')
one_hot = one_hot @ C
# Notice that this matrix multiplication is like masking out the fifth row:
# [0.0] [ 0.1961, -0.7622]
# [0.0] [-1.1182,  0.6680]
# [0.0] [ 0.3447,  0.3149]
# [0.0] [-0.5860,  0.7838]
# [0.0] [ 1.0652,  1.4711]
# [1.0] [-0.8589, -0.8324]
# [0.0] [ 0.6763,  0.6351]
# [0.0] [-0.8141, -0.2554]
# [0.0] [-0.1614, -0.5273]
# [0.0] [-0.4246,  0.2074] = [-0.8589, -0.8324]
# [0.0] [-0.7300, -0.6255]
# [0.0] [-0.2981,  0.3134]
# [0.0] [-0.2918, -0.4503]
# [0.0] [-0.8250,  0.8351]
# [0.0] [ 0.8814, -0.2109]
# [0.0] [-0.2941,  0.4049]
# [0.0] [ 0.6671,  0.4552]
# [0.0] [-0.0369,  1.2453]
# [0.0] [-2.3761, -1.0131]
# [0.0] [ 1.2863,  0.5841]
# [0.0] [-0.7500, -0.2501]
# [0.0] [-0.0798, -0.2308]
# [0.0] [-0.3383, -0.8850]
# [0.0] [ 0.3755, -0.7705]
# [0.0] [-1.1456,  0.3123]
# [0.0] [-0.7670, -1.9977]
# [0.0] [ 0.0691,  1.4713]
# So the following two are identical
print(f'{C[5]=}')
print(f'{one_hot=}')
# So that was an example of how to embedd a single character but we want to
# embedd X which is a tensor of 32x3.
print(C[[1, 4, 6]])
print(C[torch.tensor([1, 4, 6])])
print(f'{C[X]=}')
print(f'{C[X].shape=}') # 32 rows, each with 3 characters, with two embeddings.
emb = C[X]

W1 = torch.randn(6, 100)
print(f'{W1.shape=}')
