import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

"""
This is the code for Building makemore Part 2: MLP:
https://www.youtube.com/watch?v=TCH_1BHY58I&t=10s

This is pretty much the same code but with my own comments and some minor
changes.
"""

words = open('src/names.txt', 'r').read().splitlines()
print(f'words: {words[:8]}')
print(f'Number of words: {len(words)}')

chars = sorted(list(set(''.join(words))))
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}
#print(f'itos: {itos}')

# Block size is the context length, as in how many characters to use to predict
# the next character.
block_size = 3
print(f'block_size: {block_size}')

X = [] # Inputs
Y = [] # Labels

# Build the dataset
i = 0
for word in words[:2]:
#for word in words:
    context = [0] * block_size
    print(f'{word=}, context={context}')
    for ch in word + '.':
        ix = stoi[ch] # map character to integer
        print(f'{i=}, {ch=}, {ix=}')
        i += 1
        # X will represents the input for example 3 tensors that will represent
        # the word for which we want to predict the 4th character
        X.append(context) 
        Y.append(ix) # the labels
        print(''.join([itos[i] for i in context]), '---->', itos[ix], '(', ix, ')')

        # context[1:] (below) is a slicing operation which creates a new list
        # which will contain all the elements of the constext list except the
        #first one.

        # [1, 2, 3, 4][1:] -> [2, 3, 4]
        # The +[ix] will join a new list with the integer ix to the end of the
        # context list.
        context = context[1:] + [ix]

print(f'Number of characters: {i}')
# For the first word 'emma.' we will have the following in X:
# [0, 0, 0] when the characters are '.', ',', ','
# [0, 0, 5] when the characters are ',', ',', 'e'
# [0, 5, 13] when the characters are ',', 'e', 'm'
# [5, 13, 13] when the characters are 'e', 'm', 'm'
# [13, 13, 1] when the characters are 'm', 'm', 'a'
# Remember that we are taking 3 characters as input to predict the forth char.
# So if we input [0, 0, 0] we want to predict the forth character which is 'e'
# becasue X[0] = [0, 0, 0] and Y[0] = 5 which is the index of 'e'.

# Notice that all of the names/words have an entry in X which is '0, 0, 0' and
# the corresponding Y is different most of the time as they start by different
# characters.


def build_dataset(words, block_size = 3):
    X = []
    Y = []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = stoi[ch] # map character to integer
            X.append(context) 
            Y.append(ix) # the labels
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(f'{X.shape=}, {Y.shape=}')
    return X, Y

X = torch.tensor(X)
Y = torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(len(words) * 0.8) # 80% of the words. Training data
n2 = int(len(words) * 0.9) # 10% of the word. Validation data?

Xtr, Ytr = build_dataset(words[:n1]) # Training data
Xdev, Ydev = build_dataset(words[n1:n2]) # Validation/Dev data
Yte, Yte = build_dataset(words[n2:]) # Test data

# The inputs 32x3 are the 32 characters, because initially we where using 5
# words 'emma.', 'olivia.', 'ava.', 'isabella.', 'sophia.' gives us 32
# characters.
# The 3 is the context length, as in how and the 3 characters that we are using
# to predict the next character.
print(f'{X.shape=}, {X.dtype=} (Inputs)')
print(f'{Y.shape=}, {Y.dtype=} (Labels)')
print(X)
print(Y)

# We are going to take the X's and predicts the Y values for those inputs.
C = torch.randn(27, 2) # 27 characters (rows), 2 dimensions embedding (columns)
print(f'{C=} which are initially random and the shape is {C.shape=}')

# Manual embedding of the integer 5.
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

# So we are going to take C which is just a randomly initialized 27x2 matrix
# 27 because that is the number of characters we have, and the 2 is the number
# of dimensions we are using for the embedding.

# Create the embedding which will have the shape of 32 x 3 x 2, which is the
# number of characters, again initially we are only using 5 words, and the
# 3 is the number of characters in the context, and we have 2 dimensions for
# the embedding.

emb = C[X]
breakpoint()
# The above code will loop through the rows in X and use the values to index
# the randomlly generated embeddings in C. Each row will be added to a new
# tensor, so the result will be a tensor with the same shape as X, but with
# the last dimension appended by the indexed embedding.

# X are the inputs, and C are the randomly initialized embeddings.
#(Pdb) p C
#tensor([[ 6.8443e-01, -9.1421e-01], # 0 '.'
#        [ 1.7745e+00, -3.1138e-01], # 1 'a'
#        [ 7.0778e-01, -2.1498e-01], # 2 'b'
#        [-1.6449e+00,  2.1587e-01], # 3 'c'
#        [-2.8619e-02, -4.1739e-01], # 4 'd'
#        [ 2.7164e-01,  2.2510e+00], # 5 'e'
#        [-8.2070e-01,  1.8946e+00], # 6 'f'
#        [-9.4465e-01, -3.8865e-01], # 7 'g'
#        [ 5.0636e-01, -9.5188e-02], # 8 'h'
#        [ 4.3576e-01, -1.4386e+00], # 9 'i'
#        [-7.6660e-01,  1.8655e+00], # 10 'j'
#        [-2.5704e+00,  1.2103e+00], # 11 'k'
#        [ 1.8644e-01,  2.2112e-01], # 12 'l'
#        [-5.0437e-01,  1.1397e+00], # 13 'm'
#        [-3.1989e-01, -8.0815e-01], # 14 'n'
#        [-9.6159e-01,  7.8360e-02], # 15 'o'
#        [ 3.0150e-01, -1.7784e+00], # 16 'p'
#        [ 1.9133e-01,  2.8774e-01], # 17 'q'
#        [ 1.9634e-01,  2.1779e+00], # 18 'r'
#        [ 1.9811e+00, -1.7951e-01], # 19 's'
#        [ 1.0016e+00, -1.1002e+00], # 20 't'
#        [-1.1199e+00, -1.6660e-01], # 21 'u'
#        [ 9.4509e-01,  1.9054e+00], # 22 'v'
#        [ 2.1910e-03, -9.8781e-01], # 23 'w'
#        [-4.4769e-01, -1.3415e+00], # 24 'x'
#        [ 1.6452e-01, -7.0964e-01], # 25 'y'
#        [ 6.4244e-01,  1.8194e+00]])# 26
#(Pdb) p C.shape
#torch.Size([27, 2])
#
#(Pdb) p X
#tensor([[ 0,  0,  0], # 0
#        [ 0,  0,  5], # 1
#        [ 0,  5, 13], # 2
#        [ 5, 13, 13], # 3 
#        [13, 13,  1], # 4
#        [ 0,  0,  0], # 5
#        [ 0,  0, 15], # 6
#        [ 0, 15, 12], # 7
#        [15, 12,  9], # 8
#        [12,  9, 22], # 9
#        [ 9, 22,  9], # 10
#        [22,  9,  1]])# 11
#(Pdb) p X.shape
#torch.Size([12, 3])
# tensor-indexing.py contains an example of this type of indexing.

# Next we are going to create the hidden layer and the weights and biases.
# Here we are saying that we want to have 100 neurons in the hidden layer.
# Each input is 3 characters with 2 dimension embeddings, so we have 6
# inputs.
# So to make this really clear emb will have 12, 3, 2
# Where each row has 3 characters, and each character has 2 dimensions. These
# two demensions are the floats that represent the characters
# For example, the first row is [0, 0, 0], which is the first word in the
#
# (Pdb) p emb[0]
# tensor([[ 0.8671, -0.2617],
#        [ 0.8671, -0.2617],
#        [ 0.8671, -0.2617]])
# So we can see that in this case 0.8671, -0.2617 is the embedding for '.'.
# Now, we just randomly generated these values but in a real model these
# would from an embedding. The embedding is just a numerical representation of
# a of a piece of information.

# +------------------------------------------+
# |                   W1                     |
# +------------------------------------------+
#      ↑               ↑              ↑
# +------------+ +------------+ +------------+
# |  0.1, 0.1  | |  0.1, 0.1  | |  0.1, 0.1  |  < inputs 3x2=6
# +------------+ +------------+ +------------+
#      0               1              2

W1 = torch.randn(6, 100) # the choice of 100 is arbitrary
print(f'{W1.shape=}')
b1 = torch.randn(100)

# This only works if we have a block_size of 3.
#print(emb[:, 0, :].shape)
#emb_cat = torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1)
#print(emb[:, 1, 0])
#print(emb_cat.shape)

emb_cat = torch.cat(torch.unbind(emb, 1), 1)
print(emb_cat.shape)
print(emb.view(emb.shape[0], 6) == emb_cat)

h = torch.tanh(emb.view(emb.shape[0], 6) @ W1 + b1)
#h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
print(h)
print(h.shape)

# The shape of emb is: torch.Size([32, 3, 2])
#print(emb)
#print(emb.shape)
#print(torch.unbind(emb, 1))

W2 = torch.randn(100, 27)
print(f'{W1.shape=}')
b2 = torch.randn(27)

logits = h @ W2 + b2
print(logits)
print(logits.shape)
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
print(prob.shape)
print(prob[0].sum())

# Recall that Y contains the actual letters and that prob are the probabilities
# that the nueural network predicted.
# (Pdb) p torch.arange(12)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
y_prop_list = prob[torch.arange(emb.shape[0]), Y].tolist()
# The above line is indexing into the prob tensor:
# prob[0, 5] = tensor(2.8872e-1) itos[5] 'e'
# prob[1, 13] = tensor(5.7237e-12) itos[13] 'm'
# prob[2, 13] = tensor(7.2747e-05) itos[13] 'm'
# prob[3, 1] = tensor(0.0030) itos[1] 'a'
# ...

# Now, prob[0] contains the probabilities for the first character, and the
# probabilities of the next character that the neural network predicted. So
# prob[0][0] would # be the probability of '.' being the next character,
# prob[0][1] would be the probability of 'a' being the next character.
# We know the true next characters are which we have in Y. We want to pluck out
# the probabilities that the neural network predicted for the true next character
# so we can compare them to the actual probabilities.
breakpoint()
# Keep in mind that we are in the training stage here and are trying to massage
# the weights and biases so that the neural network will predict the correct
# letter.
print(" ".join(format(x, 'f') for x in y_prop_list))
# (Pdb) p Y
# tensor([ 5, 13, 13,  1,  0, 15, 12,  9, 22,  9,  1,  0])
# (Pdb) p itos[Y[0].item()]
# 'e'
loss = -prob[torch.arange(emb.shape[0]), Y].log().mean()
print(f'{loss=}')


### Refactoring
g = torch.Generator().manual_seed(2147483647)
C = torch.randn(27, 2, generator=g)
W1 = torch.randn(6, 100, generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn(100, 27, generator=g)
b2 = torch.randn(27, generator=g)
params = [C, W1, b1, W2, b2]
print(f'Number of paramters: {sum(p.nelement() for p in params)}')
for p in params:
    p.requires_grad = True

# The forward pass
for _ in range(10):
    emb = C[X]
    h = torch.tanh(emb.view(emb.shape[0], 6) @ W1 + b1)
    logits = h @ W2 + b2

    # The following three lines can be replaced by F.cross_entropy
    #counts = logits.exp()
    #probs = counts / counts.sum(1, keepdim=True)
    #manual_loss = -probs[torch.arange(32), Y].log().mean() 
    loss = F.cross_entropy(logits, Y)
    print(f'{loss.item()=}')
    #print(f'{manual_loss=}')
    #print(f'{F.cross_entropy(logits, Y)=}') 

    # The backward pass
    for p in params:
        p.grad = None
    loss.backward() # Update the parameters
    for p in params:
        p.data += -0.1 * p.grad

print(f'{loss.item()=}')
print(f'{logits.max(1)=}')
print(f'{Y=}')

# Notice that this is pretty slow if we look at the terminal output while it
# is running. Normally what is done it that forward and backward passes are
# done on batches of the data. This is done in the next section.

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
print(f'{lrs=}')

lri = []
lossi = []

# The forward pass
for i in range(10000):
    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (32,))

    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(emb.shape[0], 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    print(f'{loss.item()=}')

    # The backward pass
    for p in params:
        p.grad = None
    loss.backward() # Update the parameters
    #lr = lrs[i]
    lr = 0.1
    for p in params:
        p.data += -lr * p.grad

    # track stats
    #lri.append(lre[i])
    #lossi.append(loss.item())
# Whao that is much faster. The reason is that we are only doing 32 forward
# passes and 32 backward passes instead of 32*32 = 1024 forward passes and
# backward passes.

print(f'{loss.item()=}')
#plt.plot(lri, lossi)
#plt.show()

emb = C[Xtr]
h = torch.tanh(emb.view(emb.shape[0], 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
print(f'Training loss: {loss.item()=}')

emb = C[Xdev]
h = torch.tanh(emb.view(emb.shape[0], 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(f'Dev loss: {loss.item()=}')
