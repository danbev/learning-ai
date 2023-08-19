import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

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
#for word in words[:5]:
for word in words:
    context = [0] * block_size
    #print(f'{word=}, context={context}')
    for ch in word + '.':
        ix = stoi[ch] # map character to integer
        # X will represents the input for example 3 tensors that will represent
        # the word for which we want to predict the 4th character
        X.append(context) 
        Y.append(ix) # the labels
        #print(''.join([itos[i] for i in context]), '---->', itos[ix])
        # context[1:] is a slicing operation which creates a new list which will
        # contain all the elements of the constext list except the first one.
        # [1, 2, 3, 4][1:] -> [2, 3, 4]
        # The +[ix] will join a new list with the integer ix to the end of the
        # context list.
        context = context[1:] + [ix]

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
#h = emb.view(-1, 6) @ W1 + b1
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

#print(prob[torch.arange(32), Y])
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
