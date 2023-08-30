import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import multiword_utils as utils

"""
This is the code for Building makemore Part 3: Activations & Gradient, BatchNorm
https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5

This is pretty much the same code but with my own comments and some minor
changes.
"""

nr_embeddings = 30
vocab_size = len(utils.itos)
# Block size is the context length, as in how many characters to use to predict
# the next character.
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = utils.stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(f'{X.shape=}, {Y.shape=}')
  return X, Y

random.seed(42)
random.shuffle(utils.words)
n1 = int(0.8*len(utils.words))
n2 = int(0.9*len(utils.words))

Xtr,  Ytr  = build_dataset(utils.words[:n1])     # 80%
Xdev, Ydev = build_dataset(utils.words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(utils.words[n2:])     # 10%


# MLP revisited
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)
# We can predict the expected loss we should expect but taking the number
# of characters in the vocabulary which is 27. If we think about it, we have
# a uniform distribution over the characters, so the expected loss is the
# negative log of 1/27 which is 3.2958. This is the loss we should expect
# if we were to randomly guess the next character. If we have an initial loss
# that is much higher than that we can be pretty sure what our initial state
# is off.
C  = torch.randn((vocab_size, n_embd), generator=g)
# The multiplication part at the end is the kaiming initialization.
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3 / (n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0

# batch normalization gain
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
print("Total nr of parameters: ", sum(p.nelement() for p in parameters))
for p in parameters:
  p.requires_grad = True

print("W1 expected standard deviation: ", (5/3) / 30**0.5) # kaiming init
print("W1 actual standard deviation: ", W1.std().item())

# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

  # forward pass
  emb = C[Xb] # embed the characters into vectors
  embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
  # Linear layer
  h_pre_act = embcat @ W1 #+ b1 hidden layer pre-activation
  # Standardize the hidden layer to have a guassian distribution:
  h_pre_act = (h_pre_act - h_pre_act.mean(0, keepdim=True)) / h_pre_act.std(0, keepdim=True)
  # But we also need to scale the values with a gain and offset them using a bias.
  # This is done to allow for some wiggle room in the distribution during training.
  # Notice that for the first iteration all the gains will be 1 and  the biases
  # will be zero so this is a "no-op" for the first iteration. The gains and biases
  # will be updated during back propagation.
  h_pre_act = bngain * h_pre_act + bnbias # batch norm
  # Non-linearity
  h = torch.tanh(h_pre_act) # hidden layer
  logits = h @ W2 + b2 # output layer
  loss = F.cross_entropy(logits, Yb) # loss function

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  #plt.hist(h.view(-1).tolist(), 50)
  #plt.show()
  #breakpoint()
  #(Pdb) torch.sum(h.eq(1.0)).item()
  #290
  #(Pdb) torch.sum(h.eq(-1.0)).item()
  #235

print(f'Loss after training: {loss.item():.4f}')


#plt.plot(lossi)
#plt.show()


# Calibrate the batch normalization parameters after training
with torch.no_grad():
    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    h_pre_act = embcat @ W1 + b1
    # Now we measure the mean and standard deviation of the entire training set.
    bnmean = h_pre_act.mean(0, keepdim=True)
    bnstd = h_pre_act.std(0, keepdim=True)


@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
  x, y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
  }[split]
  emb = C[x] # (N, block_size, n_embd)
  embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
  h_pre_act = embcat @ W1 #+ b1 hidden layer pre-activation
  h_pre_act = bngain * (h_pre_act - bnmean) / bnstd + bnbias
  #h_pre_act = bngain * (h_pre_act - h_pre_act.mean(0, keepdim=True)) / h_pre_act.std(0, keepdim=True) + bnbias
  h = torch.tanh(h_pre_act)
  logits = h @ W2 + b2 # (N, vocab_size)
  loss = F.cross_entropy(logits, y)
  print(split, loss.item())

split_loss('train')
split_loss('val')

g = torch.Generator().manual_seed(2147483647)
for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all '...'
    while True:
      emb = C[torch.tensor([context])] # (1, block_size, d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)

      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    print(''.join(utils.itos[i] for i in out))
