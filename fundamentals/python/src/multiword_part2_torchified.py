import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import multiword_utils as utils

g = torch.Generator().manual_seed(2147483647)

"""
This is the code for Building makemore Part 3: Activations & Gradient, BatchNorm
https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5

This is the last part that torchifies the code making it more simlilar to what
we would find in PyTorch.
"""

class Linear:

    # The term `fan_in` refers to the number of inputs to a given node in the
    # layer and stems from the visual of a fan (for generating a breeze) with
    # blades converging to the center of the fan. We can think of the blades as
    # the input and the neuron is the center.
    def __init__(self, fan_in, fan_out, bias=True):
        self.weights = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def _call__(self, x):
        self.out = x @ self.weights
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weights] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f'Linear(fan_in={self.weights.shape[0]}, fan_out={self.weights.shape[1]})'

class BatchNorm1d:

    def __init__(self, dim, esp=1e-5, momentum=0.1):
        self.esp = esp
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.esp)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                # Exponential moving average of the mean and variance
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    
        def __call__(self, x):
            self.out = torch.tanh(x)
            return self.out

        def parameters(self):
            return []

        def __repr__(self):
            return f'Tahn()'


nr_embeddings = 10
n_hidden = 100

vocab_size = len(utils.itos)
# Block size is the context length, as in how many characters to use to predict
# the next character.
block_size = 3 # context length: how many characters do we take to predict the next one?

C = torch.randn((vocab_size, nr_embeddings), generator=g)
layers = [
    Linear(block_size * nr_embeddings, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
]

print(layers)
exit()

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
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

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
  bnmeani = h_pre_act.mean(0, keepdim=True)
  bnstdi = h_pre_act.std(0, keepdim=True)
  h_pre_act = (h_pre_act - bnmeani) / bnstdi

  with torch.no_grad():
      bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
      bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

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

print(f'{bnmean=}, {bnmean_running=}')
print(f'{bnstd=}, {bnstd_running=}')


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
  h_pre_act = bngain * (h_pre_act - bnmean_running) / bnstd_running + bnbias
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
