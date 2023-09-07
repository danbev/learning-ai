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

def tanh_histogram(layers):
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
        if isinstance(layer, Tanh):
            t = layer.out
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__})')
            plt.legend(legends);
            plt.title('activation distribution')

    plt.show()

def grad_histogram(layers):
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
        if isinstance(layer, Tanh):
            t = layer.out.grad
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__})')
            plt.legend(legends);
            plt.title('activation distribution')

    plt.show()

class Linear:

    # The term `fan_in` refers to the number of inputs to a given node in the
    # layer and stems from the visual of a fan (for generating a breeze) with
    # blades converging to the center of the fan. We can think of the blades as
    # the input and the neuron is the center.
    def __init__(self, fan_in, fan_out, bias=True):
        self.weights = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
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

# Embedding matrix
C = torch.randn((vocab_size, nr_embeddings), generator=g)

# Layers of the network
layers = [
    Linear(block_size * nr_embeddings, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, n_hidden), Tanh(),
    Linear(n_hidden, vocab_size)
]
print(layers)

with torch.no_grad():
    # The following makes the first layer less confident about its predictions.
    layers[-1].weights *= 0.1
    for layer in layers:
        if isinstance(layer, Linear):
            layer.weights *= 5/3

parameters = [C] + [p for l in layers for p in l.parameters()]
print(f'Parameters: {sum(p.nelement() for p in parameters)}')
for p in parameters:
    p.requires_grad = True

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
  x = emb.view(emb.shape[0], -1) # concatenate the vectors
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb)
 
  for layer in layers:
      layer.out.retain_grad()
  for p in parameters:
      p.grad = None

  loss.backward()

  lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  if i % 10000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  # With the following breakpoint we can stop the training and inspect the
  # network state: 
  # (Pdb) histogram(layers)
  breakpoint()

print(f'Loss after training: {loss.item():.4f}')




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
