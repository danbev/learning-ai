import torch
import matplotlib.pyplot as plt
"""
This example attempts to do the same thing as bigrams.py but using a neural
network. The idea is to use a neural network to predict the next character in
a word given the previous character. The network will be trained on a dataset
of names and then we will use it to generate new names.
"""

print("Reading names.text...")
words = open('src/names.txt', 'r').read().splitlines()
all_letters = ''.join(words)
chars = sorted(list(set(all_letters)))

# string to integer dictionary mapping.
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0
# integer to string dictionary mapping.
itos = {i: ch for ch, i in stoi.items()}

# Training sets which consists of bigrams (x, y) where x is the first character
# which is provided and we are trying to guess the second character y.
xs = [] # inputs (ints)
ys = [] # targets/labels (ints)

# Just using one word for now
for word in words[:1]:
  # Add bigrams for word
  chs = ['.'] + list(word) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    # the bigrams are added as integers.
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs) # dtype will be int64
ys = torch.tensor(ys) # dtype will be int64
print(f'When the following inputs are entered:\n {xs}')
print(f'We want the following targets/lables to have a high probability:\n {ys})')
print(f'name: {words[0]}')
print(f' when the input is "{itos[xs[0].item()]}" (xs[0]: {xs[0]})')
print(f' the desired label is "{itos[ys[0].item()]}" (y[0]) (ys[0]: {ys[0]})"')
print(f' when the input is "{itos[xs[1].item()]}" (xs[1]) (xs[1]: {xs[1]})')
print(f' the desired label is "{itos[ys[1].item()]}" (y[1])" (ys[1]: {ys[1]})"')
print("  ...")

print("So we want to take our inputs, which are integers in the tensor xs and")
print("input them into a neural network. To to this we need to convert the integers")
print("into one-hot vectors.")
#one_hot_vectors = torch.nn.functional.one_hot(xs, num_classes=xs.max()+1)
x_hot_encoded = torch.nn.functional.one_hot(xs, num_classes=27).float()
print(f'Convert xs: {xs} into one-hot vectors:')
print(f'{x_hot_encoded}, {x_hot_encoded.dtype=}')
print(f'Notice that the shape of the one-hot vectors is {x_hot_encoded.shape}')
print(f'which is the number of possible characters we have.')
print(f'and {x_hot_encoded.dtype=}. We need to have floats to be able to feed')
print(f'this into a neural network.')
plt.imshow(x_hot_encoded)
#plt.show()
print(f'We can feed this one-hot encoded tensor into a neural network.')

g = torch.Generator().manual_seed(2147483647)
# randn is for random normal distribution.
print(f'We generate random normaly distributed weights for the neural network.')
W = torch.randn((27, 27), dtype=torch.float32, requires_grad=True, generator=g)
# Recall that in our manual bigram version we had a matrix of size 27x27 and
# the entries were the counts of each bigram. We have the name type of matrix
# here but the entries are random numbers.
print(f'Weights matrix W: {W}, shape: {W.shape}, {W.dtype=}')
fig = plt.figure(figsize=(60,60), dpi=80)
fig.suptitle('Bigram NN Table')
## Display data as an image (imshow), i.e., on a 2D regular raster.
image_axis = plt.imshow(W.detach().numpy(), cmap='Blues')
print('Bigram Table lables (not showing counts):')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        print(chstr, ' ', end='', flush=True)
        #print(float(f'{W[i, j].item():.2f}'), ' ', end='', flush=True)
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, float(f'{W[i, j].item():.2f}'), ha="center", va="top", color='gray')
    print("", flush=True)
plt.axis('off');
plt.show()

# Multiply the weights by the one-hot encoded vectors, the inputs
print('\nNow we multiply the weights by the one-hot encoded vectors (the inputs)')
# The following are all equivalent:
logits = x_hot_encoded.mm(W)
logits = torch.mm(x_hot_encoded, W)
logits = torch.matmul(x_hot_encoded, W)
logits = x_hot_encoded @ W
print(f'logits: {logits}, shape: {logits.shape}, {logits.dtype=}')
# logits are the raw unnormalized predictions that a classification model
# generates, before applying the activation function. So these are the values
# before they are squashed through a function that makes them probabilities.
# In deep learning, particularly in multi-class classification, the term "logits"
# has been generalized to refer to the vector of raw scores output by the model
# before applying the softmax function.

print("\nLets take a look at calculating one of the entries in the logits matrix.")
print(f'{logits[3, 13]=}') # or print(logits[3][13])
print(f'{x_hot_encoded[3]=}')
# The following is using slicing and : is saying take everything along this
# dimension, and then only take the elements in the 13th column (starting from
# 0).
print(f'{W[:,13]=}') # this is .m
print(f'(x_hot_encoded[3] * W[:,13] = {(x_hot_encoded[3] * W[:,13]).sum()}')
# We have 27 inputs and 27 neurons in the hidden layer.

print('Notice that this matrix contains both negative and positive numbers.')
print('We can use the exponential function to convert these numbers into positive number')
counts = logits.exp() # exp is the exponential function, eˣ.
print(f'{counts=}')
# counts is similar to the N matrix in the bigrams-manual.py example. Recall
# that the P matrix was the N matrix normalized so we also need to normalize
# the counts matrix into a new tensor named probs:
probs = counts / counts.sum(dim=1, keepdim=True)
#breakpoint()
# The following two functions together are called the softmax function:
# counts = logits.exp()
# probs = counts / counts.sum(dim=1, keepdim=True)

print(f'{probs.shape=}')
print(f'{probs[0]=}')
print('The values in the above output are the probabilities of the character to come next')
print(f'We have 27 characters remember: {chars=}, count: {len(chars)} + the "." character')
print(f'probs: {probs[0, 5].data}, {probs[1, 13].data}, {probs[2, 13].data}, {probs[3, 1].data}, {probs[4, 0].data}')
print(torch.arange(5))
print(f'{ys=}')
# The following is using tensors to index into the probs tensor. This is a way
# of selecting rows and columns that we are interested in.
print(f'{probs[torch.arange(5), ys]=}')
loss = -probs[torch.arange(5), ys].log().mean()
print(f'loss: {loss.data}')
print(W.grad)
W.grad = None # reset the gradient to zero.
loss.backward()
print(probs[torch.tensor([0, 1, 2, 3, 4]), ys].data)
