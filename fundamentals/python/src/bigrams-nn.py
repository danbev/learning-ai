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

# string to integer dictionary.
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0
# integer to string dictionary.
itos = {i: ch for ch, i in stoi.items()}

# Training sets which consists of bigrams (x, y) where x is the first character
# which is provided and we are trying to guess the second character y.
xs = [] # inputs (ints)
ys = [] # targets/labels (ints)

# Create an empty matrix of size 27x27 with all zeros.
N = torch.zeros((27, 27), dtype=torch.int32)
# Instead of adding the counts to a matrix...
#for word in words:
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
