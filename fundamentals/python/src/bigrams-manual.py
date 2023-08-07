import urllib
import requests
import pprint
import torch
import matplotlib.pyplot as plt
"""
A bigram or digram is a sequence of two adjacent elements from a string of
tokens.
"""

# The following is just the code used to download the names.txt file.
"""
words = urllib.request.urlopen('https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
    ).read().decode('utf-8').splitlines()

# write words to file named names.txt
with open('src/names.txt', 'w') as f:
    for word in words:
        f.write(word + '\n')
"""

print("Reading names.text...")
words = open('src/names.txt', 'r').read().splitlines()
print(f'First 10 names: {words[:10]}')
print(f'Number of names: {len(words)}')
print(f'Shortest name: {min(len(w) for w in words)}')
print(f'Longest name: {max(len(w) for w in words)}')

all_letters = ''.join(words)
chars = sorted(list(set(all_letters)))
print(f'Number of unique characters: {len(chars)}')
print(f'Characters:\n{chars}')

# string to integer dictionary.
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0
# integer to string dictionary.
itos = {i: ch for ch, i in stoi.items()}

print('------------------------------------------')
# Create an empty matrix of size 27x27 with all zeros.
N = torch.zeros((27, 27), dtype=torch.int32)
for word in words:
  # Add bigrams for word
  chs = ['.'] + list(word) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    # the bigrams are added as integers.
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    # Add an increment the bigram count.
    N[ix1, ix2] += 1
print(f'N contains the integer representations of the bigrams.')
print(f'For example:\n{N[0]=}')
print(f'Inspect N[0][0]: {N[0][0]} (int), itos[N[0][0]]: "{itos[N[0][0].item()]}"')

#plt.imshow(N)
#plt.show()

# We can use matplotlib to display the bigram table.
fig = plt.figure(figsize=(16,16), dpi=100)
fig.suptitle('Bigram Table')
## Display data as an image (imshow), i.e., on a 2D regular raster.
image_axis = plt.imshow(N, cmap='Blues')
print('Bigram Table lables (not showing counts):')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        print(chstr, ' ', end='', flush=True)
        #print(N[i, j].item(), ' ', end='', flush=True)
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
    print("", flush=True)
plt.axis('off');
#plt.show()

print(f'N.shape: {N.shape}')
print(f'An entry in N is a Tensor: {type(N[2,2])}')
print(f'We can .item() to get the count for any label above.')
print(f'For example N[1][1].item(): {N[1][1].item()}')
p = N[0]
print(f'Notice that the values in N integers. For example the values in the first row are:\n{N[0]}')
print(f'We need to convert them to floats:\n{p}')
p = N[0].float()
p = p / p.sum()
print(f'After conversion:\n{p}')
print(f'p.sum(): {p.sum()}')
print(f'p.shape: {p.shape}')
print('------------------------------------------')

g = torch.Generator().manual_seed(18)
#p = torch.rand(3, generator=g)
#p = p / p.sum()
# So we are now going to take a stab at generating a characters and we do this
# by sampling from a multinomial distribution.
idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(f'sampled index: {idx}')
sampled_char = itos[idx]
print(f'sampled character: {sampled_char}')
# So in this case we sampled the character 'j' from the distribution p.
print(f'Number of words starting with {sampled_char}: {N[0][idx]} (out of {N[0].sum()})')
# These words are the ones starting with '.j' in the matplot lib figure.
# So j is the first character in the word, and then we "move" to the row with
# row 'j.' to get the characters that are preceded by j.
print(f'Row for .{sampled_char}: {N[idx].tolist()} , max:{N[idx].max()}')
# Now we want to sample from this row for the next character which will follow
# the character 'j' in the word that starts with 'j'.
#

print(f'p.sum: {p.sum()}')
# The +1 here is to avoid division by zero and called smoothing.
P = (N+1).float()
print(f'P.sum: {P.sum()} (sum of all counts of all of the names in N)')
# This is not what we want, we want to normalize each row separately.
print(f'P.shape: {P.shape}')
print(f'P.sum: {P.sum(dim=1, keepdim=True).shape}')
print(f'P.sum: {P.sum(dim=1, keepdim=True)}')
print(f'P[10][17]: {P[10][17].item()} (before normalization)')
print(f'P.sum(): {P.sum(dim=1, keepdim=True)}')
P /= P.sum(dim=1, keepdim=True)
print(f'P[0].sum: {P[0].sum()}')
print(f'P[10][17]: {P[10][17].item()} (after normalization)')

#import math
#image_axis = plt.imshow(P, cmap='Blues')
#for i in range(27):
#    for j in range(27):
#        chstr = itos[i] + itos[j]
#        print(chstr, ' ', end='', flush=True)
#        #print(N[i, j].item(), ' ', end='', flush=True)
#        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#        plt.text(j, i, float(f'{P[i, j].item():.4f}'), ha="center", va="top", color='gray')
#    print("", flush=True)
#plt.axis('off');

print('------------------------------------------')

for i in range(5):
    output = []
    idx = 0
    while True:
        p = P[idx]
        #p = N[idx].float()
        #p = p / p.sum()
        # We can verify that the model is actually trained despite the output
        # being mostly gibberish and not looking like real names.
        #p = torch.ones(27)/27.0 # untrained model
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        output.append(itos[idx])
        if idx == 0:
            break
    print(f"Generated: {''.join(output)}")

# So the current model is not very impressive and the following section will
# try to improve it.
log_likelihood = 0.0
bigrams_count = 0
# The following word will produce a negative_log_likelihood of infinity because
# a z followed by a q occurs zero times, zero count in the N matrix. This will
# be the case for all counts that have a zero count. This can be addressed with
# something called smoothing.
#for word in ["danielzq"]:
#for word in words[0:3]:
for word in words:
  chs = ['.'] + list(word) + ['.']
  #print(f'word: {word}')
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    bigrams_count += 1
    #print(f'bigram: {ch1}{ch2} prob: {prob:.4f}, {prob*100:.2f}%, logprob: {logprob:.4f}')

print(f'{bigrams_count=}')
print(f'We have 27 possible characters, so the probability of a random guess is 1/27 = {1/27:.4f}, {1/27*100:.2f}%')
# So we have the probablilties of all the bigrams but we want to get a single
# number for all the probabilities so that we can determine how well the model
# is working.
# Self documenting expression example (the '=') below
print(f'{log_likelihood=}')
negative_log_likelihood = -log_likelihood
# To use the log likelihood as a loss function we want don't want to have it
# be negative, so we take the negative of the log likelihood and then we can
# try to minimize this loss function so a larger value is worse and a smaller
# value is better.
print(f'sum of negative_log_likelihood: {negative_log_likelihood}')
avg_nll = negative_log_likelihood / bigrams_count
print(f'average of negative_log_likelihood: {avg_nll} which indicates the quality of the model')
# The goal of training is to minimize the average negative log likelihood. This
# is done by modifying the parameters of the model. The parameters of the model
# are the values in the matrix N which we can inspect visually in the matplotlib
# figure. These are currently stored in table and are counts of the number of
# times a character is followed by another character. Recall that we got these
# numbers by reading the names.txt file, and then we loop over all the names
# create bigrams for them, mapping the tokens to integers. These integers are
# then added to the matrix N and the counts for the bigram is incremented if it
# already exists in the matrix N. We then converted this matrix N to a matrix P
# and normalized the row.
plt.show()
# If the negative log likelihood gets lower the model is improving and is giving
# better probabilities


# In this case we have "trained" the model by looking at the counts of all the
# bigrams, and we converted those integers into floats and then normalized those
# rows of floats to get probability distributions, and stored them in matrix P
# (for parameters perhaps?).
# We could then use this matrix P to generate new names by sampling from it.
# And then we could evaluate the quality of the model by computing the
# negative log likelihood comparing it. The lower the negative log likelihood is
# the better our model is at givng high probablities to the actual next
# characters in names.
# 
