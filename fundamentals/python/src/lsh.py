import random
# Locality Sensitive Hashing (LSH) Example
a = "Dan loves ice cream"
b = "Dan likes ice cream" 
c = "LSH is a cool algorithm that can be used to find similar items in a dataset. It is used in many applications such as recommendation systems, search engines, and more. In this post, we will explore how LSH works and how it can be applied to find similar items in a dataset."

def singling(text, k):
    shingles = []
    for i in range(len(text) - k + 1):
        shingles.append(text[i: i+k])

    return set(shingles)

k = 2;
s_a = singling(a, k)
print(f'Singlings for a: "{a}" \n{s_a}')
s_b = singling(b, k)
s_c = singling(c, k)

vocab = list(s_a.union(s_b).union(s_c))
print(f'vocab: {vocab}')
#for i, word in enumerate(vocab):
#    print(f'{i}: {word}')

# Create one-hont encoding for each shingle
a_one_hot = [1 if x in s_a else 0 for x in vocab]
print(f'a_one_hot: {a_one_hot}')
b_one_hot = [1 if x in s_b else 0 for x in vocab]
c_one_hot = [1 if x in s_c else 0 for x in vocab]

print(f'vocab: {len(vocab)}')

def shuffled_index(vocab):
    shuffled_indices = list(range(1, len(vocab)+1))
    random.shuffle(shuffled_indices)
    return shuffled_indices

def shuffled_indices(vocab, n):
    shuffled = []
    for _ in range(n):
        shuffled.append(shuffled_index(vocab))
    return shuffled

shuffled = shuffled_indices(vocab, 1)[0]
print(f'shuffled_indices: {shuffled}')

for i in range(1, 10+1):
    print(f'index {i}: is at position {shuffled.index(i)}')


for i in range(1, len(vocab)+1):
    idx = shuffled.index(i)
    sig_value = a_one_hot[idx]
    #print(f'index {i}: is at position {idx} and has value {sig_value}')
    if sig_value == 1:
        print(f'Found a match at index {i}!')
        break
                        

min_hashes = shuffled_indices(vocab, 20)

def create_signature(one_hot, min_hashes, vocab):
    signature = []
    # loop through all the shuffled indices
    for min_hash in min_hashes:
        for i in range(1, len(vocab)+1):
            sig_value = one_hot[min_hash.index(i)]
            if sig_value == 1:
                signature.append(i)
                break

    return signature

a_signature = create_signature(a_one_hot, min_hashes, vocab)
print(f'a_signature: {a_signature}')
b_signature = create_signature(b_one_hot, min_hashes, vocab)
c_signature = create_signature(c_one_hot, min_hashes, vocab)

def jaccard(a, b): 
    return len(a.intersection(b)) / len(a.union(b))

print(f'Jaccard similarity between a_sig and b_sig: {jaccard(set(a_signature), set(b_signature))}')
print(f'Jaccard similarity between a and b: {jaccard(set(a), set(b))}')

print(f'Jaccard similarity between a_sig and c_sig: {jaccard(set(a_signature), set(c_signature))}')
print(f'Jaccard similarity between a and c: {jaccard(set(a), set(c))}')


def split_vector(sig, b):
    assert len(sig) % b == 0
    subvectors = []
    r = int(len(sig) / b)
    print(f'{r=}')
    for i in range(0, len(sig), r):
        subvectors.append(sig[i: i+r])
    return subvectors

bands_a = split_vector(a_signature, 10)
print(f'bands_a: {bands_a}')
bands_b = split_vector(b_signature, 10)
print(f'bands_b: {bands_b}')

for a_rows, b_rows in zip(bands_a, bands_b):
    if a_rows == b_rows:
        print('Found a candidate pair: {} == {}'.format(a_rows, b_rows))
        break
