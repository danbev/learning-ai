def pairs_count(ids, d=None):
    d = {} if d is None else d
    for pair in zip(ids, ids[1:]):
        print(f'checking pair: {pair}')
        d[pair] = d.get(pair, 0) + 1
    return d

inputs = [1, 2, 3, 1, 2]
counts = pairs_count(inputs)
print(counts)

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

ids = merge(inputs, (1, 2), 4)
print(ids)

vocab = {idx: bytes([idx]) for idx in range(256) }
print(vocab[255])
print(bytes([255]));
