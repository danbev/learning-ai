import torch

# C represents the randomly initialized embedding matrix.
C = torch.tensor([[ 6.8443e-01, -9.1421e-01], # 0  '.'
        [ 1.7745e+00, -3.1138e-01],           # 1  'a'
        [ 7.0778e-01, -2.1498e-01],           # 2  'b'
        [-1.6449e+00,  2.1587e-01],           # 3  'c'
        [-2.8619e-02, -4.1739e-01],           # 4  'd'
        [ 2.7164e-01,  2.2510e+00],           # 5  'e'
        [-8.2070e-01,  1.8946e+00],           # 6  'f'
        [-9.4465e-01, -3.8865e-01],           # 7  'g'
        [ 5.0636e-01, -9.5188e-02],           # 8  'h'
        [ 4.3576e-01, -1.4386e+00],           # 9  'i'
        [-7.6660e-01,  1.8655e+00],           # 10 'j'
        [-2.5704e+00,  1.2103e+00],           # 11 'k'
        [ 1.8644e-01,  2.2112e-01],           # 12 'l'
        [-5.0437e-01,  1.1397e+00],           # 13 'm'
        [-3.1989e-01, -8.0815e-01],           # 14 'n'
        [-9.6159e-01,  7.8360e-02],           # 15 'o'
        [ 3.0150e-01, -1.7784e+00],           # 16 'p'
        [ 1.9133e-01,  2.8774e-01],           # 17 'q'
        [ 1.9634e-01,  2.1779e+00],           # 18 'r'
        [ 1.9811e+00, -1.7951e-01],           # 19 's'
        [ 1.0016e+00, -1.1002e+00],           # 20 't'
        [-1.1199e+00, -1.6660e-01],           # 21 'u'
        [ 9.4509e-01,  1.9054e+00],           # 22 'v'
        [ 2.1910e-03, -9.8781e-01],           # 23 'w'
        [-4.4769e-01, -1.3415e+00],           # 24 'x'
        [ 1.6452e-01, -7.0964e-01],           # 25 'y'
        [ 6.4244e-01,  1.8194e+00]])          # 26 'z'

X = torch.tensor([[ 0,  0,  0], # 0
        [ 0,  0,  5],           # 1
        [ 0,  5, 13],           # 2
        [ 5, 13, 13],           # 3 
        [13, 13,  1],           # 4
        [ 0,  0,  0],           # 5
        [ 0,  0, 15],           # 6
        [ 0, 15, 12],           # 7
        [15, 12,  9],           # 8
        [12,  9, 22],           # 9
        [ 9, 22,  9],           # 10
        [22,  9,  1]])          # 11

# We use X to index into C
emb = C[X]
print(emb)
print(emb.shape)

emb = []
for i, row in enumerate(X):
    emb_row = [] # new list for each row in X
    print(f'row: {i}')
    for index in row:
        print(f'index: {index}: {C[index]}')
        emb_row.append(C[index]) # this adds the tensor from C to the list

    print(f'emb_row: {emb_row}')
    emb.append(emb_row)

print(len(emb) ,len(emb[0]), len(emb[0][0]))
for xs in emb:
    print(xs)

emb = torch.empty(X.shape[0], X.shape[1], C.shape[1])

# Loop over each row in X
print(f'{X.shape[0]=}')
print(f'{X.shape[1]=}')
for i in range(X.shape[0]):
    # Loop over each column in the row
    print(f'{i}: {X[i]=}')
    for j in range(X.shape[1]):
        # Fetch the corresponding row from C using the index from X
        print(f'{j}: {C[j]=}')
        emb[i][j] = C[X[i][j]]

print(emb.shape)

