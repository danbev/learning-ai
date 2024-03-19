import numpy as np

# Create a sample matrix A
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(f'{A=}')

# Perform Singular Value Decomposition
U, S, VT = np.linalg.svd(A)

print(f'{U=}')
print(f'{S=}')
print(f'{VT=}')
