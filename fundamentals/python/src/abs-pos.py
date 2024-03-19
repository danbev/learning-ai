import numpy as np
import matplotlib.pyplot as plt

# Example 4x4 embedding matrix (4 tokens, each with 4 dimensions)
embedding_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

def positional_encoding(position, d_model):
    """ Generate absolute positional encoding for a given position and model dimension """
    pos_enc = np.zeros((1, d_model))
    for i in range(d_model):
        if i % 2 == 0:
            pos_enc[:, i] = np.sin(position / (10000 ** (i / d_model)))
        else:
            pos_enc[:, i] = np.cos(position / (10000 ** ((i - 1) / d_model)))
    return pos_enc

# Apply absolute positional encoding to the embedding matrix
pos_encoded_embedding_matrix = np.zeros_like(embedding_matrix)
for i in range(embedding_matrix.shape[0]):
    pos_encoded_embedding_matrix[i, :] = embedding_matrix[i, :] + positional_encoding(i, embedding_matrix.shape[1])

# Plotting
plt.figure(figsize=(8, 8))
for i in range(embedding_matrix.shape[0]):
    # Original vector
    plt.arrow(0, 0, embedding_matrix[i, 0], embedding_matrix[i, 1], head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    # Positionally encoded vector
    plt.arrow(0, 0, pos_encoded_embedding_matrix[i, 0], pos_encoded_embedding_matrix[i, 1], head_width=0.2, head_length=0.3, fc='green', ec='green')

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Original (Blue) vs Positionally Encoded (Green) Vectors')
plt.grid(True)
plt.show()

