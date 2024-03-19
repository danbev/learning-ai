import numpy as np
import matplotlib.pyplot as plt

# Function to perform element-wise rotation of the token embeddings
def rotate_half(x):
    # Assume x has shape (batch_size, dim)
    dim = x.shape[1]
    
    # Rotate the first half of the dimensions to the second half and vice versa
    return np.concatenate([x[:, dim // 2:], x[:, :dim // 2]], axis=1)

# Function to apply rotary positional encoding
def apply_rotary_positional_embedding(x, r):
    return x * np.cos(r) + rotate_half(x) * np.sin(r)

# Assume we have 2D embeddings for each of the 6 words in the sentence "The cat sat on the mat"
# For simplicity, let's just use random embeddings (in practice, these would be learned)
np.random.seed(0)
token_embeddings = np.random.rand(6, 4)  # 6 words, 4-dimensional embeddings

# Assume we have pre-computed rotary positional encodings for each position in the sentence
# For simplicity, we'll use random encodings (in practice, these would be based on the position)
positional_encodings = np.random.rand(6, 4)  # 6 positions, 4-dimensional encodings

# Apply rotary positional encoding to each token
rotated_embeddings = apply_rotary_positional_embedding(token_embeddings, positional_encodings)

# Show the original and rotated embeddings
print(f'Original: {token_embeddings}')
print(f'Rotated: {rotated_embeddings}')

# For simplicity, let's consider only the first two dimensions of the embeddings
token_embeddings_2d = token_embeddings[:, :2]
rotated_embeddings_2d = rotated_embeddings[:, :2]

words = ["The", "cat", "sat", "on", "the", "mat"]

plt.figure(figsize=(10, 10))

# Plot original embeddings
for i, word in enumerate(words):
    plt.scatter(token_embeddings_2d[i, 0], token_embeddings_2d[i, 1], marker='o', label=f'Original - {word}')
    plt.text(token_embeddings_2d[i, 0], token_embeddings_2d[i, 1], f'  {word}', fontsize=12)

# Plot rotated embeddings
for i, word in enumerate(words):
    plt.scatter(rotated_embeddings_2d[i, 0], rotated_embeddings_2d[i, 1], marker='x', label=f'Rotated - {word}')
    plt.text(rotated_embeddings_2d[i, 0], rotated_embeddings_2d[i, 1], f'  {word}', fontsize=12)

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('2D Representation of Original and Rotated Embeddings')
plt.legend()
plt.show()


