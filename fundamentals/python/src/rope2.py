import numpy as np
import matplotlib.pyplot as plt

# Example 4x4 embedding matrix (4 tokens, each with 4 dimensions)
embedding_matrix = np.array([
    [1, 2, 3, 4],  # Token 1
    [5, 6, 7, 8],  # Token 2
    [9, 10, 11, 12],  # Token 3
    [13, 14, 15, 16]  # Token 4
])

def rotate(pair, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(rotation_matrix, pair)

print("Original Embedding Matrix:")
print(embedding_matrix)

plt.figure(figsize=(8, 8))
for i in range(embedding_matrix.shape[0]):
    plt.arrow(0, 0, embedding_matrix[i, 0], embedding_matrix[i, 1], head_width=0.2, head_length=0.3, fc='blue', ec='blue')

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Original Vectors')
plt.grid(True)
plt.show()

def get_rotation_angle(position, dimension):
    omega = 0.1  # A constant for demonstration purposes
    return position * (omega ** dimension)

# Apply RoPE to the embedding matrix
rotated_embedding_matrix = np.zeros_like(embedding_matrix)

for i in range(embedding_matrix.shape[0]):  # Iterate over tokens
    for j in range(0, embedding_matrix.shape[1], 2):  # Iterate over dimension pairs
        angle = get_rotation_angle(i, j // 2)
        rotated_embedding_matrix[i, j:j+2] = rotate(embedding_matrix[i, j:j+2], angle)

# Display the rotated embedding matrix
print("\nRotated Embedding Matrix:")
print(rotated_embedding_matrix)

plt.figure(figsize=(8, 8))
for i in range(embedding_matrix.shape[0]):
    plt.arrow(0, 0, rotated_embedding_matrix[i, 0], rotated_embedding_matrix[i, 1], head_width=0.2, head_length=0.3, fc='red', ec='red')

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Rotated Vectors')
plt.grid(True)
plt.show()

# Plotting
plt.figure(figsize=(8, 8))
for i in range(embedding_matrix.shape[0]):
    # Original vector
    plt.arrow(0, 0, embedding_matrix[i, 0], embedding_matrix[i, 1], head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    # Rotated vector
    plt.arrow(0, 0, rotated_embedding_matrix[i, 0], rotated_embedding_matrix[i, 1], head_width=0.2, head_length=0.3, fc='red', ec='red')

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Original (Blue) vs Rotated (Red) Vectors')
plt.grid(True)
plt.show()
