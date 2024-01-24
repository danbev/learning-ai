import numpy as np

# Simplified example to illustrate the concept
# Assume small matrices for demonstration

# Initial values
print('--------------Initial values-----------------')
Q_i = np.array([[0.1, 0.2], [0.3, 0.4]])  # A block of Q
print(f'Q_i:\n{Q_i}')
K_j = np.array([[0.1, 0.3], [0.2, 0.4]])  # A block of K (already transposed for simplicity)
print(f'K_j:\n{K_j}')
V_j = np.array([[0.1, 0.2], [0.3, 0.4]])  # A block of V
print(f'V_j:\n{V_j}')
O_i = np.zeros((2, 2))  # Initialize O_i to zeros
print(f'O_i:\n{O_i}')
m_i = np.full(2, -np.inf)  # Initialize m_i to negative infinity
print(f'm_i:\n{m_i}')
l_i = np.zeros(2)  # Initialize l_i to zeros
print(f'l_i:\n{l_i}')

print('---------------------------------------------')

# Step 9: Compute S_i_j
S_i_j = Q_i @ K_j
print(f'S_i_j (Q_i x K_j):\n{S_i_j}')

# Step 10: Compute m_hat_i_j, P_hat_i_j, l_hat_i_j
m_hat_i_j = np.max(S_i_j, axis=1)
print(f'm_hat_i_j (max of S_i_j):\n{m_hat_i_j}')
P_hat_i_j = np.exp(S_i_j - m_hat_i_j[:, None])  # Subtract m_hat_i_j from each row of S_i_j and exponentiate
print(f'P_hat_i_j:\n{P_hat_i_j}')
l_hat_i_j = np.sum(P_hat_i_j, axis=1)
print(f'l_hat_i_j (sum of P_hat_i_j):\n{l_hat_i_j}')

# Step 11: Update m_new_i and l_new_i
m_new_i = np.maximum(m_i, m_hat_i_j)
print(f'm_new_i (max of m_i and m_hat_i_j):\n{m_new_i}')
l_new_i = np.exp(m_i - m_new_i) * l_i + np.exp(m_hat_i_j - m_new_i) * l_hat_i_j
print(f'l_new_i:\n{l_new_i}')

# Step 12: Update O_i
diag_l_new_i = np.diag(l_new_i)
print(f'diag_l_new_i:\n{diag_l_new_i}')
#diag_l_new_i_inv = diag_l_new_i / l_new_i
#diag_l_new_i_inv = np.diag(1 / l_new_i)
eps = 1e-10
diag_l_new_i_inv_elements = 1 / np.where(diag_l_new_i.diagonal() == 0, eps, diag_l_new_i.diagonal())
diag_l_new_i_inv = np.diag(diag_l_new_i_inv_elements)
print(f'diag_l_new_i_inv:\n{diag_l_new_i_inv}')
diag_l_i_exp = np.diag(np.exp(m_i - m_new_i))
print(f'diag_l_i_exp:\n{diag_l_i_exp}')
O_i = diag_l_new_i_inv @ (diag_l_i_exp @ O_i + P_hat_i_j @ V_j)
print(f'O_i:\n{O_i}')

# Displaying the matrices for visualization
#print(S_i_j, m_hat_i_j, P_hat_i_j, l_hat_i_j, m_new_i, l_new_i, O_i)


