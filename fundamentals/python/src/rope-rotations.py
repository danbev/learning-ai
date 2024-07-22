d = 512  # dimensionality of rope embedding space (1024/2)
base = 10000
print(f'd={d}, base={base}')

print('--- Dimensions 0-10 ---')
for i in range(0, 10):
    theta = base**(-2*(i-1)/d)
    print(f'theta_{i}: {theta:f}')

print('')

print('--- Dimensions 502-512 ---')
for i in range(502, d):
    theta = base**(-2*(i-1)/d)
    print(f'theta_{i}: {theta:.16f}')

print('')

import numpy as np
import matplotlib.pyplot as plt

d = 512  # dimensionality of rope embedding space
base = 10000

# Calculate theta values for all dimensions
theta_values = [base**(-2 * i / d) for i in range(d)]

# Plot theta values against dimension indices using a normal scale
plt.figure(figsize=(10, 6))
plt.plot(range(d), theta_values, label=r'$\theta_i$ values')
plt.xlabel('Dimension Index (i)')
plt.ylabel(r'$\theta_i$ Value')
plt.title(r'$\theta_i$ Values Across Dimensions')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Theta value for dimension index 2
theta_2 = 0.964662

# Positions in the sequence
positions = np.arange(0, 11, 1)  # From position 0 to 10

# Calculate cosine and sine values for each position
cos_values = np.cos(theta_2 * positions)
sin_values = np.sin(theta_2 * positions)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(positions, cos_values, label='cos(theta_2 * p)', marker='o')
plt.plot(positions, sin_values, label='sin(theta_2 * p)', marker='x')
plt.xlabel('Position (p)')
plt.ylabel('Value')
plt.title('Cosine and Sine Values for Theta_2 Over Positions')
plt.legend()
plt.grid(True)
plt.show()



