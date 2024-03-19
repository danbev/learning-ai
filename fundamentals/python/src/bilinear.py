import numpy as np
import matplotlib.pyplot as plt

# Define a point in the S-plane (continuous-time domain)
# For example, let's take a point at -1 + 2j
s_point = -1 + 2j
print(s_point)

# Sampling period (choose an appropriate value, e.g., 1 second)
T = 1

# Manual bilinear transform calculation
# z = (1 + (T/2)*S) / (1 - (T/2)*S)
# z = (1 + (1/2)*S) / (1 - (1/2)*S)
# z = (1 + 0.5*S) / (1 - 0.5*S)
z_point = (1 + 0.5 * s_point) / (1 - 0.5 * s_point)
print('z_point =', z_point)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the point on the S-plane
ax1.plot(s_point.real, s_point.imag, 'rx', label='Pole')
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.axhline(0, color='black')
ax1.axvline(0, color='black')
ax1.set_title('S-Plane')
ax1.set_xlabel('Real Part')
ax1.set_ylabel('Imaginary Part')
ax1.grid(True)
ax1.legend()

# Plot the point on the Z-plane
ax2.plot(z_point.real, z_point.imag, 'rx', label='Pole')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.axhline(0, color='black')
ax2.axvline(0, color='black')
ax2.add_patch(plt.Circle((0, 0), 1, color='blue', fill=False))
ax2.set_title('Z-Plane')
ax2.set_xlabel('Real Part')
ax2.set_ylabel('Imaginary Part')
ax2.grid(True)
ax2.legend()

plt.show()

