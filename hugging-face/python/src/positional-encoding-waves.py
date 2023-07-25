import numpy as np
import matplotlib.pyplot as plt

# Generate a range of values from 0 to 2pi to represent the input to the
# sine/cosine functions
x = np.linspace(start=0, stop=(2 * np.pi), num=1000) # num is the number of samples to generate

# Compute the sine and cosine of these values
y_sin = np.sin(x)
y_cos = np.cos(x)

# Create a new figure
fig, axs = plt.subplots(2)

# Plot the cosine wave on the first subplot
axs[0].plot(x, y_cos, label='Cosine', color='orange')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title('Cosine Wave')
axs[0].legend()

# Plot the sine wave on the second subplot
axs[1].plot(x, y_sin, label='Sine')
axs[1].set_ylabel('y')
axs[1].set_title('Sine Wave')
axs[1].legend()

# Display the plot
plt.tight_layout()
plt.show()


