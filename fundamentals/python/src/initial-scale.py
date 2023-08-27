import torch
import matplotlib.pyplot as plt

x = torch.randn(100, 10) # inputs, 100 examples vectors of 10 dimensions
w = torch.randn(10, 200) # wieghts, 10 input vectors of 200 dimensions
y = x @ w # calculate the pre-activation values.

print(f'{x.mean()=}, {x.std()=}, {y.mean()=}, {y.std()=}')

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].hist(x.view(-1).tolist(), bins=50, color='b', alpha=0.7)
axes[0, 0].set_title(f'Histogram for x before')
axes[0, 0].set_xlabel('Distribution')
axes[0, 0].set_ylabel('Value')

# Plot histogram and add labels for the second subplot
axes[0, 1].hist(y.view(-1).tolist(), bins=50, color='r', alpha=0.7)
axes[0, 1].set_title('Histogram for y before')
axes[0, 1].set_xlabel('Distribution')
axes[0, 1].set_ylabel('Value')

x = torch.randn(100, 10) # inputs, 100 examples vectors of 10 dimensions
w = torch.randn(10, 200) / 10**0.5  # wieghts, 10 input vectors of 200 dimensions
y = x @ w # calculate the pre-activation values.
print(f'{x.mean()=}, {x.std()=}, {y.mean()=}, {y.std()=}')

axes[1, 0].hist(x.view(-1).tolist(), bins=50, color='g', alpha=0.7)
axes[1, 0].set_title('Histogram for x after')
axes[1, 0].set_xlabel('Distribution')
axes[1, 0].set_ylabel('Value')

# Plot histogram and add labels for the second subplot
axes[1, 1].hist(y.view(-1).tolist(), bins=50, color='m', alpha=0.7)
axes[1, 1].set_title('Histogram for y after')
axes[1, 1].set_xlabel('Distribution')
axes[1, 1].set_ylabel('Value')

plt.show()
