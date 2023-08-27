import torch
import matplotlib.pyplot as plt

x = torch.randn(100, 10) # inputs, 100 examples vectors of 10 dimensions
w = torch.randn(10, 200) # wieghts, 10 input vectors of 200 dimensions
#w = torch.randn(10, 200) / 10**0.5  # wieghts, 10 input vectors of 200 dimensions

y = x @ w # calculate the pre-activation values.

print(f'{x.mean()=}, {x.std()=}, {y.mean()=}, {y.std()=}')

fig = plt.figure(figsize=(20, 5))
x_plot = fig.add_subplot(1, 2, 1)
x_plot.set_title('Histogram for x')
plt.hist(x.view(-1).tolist(), 50, density=True)

y_plot = plt.subplot(1, 2, 2) # same as fig.add_subplot(1,2,2), 1 row, 2 cols, position 2.
y_plot.set_title('Histogram for y')
plt.hist(y.view(-1).tolist(), 50, density=True)

plt.show()
