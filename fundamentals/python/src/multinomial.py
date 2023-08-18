import numpy as np
import math
import torch

# The number of experiments
n = 20

# The probabilities for each category
p = [0.5, 0.3, 0.2]
p = [1/6.]*6
print(*p, sep='\n')

np.random.seed(18)

samples = np.random.multinomial(n, p, size=1)
print(samples)
print(len(samples))
for i, d in enumerate(samples[0]):
    print(f'landed on {i}, {d} times')


torch.random.manual_seed(18)
weights = torch.tensor(p, dtype=torch.float)
out = torch.multinomial(weights, num_samples=n, replacement=True)
print(f'Notice that torch returns the drawn indices whereas np will return counts')
out_counts = out.unique(return_counts=True)[1]
print(f'torch {out=}')
for i, d in enumerate(out_counts, 1):
    print(f'landed on {i}, {d} times')
exit()

# The observed counts for each category
counts = [5, 3, 2]
print("The number of experiments is:", n)
print("The probabilities are:", p)
print("The observed counts are:", counts)

# Calculate the multinomial coefficient
coefficient = math.factorial(n) / (math.factorial(counts[0]) * math.factorial(counts[1]) * math.factorial(counts[2]))

print("The multinomial coefficient is:", coefficient)
# Calculate the probability of this combination
probability = coefficient * (p[0] ** counts[0]) * (p[1] ** counts[1]) * (p[2] ** counts[2])

print("The probability of drawing 5 Red, 3 Green, and 2 Blue balls is:", probability)


