from numpy import random

#random.seed(18)
# generate an array of 10 values that follows the binomial distribution
# with n=10 and p=0.5
x = random.binomial(n=10, p=0.5, size=1)
print(x)

