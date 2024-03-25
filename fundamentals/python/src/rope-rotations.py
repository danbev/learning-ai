d = 4096  # dimensionality of the embedding space

nr = 10

for i in range(0, nr):
    theta = 10000**(-2*(i-1)/d)
    print(theta)

