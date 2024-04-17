d = 4096  # dimensionality of the embedding space
nr = 10
base = 10000

print(f'd={d}, base={base}, nr={nr}')
for i in range(0, nr):
    theta = base**(-2*(i-1)/d)
    print(theta)
print('---')
for i in range(0, 10):
    exponential = (-2*(i-1)/d)
    print(exponential)
