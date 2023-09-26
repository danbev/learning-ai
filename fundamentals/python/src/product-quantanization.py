from random import randint
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

x = [1, 8, 3, 9, 1, 2, 9, 4, 5, 4, 6, 2]
print(f'{x=}')

# Number of entries per subvector
m = 4
dim = len(x)
sub_len = int(dim / m)
print(f'{dim=}')
print(f'{m=}')
print(f'{sub_len=}')

# Create the subvectors by splitting the input vector.
subvectors = []
for r in range(0, dim, sub_len):
    subvectors.append(x[r: r+sub_len])
print(f'{subvectors=}')

# k is the number of total centroids.
k = 32
assert k % m == 0

# centroids_per_subvector is the number of centroids per subvector
centroids_per_subvector = int(k / m)
print(f'Total number of centroids: {k}')
print(f'Number of centroids per sub vector: {centroids_per_subvector}\n')

# Create all the centroids, initialized randomally in the range of the 
# input vector dataset (1, 9).
centroids = []
for j in range(m):
    c_j = []
    for i in range(centroids_per_subvector):
        c_ji = [randint(min(x), max(x)) for _ in range(sub_len)]
        c_j.append(c_ji)

    centroids.append(c_j)

print(f'Codebook:')
for i, cs in enumerate(centroids):
    for j, c in enumerate(cs):
        # These arrays are the coordinates for the centroids.
        print(f'Centroid {i} ID: {j}: {c}')
    print("")

fig = plt.figure()

for j in range(m):
    ax = fig.add_subplot(2, 2, j+1, projection='3d')
    X = [centroids[j][i][0] for i in range(centroids_per_subvector)]
    Y = [centroids[j][i][1] for i in range(centroids_per_subvector)]
    Z = [centroids[j][i][2] for i in range(centroids_per_subvector)]
    ax.scatter(X, Y, Z)
    ax.set_title(f'c_{j}')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

#plt.show()

def euclidean(v, u):
    distance = sum((x - y) ** 2 for x, y in zip(v, u)) ** .5
    return distance

def nearest(c_j, u_j, centroids_per_subvector):
    distance = 9e9
    for i in range(centroids_per_subvector):
        new_dist = euclidean(c_j[i], u_j)
        if new_dist < distance:
            nearest_idx = i
            distance = new_dist
    return nearest_idx

ids = []
for j in range(m):
    i = nearest(centroids[j], subvectors[j], centroids_per_subvector)
    ids.append(i)

print(f'Quantized subvector: {ids}')

q = []
for j in range(m):
    # This following is looking up the centroid coordinates for the ids. By
    # looking at the codebook table printed earlier we can match the ids
    # to the respective centroid coordinates.
    c_ji = centroids[j][ids[j]]
    q.extend(c_ji)
print(q)

def mse(v, u):
    error = sum((x - y) ** 2 for x, y in zip(v, u)) / len(v)
    return error

mse = mse(x, q)
print(f'{mse=}')
