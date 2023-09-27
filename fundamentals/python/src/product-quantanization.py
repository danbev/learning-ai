from random import randint, seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

seed(20)

# Think of this as an embedding vector representing a word or a sentence.
x = [1, 8, 3, 9, 1, 2, 9, 4, 5, 4, 6, 2]
print(f'{x=}')

# m is the number of entries per subvector
m = 4
dim = len(x)
sub_len = int(dim / m)
print(f'Number of subvectors: {m=}')
print(f'Length of each subvector: {sub_len=}')
print(f'Dimension of input vector: {dim=}\n')

# Split the input vector into subvectors.
subvectors = []
for r in range(0, dim, sub_len):
    subvectors.append(x[r: r+sub_len])
print(f'Input split into Subvectors:\n{subvectors}\n')

# k is the number of total centroids.
k = 32
assert k % m == 0
print(f'Total number of centroids: {k}')

# centroids_per_subvector is the number of centroids per subvector
centroids_per_subvector = int(k / m)
print(f'Number of centroids per sub vector: {centroids_per_subvector}\n')

# Create all the centroids, initialized randomally in the range of the 
# input vector dataset (1, 9).
centroids = []
for j in range(m):
    centroids_for_subvector = []
    for i in range(centroids_per_subvector):
        centroid = [randint(min(x), max(x)) for _ in range(sub_len)]
        centroids_for_subvector.append(centroid)

    centroids.append(centroids_for_subvector)

# At this point we have subvectors populated with centroids randomally
# initialized in the range of the input vector dataset.
# Each centroid has an id, which is the index of the centroid in the
# subvector.

print(f'Codebook:')
for i, cs in enumerate(centroids):
    for j, c in enumerate(cs):
        # These arrays are the coordinates for the centroids.
        print(f'Centroid {i} ID: {j}: {c}')
    print("")

# The following plogs the centroids for each subvector in 3D.
fig = plt.figure()
for j in range(m):
    ax = fig.add_subplot(2, 2, j+1, projection='3d')
    X = [centroids[j][i][0] for i in range(centroids_per_subvector)]
    Y = [centroids[j][i][1] for i in range(centroids_per_subvector)]
    Z = [centroids[j][i][2] for i in range(centroids_per_subvector)]
    ax.scatter(X, Y, Z)
    ax.set_title(f'centroid {j}')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
# Comment in the following to see the plot. This is commented out as it
# can become annoying to have it "pop-up" every time you run the script.
#plt.show()

def euclidean(v, u):
    distance = sum((x - y) ** 2 for x, y in zip(v, u)) ** .5
    return distance

def nearest(centroids, subvectors, centroids_per_subvector):
    print(f'calculate nearest centroid for subvector: {subvectors}')
    distance = 9e9
    nearest_idx = -1
    # Iterate over all the centroids per subvector
    for i in range(centroids_per_subvector):
        # Compute the distance between the current centroid (0-7) and the 4 subvector
        # that our input was split into.
        new_dist = euclidean(centroids[i], subvectors)
        if new_dist < distance:
            # Notice that we are not returning the distance, but the index of
            # the centroid. This is the ID of the centroid.
            nearest_idx = i 
            distance = new_dist
        print(f'euclidean distance between Centroid {i} and subvector: {subvectors}, distance: {new_dist}, nearest: {nearest_idx}')
    return nearest_idx

ids = []
for j in range(m):
    i = nearest(centroids[j], subvectors[j], centroids_per_subvector)
    print(f'Nearest centroid (ID) for subvector {subvectors[j]} = {i}')
    ids.append(i)

print(f'\nQuantized subvector: {ids}')

q = []
for j in range(m):
    # This following is looking up the centroid coordinates for the ids. By
    # looking at the codebook table printed earlier we can match the ids
    # to the respective centroid coordinates.
    c_ji = centroids[j][ids[j]]
    print(f'centroid ({j}) for {ids[j]} (ID): {c_ji}')
    q.extend(c_ji)

def mse(v, u):
    error = sum((x - y) ** 2 for x, y in zip(v, u)) / len(v)
    return error

mse = mse(x, q)
print(f'{mse=}')

# Now, if we get a query vector we want to find the closest vector in the
# database to that query. We can do this by computing the distance
# between the query vector and each vector in the database. However, this
# is computationally expensive. Instead, we can use the quantized version
# of the query vector to find the closest vector in the database.
# What we are going to do here is going to estimate the distances using
# the vector-to-centroids distances. This is an approximation of the
# actual distance, but it is much faster to compute.

lookup_tables = []
print(f'\nCreate Lookup tables:')
for j in range(m):
    lookup_table = []
    for i in range(centroids_per_subvector):
        row = []
        for l in range(centroids_per_subvector):
            print(f'Centroid {j}_{i} to centroid {j}_{l} distance: {euclidean(centroids[j][i], centroids[j][l])}')
            row.append(euclidean(centroids[j][i], centroids[j][l]))
        lookup_table.append(row)
        print("")
    lookup_tables.append(lookup_table)

for i, t in enumerate(lookup_tables):
    print(f'Lookup table (distances) for subvector {i}:')
    for j, row in enumerate(t):
        print(f'{j}: {row[0:3]}...')
    print("")


def approx_distance(q_id, db_id, lookup_tables):
    dist = 0 
    print(f'query_id: {q_id}, db_id: {db_id}')
    for j in range(m):
        print(f'Going to lookup the distance for query ID {q_id[j]} and db_id {db_id[j]} in lookup table {j}')
        # So for the query id we are going to look up the entry 
        # subvector 0: q_id = 1, db_id = 0 :
        #
        # lookup_tables[0][1][0]
        # Lookup table (distances) for subvector 0:
        # 0: [0.0, 3.7416573867739413, 4.47213595499958]...
        # 1: [3.7416573867739413, 0.0, 4.242640687119285]...
        # So this will return 3.7416573867739413 which was the distance between
        # centroid 0_0 and centroid 0_1:
        # 
        # Create Lookup tables:
        # Centroid 0_0 to centroid 0_0 distance: 0.0
        # Centroid 0_0 to centroid 0_1 distance: 3.7416573867739413
        dist_entry = lookup_tables[j][q_id[j]][db_id[j]]
        print(f'distance: {dist_entry}')
        # We add the distance to the total distance
        dist += dist_entry
    return dist


# Lets pretend that our database consists of only two vectors. Imaging these
# vectors represent embeddings.
database_vectors = [x]
# Just adding another similar vector to the database.
database_vectors.append([1, 8, 3, 9, 1, 2, 9, 4, 5, 5, 7, 3])
print(f'\nDatabase vectors: {database_vectors}')

database_ids = [ids]
database_ids.append([0, 0, 3, 6])
print(f'\nDatabase IDs: {database_ids}')

# Pretend that this is a query vector that as been split up and quantized.
query_ids = [1, 0, 2, 6]
print(f'Query IDs: {query_ids}\n')

# Find the closest vector in the database to the query vector. Notice that 
# the query vector only contains the IDs of the centroids.
min_dist = float('inf')
closest_vector = None

# We are going to iterate over all the database id vectors in the our database
# (centroid ids) and for each vector in the database we are going to compute
# the approximate distance between the query vector and that database vector.
# The distance that is closest is what will be set as the closest vector.
for idx, db_ids in enumerate(database_ids):
    dist = approx_distance(query_ids, db_ids, lookup_tables)
    if dist < min_dist:
        min_dist = dist
        closest_vector = database_vectors[idx]

print(closest_vector, min_dist)
#print(x)
