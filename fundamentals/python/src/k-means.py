import numpy as np
# Just an example of k-means clustering to help understand its usage
# in vector databases product quantization.

class KMeansClustering: 

    def __init__(self, k):
        self.k = k
        self.centroids = None

    def fit(self, X, max_iterations=100):
        #print(f'X:\n{X}')
        #print(f'X.shape:\n{X.shape}')
        # First we pick k random points as centroids, but inside the bounds of
        # the dataset of X. Each dimension will have a min and a max value in
        # the dataset. We will pick random values between these min and max
        # values for each dimension. Below we are retrieving the min and max
        # values for each dimension. If we don't to this the random centroids
        # might initally be outside of the dataset and the algorithm will have
        # to perform more iterations to find the correct centroids.
        #print(f'random min value: {np.amin(X, axis=0)}')
        #print(f'random max value: {np.amax(X, axis=0)}')
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))
        #print(f'{self.k=}')
        #print(f'Randomly chosen centroids:\n{self.centroids}')


        def euclidean_distance(x, centriods):
            return np.sqrt(np.sum((x - centriods) ** 2, axis=1))

        for _ in range(max_iterations):
            y = []
            # Now we are going to calculate the distance between each point in
            # the dataset and each centroid.
            for x in X:
                distances = euclidean_distance(x, self.centroids)
                #print(f'{distances=}')
                # distances is an array of the distances between x and each
                # centroid. We are going to pick the centroid with the smallest
                # distance to x.
                cluster = np.argmin(distances)
                #print(f'cluster for x: {cluster}')
                y.append(cluster)

            y = np.array(y)

            cluster_indices = []
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(np.abs(self.centroids - cluster_centers)) < 1e-6:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y

random_data = np.random.randint(0, 100, size=(100, 2))

kms = KMeansClustering(3)
lables = kms.fit(random_data)
print(f'{lables=}')

import matplotlib.pyplot as plt

plt.scatter(random_data[:, 0], random_data[:, 1], c=lables)
plt.scatter(kms.centroids[:, 0], kms.centroids[:, 1], c=range(len(kms.centroids)),
            marker='8', s=200, linewidths=5, zorder=10)
plt.show()
