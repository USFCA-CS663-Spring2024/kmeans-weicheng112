from cluster import cluster

import numpy as np


class KMeans(cluster):

    def __init__(self, k=5, max_iterations=100, balanced=False):
        super().__init__()
        self.k = k
        self.max_iterations = max_iterations
        self.balanced = balanced
        self.centroids = []

    def fit(self, X):
        # input X: a list  of n instances in d dimensions (features)
        # return : 1. A list (of length n) of the cluster hypotheses, one for each instance.
        #          2. A list (of length at most k) containing lists (each of length d) of the cluster centroids’ values.

        data_set = np.array(X);
        num_instances, num_features = np.shape(data_set)
        min_feature = np.min(data_set, axis=0)
        max_feature = np.max(data_set, axis=0)

        # place k centroids(𝜇1, 𝜇2, ..., 𝜇k∈ ℝn) randomly
        for _ in range(self.k):
            random_point = np.random.uniform(min_feature, max_feature)
            self.centroids.append(random_point)

        # Create a clusters array for num_instances
        clusters = np.zeros(num_instances, dtype=int)
        for _ in range(self.max_iterations):

            # foreach x ∈ test_x:
            #     c(i) = index of closest centroid to x
            for i, instance in enumerate(X):
                new_distances = self.dist_eclud(instance)
                cluster = np.argmin(new_distances)
                clusters[i] = cluster

            # Make each of the k clusters are (roughly) equal with
            #       respect to the number of instances in the cluster hypotheses
            if self.balanced:
                clusters = self.balance_clusters(clusters, num_instances, X)

            new_centroids = np.zeros((self.k, num_features))
            #     foreach k ∈ centroids:
            #         𝜇k = meanc(i) | index(c(i)) == k
            for i in range(self.k):
                my_instances_in_cluster = self.find_instance_in_cluster(i, data_set, clusters)
                if my_instances_in_cluster.any():
                    new_centroids[i] = np.mean(my_instances_in_cluster, axis=0)
                else:
                    # If a centroid lost all its members, initialize it within the bounds
                    new_centroids[i] = np.random.uniform(min_feature, max_feature)
            if np.all(new_centroids == self.centroids):
                break
            self.centroids = new_centroids

        return clusters.tolist(), self.centroids.tolist()
    def dist_eclud(self, my_instance):
        distances = []
        for centroid in self.centroids:
            distances.append(np.sqrt(np.sum(np.power(my_instance - centroid, 2))))
        return np.array(distances)
    def find_instance_in_cluster(self, index, data_set, clusters):
        instances = []
        for i in range(len(clusters)):
            if clusters[i] == index:
                instances.append(data_set[i])
        return np.array(instances)
    def balance_clusters(self, clusters, num_instances,X):
        size = num_instances // self.k
        for i in range(self.k):
            indices = np.where(clusters == i)[0]
            # clusters== i is a condition and the true value will place on [0]
            if len(indices) > size:
                num_needed_change = len(indices) - size
                for j in indices[0:num_needed_change]:
                    # 0~ num_needed_change - 1
                    distances = self.dist_eclud(X[j])
                    distances[i] = np.inf  # Make it positive infinity. Ignore the current cluster
                    clusters[j] = np.argmin(distances)
        return clusters
#
if __name__ == '__main__':

    mylist = [[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]]

    c = KMeans(k=2,balanced=True);
    mylist = [[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]]
    print(c.fit(mylist))
