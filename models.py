import numpy as np

class ClusteringModel:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.centroids = None

    def fit(self, data):
        np.random.seed(42)
        random_idx = np.random.permutation(data.shape[0])
        self.centroids = data[random_idx[:self.num_clusters]]
        
        for _ in range(300):
            distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
            closest_cluster = np.argmin(distances, axis=0)
            
            new_centroids = np.array([data[closest_cluster == k].mean(axis=0) for k in range(self.num_clusters)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, data):
        distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def silhouette_score(self, data):
        a = np.zeros(data.shape[0])
        for k in range(self.num_clusters):
            cluster = data[self.assignments == k]
            for i in range(len(cluster)):
                a[self.assignments == k][i] = np.mean(np.linalg.norm(cluster[i] - cluster, axis=1))

        b = np.inf * np.ones(data.shape[0])
        for k in range(self.num_clusters):
            cluster = data[self.assignments == k]
            for j in range(self.num_clusters):
                if k != j:
                    other_cluster = data[self.assignments == j]
                    for i in range(len(cluster)):
                        dist = np.mean(np.linalg.norm(cluster[i] - other_cluster, axis=1))
                        if dist < b[self.assignments == k][i]:
                            b[self.assignments == k][i] = dist

        s = (b - a) / np.maximum(a, b)
        return np.mean(s)

    def compute_representation(self, X):
        return self.centroids


class ClassificationModel:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def train(self, X_train, y_train):
        pass
    def predict(self, X_test):
        pass
    def evaluate(self, X_test, y_test):
        pass
