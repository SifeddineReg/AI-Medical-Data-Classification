import numpy as np

class ClusteringModel:
    def __init__(self, num_clusters):
        """
        Initialise le modèle de clustering avec un nombre spécifié de clusters.

        :param num_clusters: (int) Le nombre de clusters à former lors du clustering.
        """
        self.num_clusters = num_clusters
        self.labels = None
        self.centroids = None

    def fit(self, data):
        """
        Entraîne le modèle de clustering sur les données fournies.

        :param data: (array-like) Les données sur lesquelles entraîner le modèle.
        Chaque ligne correspond à une observation et
        chaque colonne à une caractéristique.
        """
        np.random.seed(30)
        centr = [data[np.random.randint(data.shape[0])]]
        for _ in range(1, self.num_clusters):
            dist = np.array([min(np.linalg.norm(x-c) ** 2 for c in centr) for x in data])
            cumulative_probabilities = (dist / dist.sum()).cumsum()
            r = np.random.rand()
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    centr.append(data[i])
                    break
        self.centroids = np.array(centr)

        for _ in range(300):
            self.labels = np.argmin(np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
            new_centroids = np.array([data[self.labels == k].mean(axis=0) for k in range(self.num_clusters)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, data):
        """
        Prédit les clusters pour les nouvelles données en utilisant
        le modèle de clustering entraîné.

        :param data: (array-like) Les nouvelles données pour lesquelles
        prédire les clusters. Chaque ligne correspond à une observation
        et chaque colonne à une caractéristique.
        :return: (array-like) Les prédictions de clusters pour les nouvelles données.
        """
        if self.labels is None:
            raise ValueError("model is None")
        return np.argmin(np.sqrt(((data - self.labels[:, np.newaxis])**2).sum(axis=2)), axis=0)

    def silhouette_score(self, data):
        """
        Évalue la performance du modèle de clustering en utilisant le Silhouette Score.

        :param data: (array-like) Les données sur lesquelles calculer le Silhouette Score.
        :return: (float) La valeur du Silhouette Score pour le modèle de clustering.
        """
        if self.labels is None:
            raise ValueError("model is None")

        a = np.zeros(data.shape[0])
        b = np.inf * np.ones(data.shape[0])

        for k in range(self.num_clusters):
            cluster = data[self.labels == k]
            for i in range(len(cluster)):
                a[self.labels == k][i] = np.mean(np.linalg.norm(cluster[i] - cluster, axis=1))
                other_clusters = [data[self.labels == j] for j in range(self.num_clusters) if j != k]
                b[self.labels == k][i] = min(np.mean(np.linalg.norm(cluster[i] - other_cluster, axis=1)) for other_cluster in other_clusters)

        return np.mean((b - a) / np.maximum(a, b))

    def compute_representation(self, X):
        if self.centroids is None:
            raise ValueError("model is None")
        
        return np.sqrt(((X - self.centroids[self.labels]) ** 2).sum(axis=1)).reshape(-1, 1)
    
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
        
def predict(X_train, y_train, x_test, k):
    distances = [[euclidean_distance(x, x_test), y] for x, y in zip(X_train, y_train)]
    distances.sort(key=lambda x: x[0])
    return np.argmax(np.bincount([y for _, y in distances[:k]]))

def knn(X_train, y_train, X_test, k):
    return [predict(X_train, y_train, x_test, k) for x_test in X_test]

def precision(y_true, y_pred):
    return np.mean(y_true == y_pred)

def recall(y_true, y_pred):
    return np.mean(y_true == y_pred)

def f1_score(y_true, y_pred):
    return 2 * precision(y_true, y_pred) * recall(y_true, y_pred) / (precision(y_true, y_pred) + recall(y_true, y_pred))

class ClassificationModel:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def train(self, X_train, y_train):
        """ 
        Entraîne le modèle de classification sur les données d'entraînement données.
        :param X_train: (numpy.ndarray) Les données d'entraînement, de forme (n_samples, input_dim).
        :param y_train: (numpy.ndarray) Les étiquettes d'entraînement, de forme (n_samples, output_dim).
        """

        self.model = knn(X_train, y_train, X_train, 3)

    def predict(self, X_test):
        """
        Effectue une prédiction sur de nouvelles données à l'aide
        du modèle de classification entraîné.
        :param X_test: (numpy.ndarray) Les nouvelles données,
        de forme (n_samples, input_dim).
        :return: (numpy.ndarray) Les prédictions du modèle,
        de forme (n_samples, output_dim).
        """

        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Évalue la performance du modèle de classification
        sur les données de test données à l'aide
        de métriques de classification.
        :param X_test: (numpy.ndarray) Les données de test,
        de forme (n_samples, input_dim).
        :param y_test: (numpy.ndarray) Les étiquettes de test,
        de forme (n_samples, output_dim).
        :return: (dict) Un dictionnaire contenant les métriques
        de classification calculées (precision, recall, f1-score)
        """

        y_pred = self.predict(X_test)
        return {"precision": precision(y_test, y_pred), "recall": recall(y_test, y_pred), "f1-score": f1_score(y_test, y_pred)}
