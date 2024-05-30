import numpy as np

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