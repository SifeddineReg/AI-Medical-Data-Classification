import os
import numpy as np
import pandas as pd
import pickle as pkl

from models import ClusteringModel, ClassificationModel

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    test_samples = int(num_samples * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def student_model_train():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    save_folder = "./models/"

    print("Loading the dataset")
    wdbc_data = pd.read_csv("data/wdbc.data", header=None)
    X = wdbc_data.iloc[:, 2:-1].values
    y = wdbc_data.iloc[:, 1].values

    print("Encoding target labels")
    y_int = np.zeros((len(y), 1), dtype=int)
    y_int[y == 'B'] = 0
    y_int[y == 'M'] = 1
    y = y_int.flatten()

    print("Sampling the training set")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    print("Training and saving the Clustering model")
    num_clusters = 5
    clustering_model = ClusteringModel(num_clusters)
    clustering_model.fit(X_train)
    with open(os.path.join(save_folder, 'clustering.pkl'), 'wb') as f:
        pkl.dump(clustering_model, f)

    print("Training and saving the Classification model")
    input_dim = X.shape[1]
    output_dim = len(np.unique(y))
    classif_model = ClassificationModel(input_dim, output_dim)
    classif_model.train(X_train, y_train)
    with open(os.path.join(save_folder, 'classif.pkl'), 'wb') as f:
        pkl.dump(classif_model, f)

    print("Training and saving the Classification model with centroid descriptors")
    representation_X_train = clustering_model.compute_representation(X_train)
    input_dim = representation_X_train.shape[1]
    output_dim = len(np.unique(y))
    classif_model_repr = ClassificationModel(input_dim, output_dim)
    classif_model_repr.train(representation_X_train, y_train)
    with open(os.path.join(save_folder, 'classif-centroid.pkl'), 'wb') as f:
        pkl.dump(classif_model_repr, f)

if __name__ == "__main__":
    student_model_train()

