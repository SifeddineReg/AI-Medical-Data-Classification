# Machine Learning Models for Breast Cancer Classification

This project focuses on the development of machine learning models to classify breast cancer tumors into malignant (M) or benign (B) based on features extracted from digitized images of fine needle aspirate (FNA) of breast mass. The project utilizes the Wisconsin Diagnostic Breast Cancer (WDBC) dataset and implements a clustering model alongside a K-Nearest Neighbors (KNN) classifier for the classification task.

## Project Structure

The project is structured as follows:

- `data/`: Contains the WDBC dataset files.
  - `wdbc.data`: The dataset file.
  - `wdbc.names`: Describes the dataset attributes.
- `knn.py`: Implementation of the K-Nearest Neighbors algorithm (not detailed in the provided excerpts).
- `model.py`: Script for training and saving the machine learning models.
- `models.py`: Contains the implementation of the `ClusteringModel`, `ClassificationModel`, and utility functions.
- `models/`: Directory where trained models are saved.
  - `classif-centroid.pkl`: Classification model trained with centroid descriptors.
  - `classif.pkl`: Basic classification model.
  - `clustering.pkl`: Clustering model.

## Models

### ClusteringModel

The `ClusteringModel` is designed to perform clustering on the dataset to find a specified number of clusters. It is used as a preprocessing step for the classification model to transform the input data into a representation based on cluster centroids.

### ClassificationModel

The `ClassificationModel` combines the `ClusteringModel` with a KNN classifier for the task of breast cancer classification. It supports training with the original feature set as well as with the transformed feature set based on cluster centroids.

## Usage

To train and save the models, run the `model.py` script. This script performs the following steps:

1. Loads the WDBC dataset.
2. Encodes the target labels into integers.
3. Splits the dataset into training and test sets.
4. Trains and saves the `ClusteringModel`.
5. Trains and saves the `ClassificationModel` using both the original and centroid-based feature representations.

Ensure that you have the required dependencies installed before running the script:

- numpy
- pandas
- pickle

## Dataset

The Wisconsin Diagnostic Breast Cancer (WDBC) dataset consists of features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Each instance has 30 numeric attributes and a class label indicating the diagnosis (M = malignant, B = benign).

## License

This project is open-sourced under the MIT License.