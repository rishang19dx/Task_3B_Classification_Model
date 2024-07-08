# Iris Flower Classification using K-Nearest Neighbors (KNN)

This task uses the K-Nearest Neighbors (KNN) algorithm to classify the Iris dataset. The Iris dataset is a classic dataset in machine learning and is often used for testing algorithms.

## Dataset

The Iris dataset consists of 150 samples from each of three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the lengths and the widths of the sepals and petals.

- **Number of samples**: 150
- **Number of features**: 4
  - Sepal length in cm
  - Sepal width in cm
  - Petal length in cm
  - Petal width in cm
- **Number of classes**: 3
  - Iris setosa
  - Iris versicolor
  - Iris virginica
- **Class distribution**: 50 samples per class

## Model

I have used the K-Nearest Neighbors (KNN) algorithm for this classification task. KNN is a simple and effective algorithm that classifies samples based on the majority class among the k nearest neighbors.

## Training Method

1. **Data Splitting**: The dataset is split into training (70%) and test (30%) sets.
2. **Model Training**: A KNN model with k=3 is trained on the training set.
3. **Evaluation**: The model is evaluated using the accuracy score on the test set.

## Results

The trained KNN model achieved an accuracy of 100.00% on the test set.

## Requirements

- Python 3.x
- scikit-learn

## Running the Code

To run the code, simply execute the script in a Python environment with the necessary libraries installed.

```bash
python classification.py
