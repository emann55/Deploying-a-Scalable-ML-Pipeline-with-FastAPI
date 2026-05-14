import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

# TODO: implement the first test. Change the function name and input as needed
def test_expected_data_types():
    """
    # add description for the first test
    """
    # Ingest Data
    data = pd.read_csv(
        "/mnt/c/Users/Bacon/PycharmProjects/WGUD501/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/data/census.csv")

    # Columns to process
    categorical_features = ["workclass", "education", "marital-status"]
    X, y, encoder, lb = process_data(
        X=data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Check that returns are the correct data types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

    # Check that X and y are correct shapes
    # Y should additionally be a 1 dimensional array
    assert len(y.shape) == 1
    # X should contain the same number of rows as the input data
    assert X.shape[0] == len(data)


# TODO: implement the second test. Change the function name and input as needed
def test_model_algorithm():
    """
    # add description for the second test
    """
    # Ingest Data
    data = pd.read_csv(
        "/mnt/c/Users/Bacon/PycharmProjects/WGUD501/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/data/census.csv")

    # Columns to process
    categorical_features = ["workclass", "education", "marital-status"]

    # Process data
    X, y, _, _ = process_data(X=data, categorical_features=categorical_features, label="salary", training=True)
    # Train Data
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
