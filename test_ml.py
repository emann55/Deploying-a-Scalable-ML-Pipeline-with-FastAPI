import pytest
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.model import train_model, compute_model_metrics
from ml.data import process_data

@pytest.fixture
def data():
    """Fixture to hold mock data for testing"""
    df = pd.DataFrame({
        "age": [25, 45, 30, 50],
        "workclass": ["Private", "Self-emp-not-inc", "Private", "Self-emp"],
        "education": ["Bachelors", "Masters", "Bachelors", "Doctorate"],
        "occupation": ["Adm-clerical", "Exec-managerial", "Adm-clerical", "Prof-specialty"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    })
    return df


def test_expected_data_types(data):
    """
    This unit test ensures that the function is returning the expected target data types from the process_data
    function. It uses mock data, and a subset of the categorical features to test the data against.
    """
    # Columns to process
    categorical_features = ["workclass", "occupation"]
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


def test_model_algorithm(data):
    """
    This unit test ensures that the data that is processed and trained on the correct algorithm, the Random Forest.
    """
    # Columns to process
    categorical_features = ["workclass", "education", "occupation"]

    # Process data
    X, y, _, _ = process_data(X=data, categorical_features=categorical_features, label="salary", training=True)

    # Train Data
    model = train_model(X, y)

    # Confirm that the model is using the RandomForest algorithm.
    assert isinstance(model, RandomForestClassifier)


def test_model_metrics_perfect():
    """
    This unit test ensures that predicted and true data that match up perfectly, that the computed metrics will reflect
    that, by returning a score of 1.0, the highest possible score. This test uses mocked data from a random seed of
    binary labels.
    """
    # generate a random spread of 20 binary labels for the score.
    seed = random.choices(range(0,1), k=20)

    # generate fake test and prediction data
    y_actual = np.array(seed)
    y_pred = np.array(seed)

    # test and prediction data are equal, therefore a perfect score, metrics should be all 1
    precision, recall, fbeta = compute_model_metrics(y_actual, y_pred)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
