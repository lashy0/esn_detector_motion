import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from datasets import to_forecasting


@pytest.fixture(scope="module")
def data():
    """
    Fixture that generates a random dataset for testing.

    Returns:
        np.ndarray: A 100x5 array of random float numbers between 0 and 1.
    """
    return np.random.rand(100, 5)


def test_to_forecasting(data: np.ndarray):
    """
    Test that the to_forecasting function with a default. 

    Args:
        data (np.ndarray): Input data from the fixture.
    """
    res = to_forecasting(data, 3)
    
    assert len(res) == 2
    X, y = res
    assert X.shape == (97, 5)
    assert y.shape == (97, 5)


def test_to_forecasting_test_size(data: np.ndarray):
    """
    Test that the to_forecasting function with a specified test size.

    Args:
        data (np.ndarray): Input data from the fixture.
    """
    res = to_forecasting(data, 3, test_size=0.2)
    
    assert len(res) == 4
    X, X_t, y, y_t = res
    assert X.shape == (77, 5)
    assert X_t.shape == (20, 5)
    assert y.shape == (77, 5)
    assert y_t.shape == (20, 5)


def test_to_forecasting_invalid_forecast(data: np.ndarray):
    """
    Test that the to_forecasting function with an invalid forecast size.

    Args:
        data (np.ndarray): Input data from the fixture.
    """
    with pytest.raises(ValueError):
        to_forecasting(data, -1)


def test_to_forecasting_invalid_test_size(data: np.ndarray):
    """
    Test that the to_forecasting fuction with an invalid test size.

    Args:
        data (np.ndarray): Input data from the fixture.
    """
    with pytest.raises(ValueError):
        to_forecasting(data, 3, test_size=1.5)


def test_to_forecasting_large_test_size(data: np.ndarray):
    """
    Test that the to_forecasting function with a very large test size.

    Args:
        data (np.ndarray): Input data from the fixture.
    """
    with pytest.raises(ValueError):
        to_forecasting(data, 3, test_size=0.99)
