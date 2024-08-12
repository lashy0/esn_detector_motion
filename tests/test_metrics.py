import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from utils.metrics import mse, rmse, nrmse, rsquare, _to_numpy, _check_arrays


def test_to_numpy():
    """
    Test the _to_numpy function with both numpy array and torch tensor.
    """
    np_array = np.array([1, 2, 3])
    torch_tensor = torch.tensor([1, 2, 3])
    
    assert np.array_equal(_to_numpy(np_array), np_array)
    assert np.array_equal(_to_numpy(torch_tensor), np_array)


def test_check_arrays():
    """
    Test the _check_arrays function for shape mismatch and correct conversion.
    """
    np_array1 = np.array([1, 2, 3])
    np_array2 = np.array([1, 2, 3])
    torch_tensor1 = torch.tensor([1, 2, 3])
    torch_tensor2 = torch.tensor([1, 2, 3])

    y_true, y_pred = _check_arrays(np_array1, np_array2)
    assert np.array_equal(y_true, np_array1)
    assert np.array_equal(y_pred, np_array2)

    y_true, y_pred = _check_arrays(torch_tensor1, torch_tensor2)
    assert np.array_equal(y_true, np_array1)
    assert np.array_equal(y_pred, np_array2)

    with pytest.raises(ValueError):
        _check_arrays(np_array1, np.array([1, 2]))


def test_mse():
    """
    Test the mse function with simple inputs.
    """
    np_array1 = np.array([1, 2, 3])
    np_array2 = np.array([1, 2, 3])
    assert mse(np_array1, np_array2) == 0.0

    np_array3 = np.array([1, 2, 3])
    np_array4 = np.array([4, 5, 6])
    assert mse(np_array3, np_array4) == 9.0


def test_rmse():
    """
    Test the rmse function with simple inputs.
    """
    np_array1 = np.array([1, 2, 3])
    np_array2 = np.array([1, 2, 3])
    assert rmse(np_array1, np_array2) == 0.0

    np_array3 = np.array([1, 2, 3])
    np_array4 = np.array([4, 5, 6])
    assert rmse(np_array3, np_array4) == 3.0


def test_nrmse():
    """
    Test the nrmse function with various normalization methods.
    """
    np_array1 = np.array([1, 2, 3])
    np_array2 = np.array([1, 2, 3])
    assert nrmse(np_array1, np_array2) == 0.0

    np_array3 = np.array([1, 2, 3])
    np_array4 = np.array([4, 5, 6])
    assert nrmse(np_array3, np_array4, norm='range') == 1.5
    assert nrmse(np_array3, np_array4, norm='std') == 3.0 / np.std(np_array3)
    assert nrmse(np_array3, np_array4, norm='mean') == 3.0 / np.mean(np_array3)
    assert nrmse(np_array3, np_array4, norm='q1q3') == 3.0 / (np.quantile(np_array3, 0.75) - np.quantile(np_array3, 0.25))


def test_rsquare():
    """
    Test the rsquare function with simple inputs.
    """
    np_array1 = np.array([1, 2, 3])
    np_array2 = np.array([1, 2, 3])
    assert rsquare(np_array1, np_array2) == 1.0

    np_array3 = np.array([1, 2, 3])
    np_array4 = np.array([4, 5, 6])
    assert rsquare(np_array3, np_array4) == -12.5
