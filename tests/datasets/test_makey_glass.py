import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from datasets import mackey_glass_generate


def test_makey_glass_genetate_lenght():
    """
    Test that the mackey_glass_generate function returns an array of the correct length.
    """
    n_timesteps = 100
    result = mackey_glass_generate(n_timesteps=n_timesteps)
    assert len(result) == n_timesteps, "The length of the generated time series should be equal to n_timesteps."


def test_makey_glass_generate_seed_consistency():
    """
    Test that the mackey_glass_generate function produces consistent results when the same seed is used.
    """
    n_timesteps = 100
    seed = 42
    result1 = mackey_glass_generate(n_timesteps=n_timesteps, seed=seed)
    result2 = mackey_glass_generate(n_timesteps=n_timesteps, seed=seed)
    np.testing.assert_array_equal(result1, result2, err_msg="Results should be consistent for the same seed.")


def test_mackey_glass_generate_default_parameters():
    """
    Test that the mackey_glass_generate function returns a numpy array with default parameters.
    """
    result = mackey_glass_generate(n_timesteps=100)
    assert isinstance(result, np.ndarray), "The result should be a numpy array."
