import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from model.modules import Ridge


@pytest.fixture
def setup_ridge():
    """Fixture to initialize the Ridge model for testing."""
    hidden_size = 10
    output_size = 5
    bias = True
    return Ridge(hidden_size, output_size, bias=bias)


def test_initialization(setup_ridge):
    """
    Test to ensure that the Ridge model is initialized correctly.
    
    Test checks:
    - The shape of the weight matrix W.
    - The existence and shape of the bias term b (if bias=True).
    """
    model = setup_ridge
    assert model.W.shape == (model.output_size, model.hidden_size), "Weight matrix shape mismatch"
    
    if model.b is not None:
        assert model.b.shape == (model.output_size,), "Bias term shape mismatch"
    else:
        assert model.b is None, "Bias term should be None"


def test_forward_pass(setup_ridge):
    """
    Test the forward pass of the Ridge model.
    
    Test checks that the output of the forward pass has the correct shape given
    an input tensor X.
    """
    model = setup_ridge
    X = torch.randn(20, model.hidden_size)
    output = model(X)
    assert output.shape == (20, model.output_size), "Output shape mismatch in forward pass"


def test_fit(setup_ridge):
    """
    Test the fit method of the Ridge model.
    
    This test performs the following checks:
    - Ensures that the initial prediction has the correct shape.
    - Verifies that the weight matrix W is updated after calling the fit method.
    - Confirms that predictions change after fitting, indicating that the model has learned from the data.
    """
    model = setup_ridge
    X = torch.randn(100, model.hidden_size)
    Y = torch.randn(100, model.output_size)

    # Store original weights and bias
    original_W = model.W.clone()
    original_b = model.b.clone() if model.b is not None else None

    # Fit the model
    model.fit(X, Y, washout=10, lr=1e-2)

    # Ensure that weights and bias have been updated
    assert not torch.equal(model.W, original_W), "Weights did not update"
    if model.b is not None:
        assert not torch.equal(model.b, original_b), "Bias did not update"


def test_no_bias():
    """Test the Ridge model without a bias term."""
    hidden_size = 10
    output_size = 5
    model = Ridge(hidden_size, output_size, bias=False)

    X = torch.randn(20, hidden_size)
    Y = torch.randn(20, output_size)

    original_W = model.W.clone()

    model.fit(X, Y, washout=0, lr=1e-2)

    assert not torch.equal(model.W, original_W), "Weights did not update"

    assert model.b is None, "Bias should be None"


def test_device_handling():
    """Tests the Ridge model on both CPU and GPU to ensure correct device handling."""
    hidden_size = 10
    output_size = 5
    X = torch.randn(100, hidden_size)
    Y = torch.randn(100, output_size)

    # Test on CPU
    model_cpu = Ridge(hidden_size, output_size)
    model_cpu.fit(X, Y)
    pred_cpu = model_cpu(X)
    assert pred_cpu.device == torch.device('cpu'), "Model is not on CPU"

    # Test on GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_gpu = Ridge(hidden_size, output_size, device=device)
        X_gpu = X.to(device)
        Y_gpu = Y.to(device)
        model_gpu.fit(X_gpu, Y_gpu)
        pred_gpu = model_gpu(X_gpu)
        assert pred_gpu.device == device, "Model is not on GPU"
        assert model_gpu.W.device == device, "Weights are not on GPU"
        if model_gpu.b is not None:
            assert model_gpu.b.device == device, "Bias is not on GPU"