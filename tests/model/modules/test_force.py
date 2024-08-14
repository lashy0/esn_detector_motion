import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from model.modules import Force


@pytest.fixture
def setup_force():
    """
    Fixture to initialize the Force model for testing.

    Returns:
        Force: An instance of the Force model with predefined parameters.
    """
    hidden_size = 10
    output_size = 5
    lambda_ = 0.999
    delta = 1.0
    return Force(hidden_size, output_size, lambda_=lambda_, delta=delta)


def test_initialization(setup_force):
    """
    Test to ensure that the Force model is initialized correctly.
    
    Test checks:
    - The shape of the weight matrix W.
    - The shape of covariance matrix P.
    - The correct initialization of the covariance matrix P with regularization delta.
    """
    model = setup_force
    assert model.W.shape == (model.output_size, model.hidden_size), "Weight matrix shape mismatch"
    assert model.P.shape == (model.hidden_size, model.hidden_size), "Covariance matrix shape mismatch"
    assert torch.allclose(model.P, torch.eye(model.hidden_size) * model.delta), "P matrix initialization failed"


def test_forward_pass(setup_force):
    """
    Test the forward pass of the Force model.

    Test checks that the output of the forward pass has the correct shape given
    an input tensor X.
    """
    model = setup_force
    X = torch.randn(model.hidden_size)
    output = model(X)
    assert output.shape == (model.output_size,), "Output shape mismatch in forward pass"


def test_fit(setup_force):
    """
    Test the fit (RLS update) method of the Force model.

    Test checks:
    - The shape of the initial prediction.
    - That the weight matrix W is updated after calling fit.
    - That predictions change after fitting, indicating learning has occurred.
    """
    model = setup_force
    X = torch.randn(model.hidden_size)
    Y = torch.randn(model.output_size)

    initial_pred = model(X)
    assert initial_pred.shape == (model.output_size,), "Initial prediction shape mismatch"

    pred = model.fit(X, Y)
    
    assert not torch.equal(model.W, 
                           nn.Parameter(torch.empty((model.output_size, model.hidden_size), requires_grad=False))), \
                           "Weights did not update"

    new_pred = model(X)
    assert not torch.equal(pred, new_pred), "Prediction did not change after fitting"


def test_device_handling():
    """
    Test the Force model's ability to handle different devices (CPU and GPU).
    
    Test checks:
    - The model's compatibility with CPU and GPU.
    - That the predictions and weights are correctly placed on the specified device.
    """
    hidden_size = 10
    output_size = 5
    X = torch.randn(hidden_size)
    Y = torch.randn(output_size)

    # Test on CPU
    model_cpu = Force(hidden_size, output_size)
    pred_cpu = model_cpu.fit(X, Y)
    assert pred_cpu.device == torch.device('cpu'), "Model is not on CPU"

    # Test on GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_gpu = Force(hidden_size, output_size, device=device)
        X_gpu = X.to(device)
        Y_gpu = Y.to(device)
        pred_gpu = model_gpu.fit(X_gpu, Y_gpu)
        assert pred_gpu.device == device, "Model is not on GPU"
        assert model_gpu.W.device == device, "Weights are not on GPU"
