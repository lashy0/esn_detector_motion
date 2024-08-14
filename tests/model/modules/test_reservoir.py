import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from model.modules import Reservoir
from model.utils import spectral_radius


@pytest.fixture
def setup_reservoir():
    """Fixture to initialize the Reservoir model for testing."""
    input_size = 10
    hidden_size = 20
    output_size = 5
    return Reservoir(input_size, hidden_size, output_size, spectral_radius=0.9, leaky_rate=0.2)


def test_initialization(setup_reservoir):
    """Test to ensure that the Reservoir model is initialized correctly."""
    model = setup_reservoir
    assert model.W_in.shape == (model.hidden_size, model.input_size), "W_in shape mismatch"
    assert model.W.shape == (model.hidden_size, model.hidden_size), "W shape mismatch"
    assert model.state.shape == (model.hidden_size,), "State shape mismatch"
    assert model.W_fb.shape == (model.hidden_size, model.output_size), "W_fb shape mismatch"

    if model.b_in is not None:
        assert model.b_in.shape == (model.hidden_size,), "Bias shape mismatch"

# TODO: доработать резервуар и тест
def test_forward_pass(setup_reservoir):
    """Test the forward pass of the Reservoir model."""
    model = setup_reservoir
    X = torch.randn(model.input_size)
    y = torch.randn(model.output_size)

    original_state = model.state.clone()

    output = model(X)
    assert output.shape[0] == (model.hidden_size), "Output shape mismatch in forward pass without feedback"
    assert not torch.equal(model.state, original_state), "State did not update in forward pass without feedback"

    output_with_feedback = model(X, y)
    assert output_with_feedback.shape[0] == (model.hidden_size), "Output shape mismatch in forward pass with feedback"


def test_reset_state(setup_reservoir):
    """Test the reset_state method to ensure the reservoir state is correctly reset to zeros."""
    model = setup_reservoir
    model.state = torch.randn(model.hidden_size)
    model.reset_state()
    assert torch.all(model.state == 0), "State did not reset to zeros"


def test_weight_initialization(setup_reservoir):
    """Test the initialization of reservoir weights with correct spectral radius and sparsity."""
    model = setup_reservoir

    non_zero_elements = torch.count_nonzero(model.W)
    total_elements = model.W.numel()
    calculated_sparsity = non_zero_elements / total_elements
    assert abs(calculated_sparsity - model.sparsity) < 0.05, "Sparsity of W matrix is incorrect"

    rhow = spectral_radius(model.W)
    assert abs(rhow - model.spectral_radius) < 0.1, "Spectral radius of W matrix is incorrect"


def test_device_handling():
    """Tests the Reservoir model on both CPU and GPU to ensure correct device handling."""
    input_size = 10
    hidden_size = 20
    output_size = 5

    X = torch.randn(1, input_size)
    y = torch.randn(1, output_size)

    # Test on CPU
    model_cpu = Reservoir(input_size, hidden_size, output_size)
    output_cpu = model_cpu(X, y)
    assert output_cpu.device == torch.device('cpu'), "Model is not on CPU"

    # Test on GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_gpu = Reservoir(input_size, hidden_size, output_size, device=device)
        X_gpu = X.to(device)
        y_gpu = y.to(device)
        output_gpu = model_gpu(X_gpu, y_gpu)
        assert output_gpu.device == device, "Model is not on GPU"
        assert model_gpu.W.device == device, "W is not on GPU"
        if model_gpu.b_in is not None:
            assert model_gpu.b_in.device == device, "Bias is not on GPU"