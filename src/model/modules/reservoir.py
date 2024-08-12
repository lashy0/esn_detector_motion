import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from ..utils import spectral_radius


class Reservoir(nn.Module):
    """
    Reservoir layer for Echo State Networks.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of reservoir features.
        output_size (int): Number of output features.
        spectral_radius (float, optional): Desired spectral radius of the reservoir weight matrix.
        leaky_rate (float, optional): Leaky integration rate.
        input_scaling (float, optional): Scaling factor for input weights.
        sparsity (float, optional): Proportion of non-zero connections in the reservoir.
        bias (bool, optional): Includes a bias term in the input transformation
        random_seed (Optional[int], optional): Seed for random number generation.
        device (Optional[torch.device], optional): Device to run the model on.
        dtype (Optional[torch.dtype], optional): Data type for the model parameters.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        spectral_radius: float = 0.95,
        leaky_rate: float = 1.0,
        input_scaling: float = 1.0,
        sparsity: float = 0.1,
        bias: bool = True,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.leaky_rate = leaky_rate
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        for name, value in [("input_size", input_size), ("hidden_size", hidden_size), ("output_size", output_size)]:
            if not isinstance(value, int):
                raise TypeError(f"{name} should be of type int, got: {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        
        if not (0 < leaky_rate <= 1.0):
            raise ValueError("leaky_rate must be in the range (0, 1]")
        if spectral_radius <= 0:
            raise ValueError("spectral_radius must be positive")
        if not (0 < sparsity <= 1.0):
            raise ValueError("sparsity must be in the range (0, 1]")
        
        self.W_in = nn.Parameter(torch.empty((hidden_size, input_size), **self.factory_kwargs), requires_grad=False)
        if bias:
            self.b_in = nn.Parameter(torch.empty(hidden_size, **self.factory_kwargs), requires_grad=False)
        else:
            self.register_parameter("b_in", None)
        self.W = nn.Parameter(torch.empty((hidden_size, hidden_size), **self.factory_kwargs), requires_grad=False)
        self.W_fb = nn.Parameter(torch.empty((hidden_size, output_size), **self.factory_kwargs), requires_grad=False)
        self.reset_parameters()
        
        self.state = torch.zeros(hidden_size, **self.factory_kwargs)
    
    # TODO: сделать разные способы инициализации весов
    def reset_parameters(self) -> None:
        """Initialize parameters of the reservoir."""
        nn.init.uniform_(self.W_in, -self.input_scaling, self.input_scaling)
        if self.b_in is not None:
            nn.init.uniform_(self.b_in, -self.input_scaling, self.input_scaling)
        self._initialize_reservoir_weights()
        nn.init.normal_(self.W_fb, mean=0, std=1)
    
    def _initialize_reservoir_weights(self) -> None:
        """Initialize the reservoir weight matrix with the desired spectral radius and sparsity."""
        nn.init.normal_(self.W, mean=0, std=1)
        mask = torch.rand(self.W.shape, **self.factory_kwargs) < self.sparsity
        self.W.data *= mask
        rhow = spectral_radius(self.W)
        self.W.data *= (self.spectral_radius / rhow)
    
    def reset_state(self) -> None:
        """Reset the state of the reservoir to zeros."""
        self.state.zero_()
    
    # принимает 1D тензор + прошлый результат 1D тензор для feedback
    def forward(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        pre_state = F.linear(X, self.W_in, self.b_in) + torch.matmul(self.W, self.state)
        if y is not None:
            pre_state += torch.matmul(self.W_fb, y)
        
        self.state = (1.0 - self.leaky_rate) * self.state + self.leaky_rate * F.tanh(pre_state)
        
        return self.state
    
    def extra_repr(self) -> str:
        ...
