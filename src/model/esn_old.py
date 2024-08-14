import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any


class Reservoir(nn.Module):
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        rhow: float = 0.95,
        sparsity: float = 0.1,
        leaky_rate: float = 0.1,
        input_scaling: float = 1.0
    ) -> None:
        """Echo State Network (ESN) Reservoir.

        Args:
            input_size (int): Dimensionality of the input.
            reservoir_size (int): Number of neurons in the reservoir.
            rhow (float, optional): Desired spectral radius of the reservoir weight matrix.
            sparsity (float, optional): Fraction of non-zero connections in the reservoir weight matrix.
            leaky_rate (float, optional): Leaky integration rate.
            input_scaling (float, optional): Scaling factor for the input weight matrix.
        """
        super(Reservoir, self).__init__()
        
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.rhow = rhow
        self.sparsity = sparsity
        self.leaky_rate = leaky_rate
        self.input_scaling = input_scaling
        
        self.state = torch.zeros(reservoir_size)
        
        self.W_in = nn.Parameter(
            torch.empty((reservoir_size, input_size)).uniform_(-1, 1) * input_scaling,
            requires_grad=False
        )
        
        self.W = torch.rand((reservoir_size, reservoir_size)) * 2.0 - 1.0
        mask = torch.rand((reservoir_size, reservoir_size)) < sparsity
        self.W = torch.where(mask, self.W, torch.zeros_like(self.W))
        self.W *= (rhow / self.spectral_radius)
        self.W = nn.Parameter(
            self.W,
            requires_grad=False
        )
    
    @property    
    def spectral_radius(self) -> float:
        """Calculate the spectral radius of the reservoir weight matrix.

        Returns:
            float: The largest absolute eigenvalue of the reservoir weight matrix.
        """
        return torch.abs(torch.linalg.eigvals(self.W)).max().item()
    
    def reset_state(self) -> None:
        """Reset the reservoir state to zero."""
        self.state = torch.zeros(self.reservoir_size)
    
    def forward(self, x: torch.Tensor, init_state: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the reservoir.

        Args:
            x (torch.Tensor): Input tensor of shape (n_timesteps, input_size).
            init_state (torch.Tensor, optional): Initial state of the reservoir.

        Returns:
            torch.Tensor: Output tensor of shape (n_timesteps, reservoir_size).
        """
        if init_state is not None:
            self.state = init_state
            
        n_timesteps, _ = x.size()
        
        outputs = torch.zeros((n_timesteps, self.reservoir_size))
        
        for t in range(n_timesteps):
            xt = x[t, :]
            state = F.tanh((xt @ self.W_in.T) + (self.state @ self.W.T))
            self.state = (1.0 - self.leaky_rate) * self.state + self.leaky_rate * state
            
            outputs[t, :] = self.state
            
        return outputs

class ESN(nn.Module):
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        rhow: float = 1.25,
        sparsity: float = 0.5,
        leaky_rate: float = 0.3,
        input_scaling: float = 1.0
    ) -> None:
        """Echo State Network (ESN) with a reservoir and readout layer.

        Args:
            input_size (int): Dimensionality of the input.
            reservoir_size (int): Number of neurons in the reservoir.
            output_size (int): Dimensionality of the output.
            rhow (float, optional): Desired spectral radius of the reservoir weight matrix.
            sparsity (float, optional): Fraction of non-zero connections in the reservoir weight matrix.
            leaky_rate (float, optional): Leaky integration rate.
            input_scaling (float, optional): Scaling factor for the input weight matrix.
        """
        super(ESN, self).__init__()
        
        self.reservoir = Reservoir(
            input_size=input_size,
            reservoir_size=reservoir_size,
            rhow=rhow,
            sparsity=sparsity,
            leaky_rate=leaky_rate,
            input_scaling=input_scaling
        )
        
        self.W_out = nn.Parameter(
            torch.empty((reservoir_size, output_size)).uniform_(-1, 1),
            requires_grad=False
        )
    
    def reset_reservoir(self) -> None:
        """Reset the reservoir state to zero."""
        self.reservoir.reset_state()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
        """Forward pass through the ESN.

        Args:
            x (torch.Tensor): Input tensor of shape (n_timesteps, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (n_timesteps, output_size) and the final reservoir state.
        """
        state = self.reservoir(x)
        output = torch.matmul(state, self.W_out)
        return output, state
    
    # Ridge Regression
    def fit(self, x: torch.Tensor, y: torch.Tensor, ridge_lambda: float) -> None:
        """Fit the readout layer using Ridge Regression.

        Args:
            x (torch.Tensor): Input tensor of shape (n_timesteps, input_size).
            y (torch.Tensor): Target tensor of shape (n_timesteps, output_size).
            ridge_lambda (float): Regularization parameter for Ridge Regression.
        """
        state = self.reservoir(x)
        
        identity = torch.eye(state.shape[1])
        X_transpose_X = torch.matmul(state.T, state) + ridge_lambda * identity
        X_transpose_Y = torch.matmul(state.T, y)
        
        self.W_out.data = torch.linalg.solve(X_transpose_X, X_transpose_Y)
    
    # Online learning with LMS
    def update_redaut_lms(self, x: torch.Tensor, y: torch.Tensor, learning_rate: float) -> None:
        """Update the readout layer using Least Mean Squares (LMS) online learning.

        Args:
            x (torch.Tensor): Input tensor of shape (n_timesteps, input_size).
            y (torch.Tensor): Target tensor of shape (n_timesteps, output_size).
            learning_rate (float): Learning rate for the LMS update.
        """
        state = self.reservoir(x)
        pred_out = torch.matmul(state, self.W_out)
        error = y - pred_out
        self.W_out += learning_rate * torch.matmul(state.T, error)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the output for the given input.

        Args:
            x (torch.Tensor): Input tensor of shape (n_timesteps, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (n_timesteps, output_size).
        """
        out, _ = self.forward(x)
        return out