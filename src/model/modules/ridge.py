import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class Ridge(nn.Module):
    """
    Implementation of a Ridge regression for training output layers in ESN.
    
    This module performs ridge regression on input features to fot the output
    weights, minimizing the squared error with an L2 reguarlization term to
    prevent overfitting.

    Args:
        hidden_size (int): Number of neurons in the reservoir (hidden layer).
        output_size (int): Number of output features.
        bias (bool): If True, includes a bias term in the regression.
        device (Optional[torch.device], optional): Device to run the model on.
        dtype (Optional[torch.dtype], optional): Data type for the model parameters.
    """
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        for name, value in [("hidden_size", hidden_size), ("output_size", output_size)]:
            if not isinstance(value, int):
                raise TypeError(f"{name} should be of type int, got: {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        
        self.W = nn.Parameter(torch.empty((output_size, hidden_size), **self.factory_kwargs), requires_grad=False)
        if bias:
            self.b = nn.Parameter(torch.empty(output_size, **self.factory_kwargs), requires_grad=False)
        else:
            self.register_parameter("b", None)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters of the ridge."""
        nn.init.uniform_(self.W, -1, 1)
        if self.b is not None:
            nn.init.uniform_(self.b, -1, 1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = F.linear(X, self.W, self.b)
        return out
    
    def fit(self, X: torch.Tensor, Y: torch.Tensor, washout: Optional[int] = 0, lr: float = 1e-6) -> None:
        """
        Fit the Ridge regression model to the provided data.
        
        This method uses the closed-form solution to Ridge regression, incorporating
        a regularization term to prevent overfitting.

        Args:
            X (torch.Tensor): Input tensor representing the reservoir states, shape (time_steps, hidden_size).
            Y (torch.Tensor): Target output tensor, shape (time_steps, output_size).
            washout (int): Number of initial time steps to discard in fitting.
            lr (float): Regularization strength.
        """
        X = X[washout:]
        Y = Y[washout:]
        
        if self.b is not None:
            ones = torch.ones(X.size(0), 1, **self.factory_kwargs)
            X = torch.cat([ones, X], dim=1)
        
        XTX = torch.matmul(X.T, X)
        XTY = torch.matmul(X.T, Y)
        
        I_eye = torch.eye(XTX.size(0), **self.factory_kwargs)
        ridge_term = I_eye * lr
        
        if self.b is not None:
            ridge_term[0, 0] = 0
        
        res = torch.linalg.solve(XTX + ridge_term, XTY)
        
        if self.b is not None:
            self.W.data = res[1:].T
            self.b.data = res[0]
        else:
            self.W.data = res.T
    
    def extra_repr(self) -> str:
        ...
