import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class Force(nn.Module):
    """
    Implementation of a FORCE output layer using Recursive Least Squares (RLS)
    for real-time learning in ESN.
    
    The FORCE learning algorithm is designed to adjust output weights
    dynamically in response to the error between predicted and actual outputs,
    ensuring stability and accurate preictions.

    Args:
        hidden_size (int): Number of neurons in the reservoir (hidden layer).
        output_size (int): Number of output features.
        lambda_ (float): Forgetting factor in RLS, typically close to 1.0.
        Higher values result in slower adaptation.
        delta (float): Regularization parameter for initializing the covariance matrix `P`.
        device (Optional[torch.device], optional): Device to run the model on.
        dtype (Optional[torch.dtype], optional): Data type for the model parameters.
    """
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        lambda_: float = 0.99,
        delta: float = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.delta = delta
        self.lambda_ = lambda_
        
        for name, value in [("hidden_size", hidden_size), ("output_size", output_size)]:
            if not isinstance(value, int):
                raise TypeError(f"{name} should be of type int, got: {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        
        self.W = nn.Parameter(torch.empty((output_size, hidden_size), **self.factory_kwargs), requires_grad=False)
        
        self.P = None
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize parameters of the force."""
        nn.init.uniform_(self.W, -1, 1)
        
        self.P = torch.eye(self.hidden_size, **self.factory_kwargs) * self.delta
       
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pred = F.linear(X, self.W)
        return pred
    
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Update the output weights using RLS algorithm based on the error
        between the predicted and target outputs.

        Args:
            X (torch.Tensor): Input tensor representing the reservoir states, shape (hidden_size,).
            Y (torch.Tensor): Target output tensor, shape (output_size,).

        Returns:
            torch.Tensor: Predicted output tensor before updating the wights, shape (output_size,)
        """
        pred = self.forward(X)
        error = Y - pred
            
        # RLS
        Px = torch.matmul(self.P, X)
        gain = Px / (self.lambda_ + torch.matmul(X, Px))
        
        if gain.dim() == 0:
            gain = gain.unsqueeze(0)
        if error.dim() == 0:
            error = error.unsqueeze(0)
        
        self.P = (self.P - torch.outer(gain, Px)) / self.lambda_
        
        self.W += torch.outer(error, gain)
        
        return pred
    
    def extra_repr(self) -> str:
        ...
