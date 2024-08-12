import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class Ridge(nn.Module):
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
        nn.init.uniform_(self.W, -1, 1)
        if self.b is not None:
            nn.init.uniform_(self.b, -1, 1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out = F.linear(X, self.W, self.b)
        return out
    
    def fit(self, X: torch.Tensor, Y: torch.Tensor, washout: int = 0, lr: float=1e-6) -> None:
        X = X[washout:]
        Y = Y[washout:]
        
        if self.b is not None:
            ones = torch.ones(X.size(0), 1, **self.factory_kwargs)
            X = torch.cat([ones, X], dim=1)
        
        XTX = torch.matmul(X.T, X)
        XTY = torch.matmul(X.T, Y)
        
        I = torch.eye(XTX.size(0), **self.factory_kwargs)
        ridge_term = I * lr
        
        if self.b is not None:
            ridge_term[0, 0] = 0
        
        res = torch.linalg.solve(XTX + ridge_term, XTY)
        
        if self.b is not None:
            self.W.data = res[1:].T
            self.b.data = res[0]
        else:
            self.W.data = res.T
