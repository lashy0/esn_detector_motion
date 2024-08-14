import torch
import torch.nn as nn

from typing import Optional

from .modules import Reservoir, Force


class ESNForce(nn.Module):
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
        feedback: bool = True,
        feedback_scaling: float = 0.1,
        noise: bool = False,
        noise_level: float = 0.01,
        lambda_: float = 0.99,
        delta: float = 1,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feedback = feedback
        
        self.reservoir = Reservoir(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            spectral_radius=spectral_radius,
            leaky_rate=leaky_rate,
            input_scaling=input_scaling,
            feedback_scaling=feedback_scaling,
            sparsity=sparsity,
            noise=noise,
            noise_level=noise_level,
            bias=bias,
            random_seed=random_seed,
            **self.factory_kwargs
        )
        
        self.output = Force(
            hidden_size=hidden_size,
            output_size=output_size,
            lambda_=lambda_,
            delta=delta,
            **self.factory_kwargs
        )
        
    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len, _ = X.size()
        
        outputs = torch.zeros((seq_len, self.hidden_size), **self.factory_kwargs)
        results = torch.zeros((seq_len, self.output_size), **self.factory_kwargs)
        
        for t in range(seq_len):
            if self.feedback:
                state = self.reservoir(X[t, :], results[t-1, :] if t > 0 else None)
            else:
                state = self.reservoir(X[t, :])
            outputs[t, :] = state
            result = self.output(state)
            results[t, :] = result
        
        return results, outputs

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        seq_len, _ = X.size()
        
        outputs = torch.zeros((seq_len, self.hidden_size), **self.factory_kwargs)
        results = torch.zeros((seq_len, self.output_size), **self.factory_kwargs)
        
        for t in range(seq_len):
            if self.feedback:
                state = self.reservoir(X[t, :], results[t-1, :] if t > 0 else None)
            else:
                state = self.reservoir(X[t, :])
            outputs[t, :] = state
            pred = self.output.fit(state, Y[t, :])
            results[t, :] = pred

        return results, outputs