import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional

plt.style.use("fivethirtyeight")


# TODO: доработать 
def plot_result(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    start: int = 0,
    length: int = 500,
    batch_size: Optional[int] = None
) -> None:
    """
    Plots the predicted and true values over a specified range.

    Args:
        y_pred (Union[np.ndarray, torch.Tensor]): Predicted values.
        y_true (Union[np.ndarray, torch.Tensor]): True values.
        start (int, optional): Starting index for plotting.
        length (int, optional): Number of points to plot.
        batch_size (Optional[int], optional): Index of the batch to plot if input arrays are 3D.
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().numpy()
    
    if y_pred.ndim not in {2, 3} or y_true.ndim not in {2, 3}:
        raise ValueError("Input arrays must be 2D or 3D")
    
    if y_pred.shape != y_true.shape:
        raise ValueError("Shape of y_pred does not match shape of y_true")
    
    if y_pred.ndim == 3:
        if batch_size is None:
            y_pred = y_pred[0, :, :]
            y_true = y_true[0, :, :]
        else:
            if batch_size >= y_pred.shape[0]:
                raise ValueError("Batch size exceeds available batches in the input arrays")
            y_pred = y_pred[batch_size, :, :]
            y_true = y_true[batch_size, :, :]
    
    end = start + length
    if start < 0:
        raise ValueError("Start index must be non-negative")
    if end > y_pred.shape[0]:
        end = y_pred.shape[0]
        print(f"Adjusted end index to {end} as it exceeded array dimensions")
    
    fig, ax = plt.subplot(1, 1, figsize=(12, 4))
    
    ax.plot(np.arange(start, end), y_pred[start:end], lw=3, label="Predict")
    ax.plot(np.arange(start, end), y_true[start:end], lw=2, linestyle="--", label="True")
    ax.plot(np.arange(start, end), np.abs(y_true[start:end] - y_pred[start:end]), label="Absolute deviation")
    
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Predict vs True Values")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Values")
    
    plt.legend(loc="upper left", fancybox=False, edgecolor="black")
    plt.show()
