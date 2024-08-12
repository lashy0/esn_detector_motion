import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Optional

# plt.style.use("fivethirtyeight")


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
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
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
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    ax.plot(np.arange(start, end), y_pred[start:end], lw=3, label="Predict")
    ax.plot(np.arange(start, end), y_true[start:end], lw=2, linestyle="--", label="True")
    ax.plot(np.arange(start, end), np.abs(y_true[start:end] - y_pred[start:end]), label="Absolute deviation")
    
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Predict vs True Values")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Values")
    
    ax.legend(loc="upper left", fancybox=False, edgecolor="black")
    plt.show()


def plot_reservoir_states(
    states: Union[np.ndarray, torch.Tensor],
    start_neuron: int = 0,
    end_neuron: int = 20
) -> None:
    """
    Plots the states Reservoir ESN of specified neurons over time.

    Args:
        states (Union[np.ndarray, torch.Tensor]): The reservoir states.
        start_neuron (int, optional): The index of the starting neuron to plot.
        end_neuron (int, optional): The index of the ending neuron to plot.
    """
    if not isinstance(states, (np.ndarray, torch.Tensor)):
        raise TypeError("states must be a numpy array or a PyTorch tensor.")
    
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()
    
    if not (isinstance(start_neuron, int) and isinstance(end_neuron, int)):
        raise TypeError("start_neuron and end_neuron must be integers.")
    
    if start_neuron < 0 or end_neuron > states.shape[1] or start_neuron >= end_neuron:
        raise ValueError("Invalid start_neuron or end_neuron values.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    for i in range(start_neuron, end_neuron):
        ax.plot(states[:, i])
    ax.set_title("Reservoir States Over Time")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Values")
    
    plt.show()
