import torch
import numpy as np

from typing import Union, Literal


def _to_numpy(y: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert input array to numpy array.

    Args:
        y: Union[np.ndarray, torch.Tensor]
            Input array to be converted.

    Returns:
        np.ndarray
            Converted numpy array.
    """
    if isinstance(y, torch.Tensor):
        return y.numpy()
    return np.asarray(y)


def _check_arrays(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> tuple:
    """
    Check and convert input arrays to numpy arrays.

    Args:
        y_true: Union[np.ndarray, torch.Tensor]
            Correct target values.
        y_pred: Union[np.ndarray, torch.Tensor]
            Estimated target values.

    Returns:
        tuple
            Converted numpy array of y_true and y_pred.
    """
    y_true_array = _to_numpy(y_true)
    y_pred_array = _to_numpy(y_pred)
    
    if not y_true_array.shape == y_pred_array.shape:
        raise ValueError(
            "Shape mismatch between y_true and y_pred: "
            f"{y_true_array.shape} != {y_pred_array.shape}"
        )
    
    return y_true_array, y_pred_array


def mse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute the Mean Squared Error (MSE) between the ground truth and predicted values.

    Args:
        y_true: Union[np.ndarray, torch.Tensor]
            Correct target values.
        y_pred: Union[np.ndarray, torch.Tensor]
            Estimated target values.

    Returns:
        float
            Mean Squared Error between y_true and y_pred.
    """
    y_true_array, y_pred_array = _check_arrays(y_true, y_pred)
    return float(np.mean((y_true_array - y_pred_array) ** 2))


def rmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between the ground truth and predicted values.

    Args:
        y_true: Union[np.ndarray, torch.Tensor]
            Correct target values.
        y_pred: Union[np.ndarray, torch.Tensor]
            Estimated target values.

    Returns:
        float
            Root Mean Squared Error between y_true and y_pred.
    """
    return np.sqrt(mse(y_true, y_pred))


def nrmse(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    norm: Literal["range", "std", "mean", "q1q3"] = "range",
    norm_value: float = None
) -> float:
    """
    Compute the Normalized Root Mean Squared Error (NRMSE) between the ground truth and predicted values.

    Args:
        y_true: Union[np.ndarray, torch.Tensor]
            Correct target values.
        y_pred: Union[np.ndarray, torch.Tensor]
            Estimated target values.
        norm: Literal[&quot;range&quot;, &quot;std&quot;, &quot;mean&quot;, &quot;q1q3&quot;], optional)
            Normalization method.
        norm_value: float, optional
            A normalization factor.

    Returns:
        float
            Normalized Root Mean Squared Error between y_true and y_pred.
    """
    error = rmse(y_true, y_pred)
    if norm_value is not None:
        return error / norm_value
    else:
        norms = {
            "range": lambda y: y.ptp(),
            "std": lambda y: y.std(),
            "mean": lambda y: y.mean(),
            "q1q3": lambda y: np.quantile(y, 0.75) - np.quantile(y, 0.25),
        }
        
        if norms.get(norm) is None:
            raise ValueError(
                "Unknown normalization method. "
                f"Available methods are {list(norms.keys())}."
            )
        else:
            y_true_array = _to_numpy(y_true)
            return error / norms[norm](y_true_array)


def rsquare(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute the R-squared (coefficient of determination) regression score function.

    Args:
        y_true: Union[np.ndarray, torch.Tensor]
            Correct target values.
        y_pred: Union[np.ndarray, torch.Tensor]
            Estimated target values.

    Returns:
        float
            R-squared score.
    """
    y_true_array, y_pred_array = _check_arrays(y_true, y_pred)
    
    d = (y_true_array - y_pred_array) ** 2
    D = (y_true_array - y_true_array.mean()) ** 2
    return 1 - np.sum(d) / np.sum(D)
