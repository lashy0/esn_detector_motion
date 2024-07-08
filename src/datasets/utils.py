import torch
import numpy as np

from typing import Union, Tuple, Optional, TypeAlias

ForecastingData: TypeAlias = Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]

TorchForecastingData: TypeAlias = Union[
    Tuple[torch.Tensor, torch.Tensor], 
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]


def to_forecasting(
    data: np.ndarray,
    forecast: int = 1,
    axis: int = 0,
    test_size: Union[int, float] = None,
) -> ForecastingData:
    """
    Prepares data for time series forecasting.

    Args:
        data (np.ndarray): Input time series data.
        forecast (int, optional): Number of steps to forecast.
        axis (int, optional): Axis along which the time series data is located.
        test_size (Union[int, float], optional): Size of the test set. If float, it represents the fraction of the data to be used as the test set. If int, it represents the number of elements in the test set. 

    Returns:
        ForecastingData: If test_size is None, returns (X, y). If test_size is specified, returns (X, X_t, y, y_t).
    """
    if forecast <= 0:
        raise ValueError("Forecast must be a positive integer.")
    
    series_ = np.moveaxis(data.view(), axis, 0)
    time_len = series_.shape[0]
    
    if time_len <= forecast:
        raise ValueError("Forecast period is longer than the time series length.")
    
    if test_size is not None:
        if isinstance(test_size, float) and 0 <= test_size < 1:
            test_len = round(time_len * test_size)
        elif isinstance(test_size, int):
            if test_size < 0:
                raise ValueError("Test size must be non-negative.")
            test_len = test_size
        else:
            raise ValueError(
                "invalid test_size argument: "
                "test_size can be an integer or a float "
                f"in [0, 1], but is {test_size}."
            )
    else:
        test_len = 0
    
    X = series_[:-forecast]
    y = series_[forecast:]
    
    if test_len > 0:
        if test_len > time_len - forecast:
            raise ValueError("Test size is too large for the given data and forecast length.")
        
        X_t = X[-test_len:]
        y_t = y[-test_len:]
        X = X[:-test_len]
        y = y[:-test_len]
        
        X = np.moveaxis(X, 0, axis)
        X_t = np.moveaxis(X_t, 0, axis)
        y = np.moveaxis(y, 0, axis)
        y_t = np.moveaxis(y_t, 0, axis)
        
        return X, X_t, y, y_t
    
    return np.moveaxis(X, 0, axis), np.moveaxis(y, 0, axis)


def to_torch_tensor(
    data: ForecastingData,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> TorchForecastingData:
    """
    Converts numpy arrays returned by to_forecasting to torch tensor.

    Args:
        data (tuple): Tuple of numpy arrays returned by to_forecasting.
        device (torch.device, optional): The desired device of returned tensor.
        dtype (torch.dtype, optional): The desired data type of returned tensor.

    Returns:
        TorchForecastingData: Tuple of torch tensors with the same structure.
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    
    if not isinstance(data, tuple):
        raise ValueError("Input data should be a tuple of numpy arrays.")
    
    for array in data:
        if not isinstance(array, np.ndarray):
            raise ValueError("Each element in the input data tuple should be a numpy array.")
    
    if len(data) == 2:
        X, y = data
        return torch.tensor(X, **factory_kwargs), torch.tensor(y, **factory_kwargs)
    elif len(data) == 4:
        X, X_t, y, y_t = data
        return (torch.tensor(X, **factory_kwargs), torch.tensor(X_t, **factory_kwargs),
                torch.tensor(y, **factory_kwargs), torch.tensor(y_t, **factory_kwargs))
    else:
        raise ValueError("Invalid data format: expected tuple of length 2 or 4.")
