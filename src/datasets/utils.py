import torch
import numpy as np

from typing import Union


def to_forecasting(
    data: np.ndarray,
    forecast: int = 1,
    axis: int = 0,
    test_size: Union[int, float] = None
) -> np.ndarray:
    series_ = np.moveaxis(data.view(), axis, 0)
    time_len = series_.shape[0]
    
    if test_size is not None:
        if isinstance(test_size, float) and test_size < 1 and test_size >= 0:
            test_len = round(time_len * test_size)
        elif isinstance(test_size, int):
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
