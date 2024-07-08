import numpy as np
import matplotlib.pyplot as plt
from numpy.random import Generator, RandomState

from typing import Union, List


def _mg_eq(
    xt: float,
    xtau: float,
    a: float = 0.2,
    b: float = 0.1,
    n: int = 10
) -> float:
    """
    Mackey-Glass time delay differential equation, at values x(t) and x(t-tau).

    Args:
        xt (float): Current state value x(t).
        xtau (float): Delayed state value x(t-tau).
        a (float, optional): Parameter a of the Mackey-Glass equation.
        b (float, optional): Parameter b of the Mackey-Glass equation.
        n (int, optional): Parameter n of the Mackey-Glass equation.

    Returns:
        float: The result of the Mackey-Glass differential equation.
    """
    return -b * xt + a * xtau / (1 + xtau ** n)


def _mg_rk4(
    xt: float,
    xtau: float,
    a: float,
    b: float,
    n: int,
    h: float = 1.0
) -> float:
    """
    Runge-Kutta method (RK4) for Mackey-Glass timeseries discretization.

    Args:
        xt (float): Current state value x(t).
        xtau (float): Delayed state value x(t-tau).
        a (float): Parameter a of the Mackey-Glass equation.
        b (float): Parameter b of the Mackey-Glass equation.
        n (int): Parameter n of the Mackey-Glass equation.
        h (float, optional): Time step size.

    Returns:
        float: The next state value x(t+h).
    """
    k1 = h * _mg_eq(xt, xtau, a, b, n)
    k2 = h * _mg_eq(xt + 0.5 * k1, xtau, a, b, n)
    k3 = h * _mg_eq(xt + 0.5 * k2, xtau, a, b ,n)
    k4 = h * _mg_eq(xt + k3, xtau, a, b, n)
    
    return xt + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6


def mackey_glass_generate(
    n_timesteps: int,
    tau: int = 17,
    a: float = 0.2,
    b: float = 0.1,
    n: int = 10,
    x0: float = 1.2,
    h: float = 1.0,
    seed: Union[int, RandomState, Generator] = None
) -> np.ndarray:
    """
    Generate a Mackey-Glass timeseries using the Mackey-Glass delayed differential equation.

    Args:
        n_timesteps (int): Number of timesteps to compute.
        tau (int, optional): Time delay tau of the Mackey-Glass equation.
        a (float, optional): Parameter a of the Mackey-Glass equation.
        b (float, optional): Parameter b of the Mackey-Glass equation.
        n (int, optional): Parameter n of the Mackey-Glass equation.
        x0 (float, optional): Initial condition of the timeseries.
        h (float, optional): Time delta between two discrete timesteps.
        seed (Union[int, RandomState, Generator], optional): Random state seed for reproducibility.

    Returns:
        np.ndarray: Mackey-Glass timeseries of shape (n_timesteps, 1).
    """
    
    # https://blog.csdn.net/ddpiccolo/article/details/89464435
    # https://blog.csdn.net/u013007900/article/details/45922331
    if seed is None:
        rs = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rs = seed
    else:
        rs = np.random.default_rng(seed)
    
    history_length = int(np.floor(tau / h))
    history = x0 * np.ones(history_length) + 0.2 * (rs.random(history_length) - 0.5)
    xt = x0
    
    X = np.zeros(n_timesteps)
    
    for i in range(0, n_timesteps):
        X[i] = xt
        
        if tau == 0:
            xtau = 0
        else:
            # TODO: не уверен в данных строчках
            xtau = history[i % history_length]
            history[i % history_length] = xt
        
        xth = _mg_rk4(xt, xtau, a, b, n, h)
        xt = xth
    
    return X.reshape(-1, 1)


def plot_mackey_glass(
    data: np.ndarray,
    tau: int,
    sample: int,
    start_index: int = 0
) -> None:
    """
    Plot the Mackey-Glass timeseries and its phase diagram from a specified starting index.

    Args:
        data (np.ndarray): The Mackey-Glass timeseries data.
        tau (int): Time delay for the phase diagram.
        sample (int): Number of timesteps to plot.
        start_index (int, optional): Index to start displaying the data from, by default 0.
    """
    if start_index < 0 or start_index >= len(data):
        raise ValueError(
            f"start_index {start_index} is out of bounds for the data of length {len(data)}."
        )
    if start_index + sample > len(data):
        raise ValueError(
            f"The combination of start_index {start_index} and sample {sample} exceeds data length {len(data)}."
        )
    
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    t = np.linspace(start_index, start_index + sample, sample)
    for i in range(sample - 1):
        ax.plot(
            t[i:i+2], data[start_index + i : start_index + i + 2],
            color=plt.cm.magma(255 * i // sample), lw=1.0
        )
    ax.set_title(f"Timeseries - {sample} timesteps (starting from index {start_index})", fontdict={'fontsize': 14})
    ax.set_xlabel("$t$", fontdict={'fontsize': 14})
    ax.set_ylabel("$P(t)$", fontdict={'fontsize': 14})
    
    ax2.margins(0.05)
    for i in range(sample - 1):
        ax2.plot(
            data[start_index + i : start_index + i + 2],
            data[start_index + i + tau : start_index + i + tau + 2],
            color=plt.cm.magma(255 * i // sample),
            lw=1.0
        )
    ax2.set_title("Phase diagram: $P(t) = f(P(t-\\tau))$", fontdict={'fontsize': 14})
    ax2.set_xlabel("$P(t-\\tau)$", fontdict={'fontsize': 14})
    ax2.set_ylabel("$P(t)$", fontdict={'fontsize': 14})
    
    plt.tight_layout()
    plt.show()
