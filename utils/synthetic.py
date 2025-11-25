import numpy as np
import pandas as pd


def stationary_bootstrap(series, block_prob=0.1, size=None):
    """
    Generate a synthetic series using stationary bootstrap
    (Politis & Romano, 1994).

    series: 1D numpy array or pandas series
    block_prob: probability of starting a new block (p)
    size: length of synthetic output
    """
    series = np.asarray(series)
    T = len(series)
    if size is None:
        size = T

    synthetic = np.zeros(size)
    idx = np.random.randint(0, T)  # initial point

    for t in range(size):
        synthetic[t] = series[idx]

        # continue block or start new block?
        if np.random.rand() < block_prob:
            idx = np.random.randint(0, T)  # new block
        else:
            idx = (idx + 1) % T  # continue block

    return synthetic


def generate_synthetic_paths(n_paths: int, n_steps: int, mu: float, sigma: float, s0: float):
    """
    Génère des trajectoires de prix avec un mouvement brownien géométrique.
    """

    dt = 1.0
    paths = np.zeros((n_paths, n_steps))

    for i in range(n_paths):
        path = np.zeros(n_steps)
        path[0] = s0

        for t in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            path[t] = path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

        paths[i] = path

    return paths
