import string

import numpy as np


def rand_ticker(size=5):
    return ''.join(np.random.choice(list(string.ascii_uppercase), size))


def rand_tickers(n, size=5):
    return np.array([rand_ticker(size) for _ in range(n)])


def simulated_brownian_motion(init_prices, steps: int, mu: float = 0, sigma: float = 0.1):
    sim = np.cumsum(np.random.normal(mu, sigma, (steps, init_prices.shape[0])), axis=0) + 1
    return sim * init_prices
