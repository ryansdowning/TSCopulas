import string

import numpy as np
import pytest

from tscopulas import utils


def test_rand_ticker():
    ticker = utils.rand_ticker(5)
    assert isinstance(ticker, str)
    assert len(ticker) == 5
    assert all(i in string.ascii_uppercase for i in ticker)

    ticker = utils.rand_ticker(10000)
    assert len(ticker) == 10000

    ticker = utils.rand_ticker(0)
    assert isinstance(ticker, str)
    assert len(ticker) == 0


@pytest.mark.parametrize("n", [0, 1, 5, 10, 20, 50, 100, 1000, 10000])
def test_rand_tickers(n):
    tickers = utils.rand_tickers(n)
    assert isinstance(tickers, np.ndarray)
    assert tickers.shape[0] == n
    assert all(isinstance(i, str) for i in tickers)


@pytest.mark.parametrize("n", [5, 10, 50, 100])
@pytest.mark.parametrize("m", [1, 10, 100, 1000, 10000])
def test_brownian_motion(n, m):
    sim = utils.simulated_brownian_motion(np.full(n, 100), m, mu=0, sigma=0.1)
    assert sim.shape == (m, n)
