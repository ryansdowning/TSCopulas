import datetime
from functools import partial

import numpy as np
import pandas as pd
import pytest
from copulas.multivariate import vine

from tscopulas import utils
from tscopulas.models.vine import *


class MockVineCopula:
    def __init__(self, vine_type):
        self.vine_type = vine_type
        self.shape = None
        self.columns = None

    def fit(self, data):
        self.shape = data.shape
        self.columns = data.columns
        return

    def sample(self, num_samples, cond_col, cond_lag):
        samples = np.arange(self.shape[1] * num_samples).reshape(num_samples, self.shape[1])
        return pd.DataFrame(samples, columns=self.columns)


class MockVine:
    def sample(self, num_samples, cond_col, cond_lag):
        samples = np.arange(100 * num_samples).reshape(num_samples, 100)
        return pd.DataFrame(samples, columns=range(100))


@pytest.fixture
def data():
    tickers = utils.rand_tickers(100)
    dates = pd.date_range(datetime.datetime(2015, 1, 1), periods=500, freq="d")
    init_prices = np.full(100, 100)
    sim = utils.simulated_brownian_motion(init_prices, dates.shape[0], mu=0.0002, sigma=0.01)
    data = pd.DataFrame(sim, columns=tickers, index=dates)
    return data


@pytest.fixture
def small_data():
    tickers = utils.rand_tickers(10)
    dates = pd.date_range(datetime.datetime(2015, 1, 1), periods=10, freq="d")
    init_prices = np.full(10, 10)
    sim = utils.simulated_brownian_motion(init_prices, dates.shape[0], mu=0.0002, sigma=0.01)
    data = pd.DataFrame(sim, columns=tickers, index=dates)
    return data


@pytest.mark.parametrize("shape,max_lag", [((10, 10), 5), ((100, 10), 10), ((1000, 10), 10), ((1000, 100), 10)])
@pytest.mark.parametrize("vine_type", ["regular", "center", "direct"])
def test_vine_init(shape, max_lag, vine_type):
    data = pd.DataFrame(np.random.random(shape))
    vi = Vine(max_lag, vine_type=vine_type)
    assert vi.max_lag == max_lag
    assert "vine_type" in vi._config
    assert vi._config["vine_type"] == vine_type

    vi.model = MockVineCopula(vine_type)
    assert vi.model.vine_type == vine_type
    vi.fit(data)
    assert vi._transform_data.shape == (data.shape[0], data.shape[1] * max_lag + data.shape[1])


@pytest.mark.parametrize("vine_type", ["regular", "center", "direct"])
def test_gaussian_fit(monkeypatch, data, vine_type):
    monkeypatch.setattr(multivariate, "VineCopula", MockVineCopula)

    vi = Vine(10, vine_type=vine_type)
    assert vi._transform_data is None
    vi.fit(data)
    assert isinstance(vi._transform_data, pd.DataFrame)


@pytest.mark.parametrize("num_samples", [1, 5, 10, 15, 20, 25])
@pytest.mark.parametrize("cond_lag", [2, 3, 4])
def test_gaussian_sample(small_data, num_samples, cond_lag):
    vi = Vine(cond_lag, vine_type="regular")
    cond_col = str(small_data.columns[0])

    # Cannot sample before fitting
    with pytest.raises(ValueError):
        vi.sample(num_samples, cond_col, cond_lag)

    try:
        vi.fit(small_data)
        # Invalid column name
        with pytest.raises(ValueError):
            vi.sample(num_samples, "__BAD_COLUMN__", cond_lag)

        samples = vi.sample(num_samples=num_samples, cond_col=cond_col, cond_lag=cond_lag)
        sample_shape = (num_samples, small_data.shape[1])
        assert samples.shape == sample_shape
    except ValueError:
        assert True


@pytest.mark.parametrize("lag", [1, 2, 3, 5, 10, 20, 40, 50])
@pytest.mark.parametrize("vine_type", ["regular", "center", "direct"])
def test_gaussian_series_sample(monkeypatch, data, lag, vine_type):
    monkeypatch.setattr(multivariate, "VineCopula", MockVineCopula)

    vi = Vine(10, vine_type=vine_type)
    cond_col = str(data.columns[0])

    # Cannot sample before fitting
    with pytest.raises(ValueError):
        vi.series_sample(cond_col, lag)

    vi.fit(data)
    if lag > 10:
        with pytest.raises(ValueError):
            vi.series_sample(cond_col, lag)
    else:
        monkeypatch.setattr(Vine, "sample", MockVine.sample)
        samples = vi.series_sample(cond_col, lag)
        sample_shape = (lag, data.shape[1])
        assert samples.shape == sample_shape
        np.testing.assert_array_equal(
            samples.values, np.tile(np.arange(sample_shape[1]), lag).reshape(lag, -1)
        )
