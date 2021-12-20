import datetime

import numpy as np
import pandas as pd
import pytest
from copulas.multivariate import gaussian

from tscopulas import utils
from tscopulas.models.gaussian import *


class MockGaussianCopula:
    def __init__(self):
        self.shape = None
        self.columns = None

    def fit(self, data):
        self.shape = data.shape
        self.columns = data.columns
        return

    def sample(self, num_samples, conditions):
        samples = np.arange(self.shape[1] * num_samples).reshape(num_samples, self.shape[1])
        return pd.DataFrame(samples, columns=self.columns)


@pytest.fixture
def data():
    tickers = utils.rand_tickers(100)
    dates = pd.date_range(datetime.datetime(2015, 1, 1), periods=500, freq="d")
    init_prices = np.full(100, 100)
    sim = utils.simulated_brownian_motion(init_prices, dates.shape[0], mu=0.0002, sigma=0.01)
    data = pd.DataFrame(sim, columns=tickers, index=dates)
    return data


@pytest.mark.parametrize("shape,max_lag", [((10, 10), 5), ((100, 10), 10), ((1000, 10), 10), ((1000, 100), 10)])
def test_gaussian_init(monkeypatch, shape, max_lag):
    data = pd.DataFrame(np.random.random(shape))
    gauss = Gaussian(max_lag)
    assert gauss.max_lag == max_lag

    gauss.model = MockGaussianCopula()
    gauss.fit(data)
    assert gauss._transform_data.shape == (data.shape[0], data.shape[1] * max_lag + data.shape[1])


def test_gaussian_fit(monkeypatch, data):
    monkeypatch.setattr(multivariate, "GaussianMultivariate", MockGaussianCopula)

    gauss = Gaussian(10)
    assert gauss._transform_data is None
    gauss.fit(data)
    assert isinstance(gauss._transform_data, pd.DataFrame)


@pytest.mark.parametrize("num_samples", [1, 5, 10, 100])
@pytest.mark.parametrize("cond_lag", [1, 2, 3, 5, 10])
def test_gaussian_sample(monkeypatch, data, num_samples, cond_lag):
    monkeypatch.setattr(multivariate, "GaussianMultivariate", MockGaussianCopula)

    gauss = Gaussian(10)
    cond_col = str(data.columns[0])

    # Cannot sample before fitting
    with pytest.raises(ValueError):
        gauss.sample(num_samples, cond_col, cond_lag)

    # Invalid column name
    with pytest.raises(ValueError):
        gauss.sample(num_samples, "__BAD_COLUMN__", cond_lag)

    gauss.fit(data)
    samples = gauss.sample(num_samples=num_samples, cond_col=cond_col, cond_lag=cond_lag)
    sample_shape = (num_samples, data.shape[1])
    assert samples.shape == sample_shape
    np.testing.assert_array_equal(
        samples.values, np.arange(
            num_samples * (data.shape[1] * 10 + data.shape[1])
        ).reshape(num_samples, -1)[:, :data.shape[1]]
    )


@pytest.mark.parametrize("lag", [1, 2, 3, 5, 10, 20, 40, 50])
def test_gaussian_series_sample(monkeypatch, data, lag):
    monkeypatch.setattr(multivariate, "GaussianMultivariate", MockGaussianCopula)

    gauss = Gaussian(10)
    cond_col = str(data.columns[0])

    # Cannot sample before fitting
    with pytest.raises(ValueError):
        gauss.series_sample(cond_col, lag)

    gauss.fit(data)
    if lag > 10:
        with pytest.raises(ValueError):
            gauss.series_sample(cond_col, lag)
    else:
        samples = gauss.series_sample(cond_col, lag)
        sample_shape = (lag, data.shape[1])
        assert samples.shape == sample_shape
        np.testing.assert_array_equal(
            samples.values, np.tile(np.arange(sample_shape[1]), lag).reshape(lag, -1)
        )
