# TSCopulas

![Travis-CI](https://img.shields.io/travis/com/ryansdowning/TSCopulas/master)
[![codecov](https://codecov.io/gh/ryansdowning/TSCopulas/branch/master/graph/badge.svg?token=N8H3VW5H0T)](https://codecov.io/gh/ryansdowning/TSCopulas)


Applying copula modelling to time series data for forecasting and synthetic data generation. This package implements methods
from the paper [COMING SOON] on time series modeling with copulas and conditioning with vine models.

## Usage

```python
import pandas as pd
from tscopulas.utils import *
from tscopulas.models import gaussian, vine

tickers = rand_tickers(100)
dates = pd.date_range(datetime.datetime(2015, 1, 1), periods=500, freq='d')
init_prices = np.full(100, 100)
sim = simulated_brownian_motion(init_prices, dates.shape[0], mu=0.0002, sigma=0.01)

copula = guassian.Gaussian(5)  # 5 time step maximum lag
copula.fit(sim)

# Generate 100 samples conditioned on the first column with a 5 day lag
synthetic_data = copula.sample(100, cond_col=sim.columns[0], cond_lag=5)

# Similarly for a Vine copula we have
copula = vine.Vine(5, vine_type="regular")
copula.fit(sim)

# Generate 100 samples conditioned on the first column with a 5 day lag
synthetic_data = copula.sample(100, cond_col=sim.columns[0], cond_lag=5)
```
