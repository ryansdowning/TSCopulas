import pandas as pd
from copulas.multivariate import GaussianMultivariate

from tscopulas.preprocessing import lag_transform as lt


class Gaussian:
    """Gaussian copula model modified to work with time series applications"""

    def __init__(self, data, max_lag):
        """
        Args:
            data:
            max_lag:
        """
        self.model = GaussianMultivariate()
        self.data = data
        self.transform_data = self.transform(self.data)
        self.max_lag = max_lag
        self._is_fit = False

    def transform(self, data):
        """
        Applies lag transform to data passed in given max_lag

        Args:
            data:

        Returns:
            2D pd.DataFrame of transformed data with lag variables as columns
        """
        # Apply lag transform to original data
        lagged_2d = lt.apply_lag_transform(data.values, self.max_lag)

        # Generate 2D dataframe of lagged variables
        lagged_data = pd.DataFrame()
        for i in range(self.max_lag + 1):
            temp = pd.DataFrame(lagged_2d[..., i], columns=[f"{j}_lagged_{i}" for j in data.columns])
            lagged_data = pd.concat([lagged_data, temp], axis=1)

        # Apply NaN Filter
        lagged_data = lagged_data.fillna(lagged_data.mean())
        return lagged_data

    def fit(self):
        """
        Transform data to lagged dataframe and fit Multivariate Gaussian on lagged variables
        """
        if self._is_fit:
            return
        self.model.fit(self.transform_data)
        self._is_fit = True

    def sample(self, num_samples, cond_col, cond_lag):
        """
        Generate num_samples new samples of data given cond_col lagged by cond_lag time units
        is equal to most recently seen observation of variable (sequentiality concept)

        In order to generate data from current-day lag variable, set cond_lag=0

        Args:
            num_samples:
            cond_col:
            cond_lag:

        Returns:
             new sample row
        """
        conditional = f"{cond_col}_lagged_{cond_lag}"
        if conditional not in self.transform_data:
            raise ValueError("Conditional Column not found in transformed data")

        if not self._is_fit:
            raise ValueError("Please fit model on data")

        new_sample = self.model.sample(
            num_samples,
            conditions={conditional: self.transform_data[conditional].iloc[-1]},
        )
        return new_sample

    def series_sample(self, cond_col, lag: int):
        """
        Generates chunk of samples equal to length (max_lag + 1) based off of lagging variables
        of chosen conditional feature/column

        Args:
            cond_col:
            lag:

        Returns:
        """
        # Need new fit a new model if on larger lag
        if lag > self.max_lag:
            raise ValueError(
                f"This model has a max lag of {self.max_lag}. Please create a new model with max lag >= {self.max_lag}"
                f", i.e. Gaussian(data, max_lag={self.max_lag})"
            )

        samples = pd.DataFrame(index=range(1, lag + 1), columns=self.transform_data.columns)
        for i in range(lag):
            conditional = f"{cond_col}_lagged_{i}"
            new_sample = self.model.sample(1, conditions={conditional: self.transform_data[conditional].iloc[-1]})
            samples.iloc[i, :] = new_sample.iloc[0, :]
        return samples
