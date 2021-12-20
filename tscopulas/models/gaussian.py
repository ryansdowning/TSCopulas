import pandas as pd
from copulas import multivariate

from tscopulas.models.base import BaseModel


class Gaussian(BaseModel):
    """Gaussian copula model modified to work with time series applications"""

    def __init__(self, max_lag: int, **config):
        """
        Args:
            max_lag: The maximum offset to form relationships for, note this is a cartesian product of the variables
                     meaning that the data to fit on grows by a factor of <max_lag>
        """
        super().__init__(max_lag, **config)
        self.model = multivariate.GaussianMultivariate()

    def sample(self, num_samples: int, cond_col: str, cond_lag: int):
        """
        Generate num_samples new samples of data given cond_col lagged by cond_lag time units
        is equal to most recently seen observation of variable (sequentiality concept)

        In order to generate data from current-day lag variable, set cond_lag=0

        Args:
            num_samples: Integer, number of samples to generate
            cond_col: The name of the column to condition on
            cond_lag: The offset to condition on for the selected column

        Returns:
             new sample rows
        """
        if self._data is None:
            raise ValueError("Please fit model on data")

        conditional = f"{cond_col}_lagged_{cond_lag}"
        if conditional not in self._transform_data:
            raise ValueError("Conditional Column not found in transformed data")

        new_sample = self.model.sample(
            num_samples,
            conditions={conditional: self._transform_data[conditional].iloc[-1]},
        )
        new_sample = new_sample[self._zero_columns]
        new_sample.columns = new_sample.columns.str.slice(0, -9)
        return new_sample

    def series_sample(self, cond_col, lag: int):
        """
        Generates chunk of samples equal to length (max_lag + 1) based off of lagging variables
        of chosen conditional feature/column

        Args:
            cond_col: Name of the column to condition on
            lag: The number of offsets to use for conditioning on the selected column

        Returns:
            Pandas dataframe of <lag> number of rows where each row represents the sample of all other columns given
            conditioned on the selected column's previous value, rolling
        """
        # Need new fit a new model if on larger lag
        if lag > self.max_lag:
            raise ValueError(
                f"This model has a max lag of {self.max_lag}. Please create a new model with max lag >= {self.max_lag}"
                f", i.e. Gaussian(data, max_lag={self.max_lag})"
            )

        if self._data is None:
            raise ValueError("Please fit model on data")

        samples = pd.DataFrame(index=range(1, lag + 1), columns=self._transform_data.columns)
        for i in range(lag):
            conditional = f"{cond_col}_lagged_{i}"
            new_sample = self.model.sample(1, conditions={conditional: self._transform_data[conditional].iloc[-1]})
            samples.iloc[i, :] = new_sample.iloc[0, :]

        samples = samples[self._zero_columns]
        samples.columns = samples.columns.str.slice(0, -9)
        return samples
