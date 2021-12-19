from abc import abstractmethod

import pandas as pd

from tscopulas.preprocessing import lag_transform as lt


class BaseModel:
    def __init__(self, max_lag: int, **config):
        """
        Args:
            max_lag: The maximum offset to form relationships for, note this is a cartesian product of the variables
                     meaning that the data to fit on grows by a factor of <max_lag>
        """
        self.max_lag = max_lag
        self._data = None
        self._transform_data = None
        self._zero_columns = None
        self._config = config

    def transform(self, data):
        """
        Applies lag transform to data passed in given max_lag

        Args:
            data: A pandas dataframe of time series data that will be used to model time dependent relations between
                  variables (columns)

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

        self._zero_columns = lagged_data.columns[lagged_data.columns.str.endswith("lagged_0")]

        # Apply NaN Filter
        lagged_data = lagged_data.fillna(lagged_data.mean())
        return lagged_data

    def fit(self, data: pd.DataFrame):
        """
        Transform data to lagged dataframe and fit Multivariate Gaussian on lagged variables

        Marks the model as fitted
        """
        self._data = data
        self._transform_data = self.transform(self._data)
        self.model.fit(self._transform_data)

    @abstractmethod
    def sample(self, num_samples: int, cond_col: str, cond_lag: int):
        raise NotImplementedError

    @abstractmethod
    def series_sample(self, cond_col: str, lag: int):
        raise NotImplementedError
