import numpy as np
import pandas as pd
from copulas import bivariate, multivariate

from tscopulas.models.base import BaseModel


class Vine(BaseModel):
    """Vine copula model modified to work with time series applications"""

    def __init__(self, max_lag: int, **config):
        """
        Args:
            max_lag: The maximum offset to form relationships for, note this is a cartesian product of the variables
                     meaning that the data to fit on grows by a factor of <max_lag>
        """
        super().__init__(max_lag, **config)
        self.model = multivariate.VineCopula(self._config["vine_type"])
        self.n_var = None

    def fit(self, data: pd.DataFrame):
        """
        Transform data to lagged dataframe and fit Multivariate Gaussian on lagged variables

        Marks the model as fitted
        """
        super().fit(data)
        self.n_var = self._transform_data.shape[1]

    def _conditional_sample_row(self, condition):
        """Generate a single sampled row from vine model.

        Args:
            condition: Tuple(feature, value) of value we would like to condition the copula on for a certain feature

        Returns:
            numpy.ndarray
        """
        unis = np.random.uniform(0, 1, self.n_var)
        # select conditional node to start with
        # condition[0] is a numerical value index of which feature column we would like to condition on
        first_ind = condition[0]
        adj = self.model.trees[0].get_adjacent_matrix()

        sampled = np.zeros(self.n_var)
        sampled[first_ind] = condition[1]

        second_ind = np.random.randint(0, self.n_var)
        while second_ind == first_ind:
            second_ind = np.random.randint(0, self.n_var)

        explore = [second_ind]
        visited = [first_ind]

        itr = 0
        while explore:
            if first_ind in explore:
                explore.remove(first_ind)
            current = explore.pop(0)
            neighbors = np.where(adj[current, :] == 1)[0].tolist()
            if itr == 0:
                new_x = self.model.ppfs[current](unis[current])

            else:
                for i in range(itr - 1, -1, -1):
                    current_ind = -1

                    if i >= self.model.truncated:
                        continue

                    current_tree = self.model.trees[i].edges
                    # get index of edge to retrieve
                    for edge in current_tree:
                        if i == 0:
                            if (edge.L == current and edge.R == visited[0]) or \
                                    (edge.R == current and edge.L == visited[0]):
                                current_ind = edge.index
                                break
                        else:
                            if edge.L == current or edge.R == current:
                                condition = set(edge.D)
                                condition.add(edge.L)
                                condition.add(edge.R)

                                visit_set = set(visited)
                                visit_set.add(current)

                                if condition.issubset(visit_set):
                                    current_ind = edge.index
                                break

                    if current_ind != -1:
                        # the node is not indepedent contional on visited node
                        copula_type = current_tree[current_ind].name
                        copula = bivariate.Bivariate(copula_type=bivariate.CopulaTypes(copula_type))
                        copula.theta = current_tree[current_ind].theta

                        U = np.array([unis[visited[0]]])
                        if i == itr - 1:
                            tmp = copula.percent_point(np.array([unis[current]]), U)[0]
                        else:
                            tmp = copula.percent_point(np.array([tmp]), U)[0]

                        tmp = min(max(tmp, bivariate.EPSILON), 0.99)

                new_x = self.model.ppfs[current](np.array([tmp]))

            sampled[current] = new_x

            for s in neighbors:
                if s not in visited:
                    explore.insert(0, s)

            itr += 1
            visited.insert(0, current)

        return sampled

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
        if conditional not in self._transform_data.columns:
            raise ValueError("Conditional Column not found in transformed data")

        sampled_values = []
        idx = self._transform_data.columns.get_loc(conditional)
        value = self._transform_data[conditional].iloc[-1]
        i = 0
        while i < num_samples:
            sampled_values.append(self._conditional_sample_row((idx, value)))
            i += 1

        new_samples = pd.DataFrame(sampled_values, columns=self._transform_data.columns)
        new_samples = new_samples[self._zero_columns]
        new_samples.columns = new_samples.columns.str.slice(0, -9)
        return new_samples

    def series_sample(self, cond_col: str, lag: int):
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
        if self._data is None:
            raise ValueError("Please fit model on data")

        # Need new fit a new model if on larger lag
        if lag > self.max_lag:
            raise ValueError(
                f"This model has a max lag of {self.max_lag}. Please create a new model with max lag >= {self.max_lag}"
                f", i.e. Gaussian(data, max_lag={self.max_lag})"
            )

        samples = pd.DataFrame(index=range(1, lag + 1), columns=self._zero_columns)
        for i in range(1, lag + 1):
            new_sample = self.sample(1, cond_col=cond_col, cond_lag=i)
            samples.iloc[i-1, :] = new_sample.values[0]

        samples = samples[self._zero_columns]
        samples.columns = samples.columns.str.slice(0, -9)
        return samples
