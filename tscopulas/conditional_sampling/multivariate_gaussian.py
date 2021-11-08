import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tscopulas.utils import *
from tscopulas.preprocessing import lag_transform as lt
from copulas.multivariate import GaussianMultivariate


class conditionalMultivariateGaussian():
    
    def __init__(self, data, max_lag):
        self.model = GaussianMultivariate()
        self.data = data
        self.max_lag = max_lag
        self.fit = False
        
    def transform(self):
        """
        Applies lag transform to data passed in given max_lag
        
        Returns: 2D pd.DataFrame of transformed data with lag variables as columns
        """
        # Apply lag transform to original data
        lagged_2d = lt.apply_lag_transform(self.data.values, self.max_lag)
        # Generate 2D dataframe of lagged variables
        lagged_data = pd.DataFrame()
        for i in range(self.max_lag+1):
            temp = pd.DataFrame(lagged_2d[...,i], columns=[f"{j}_lagged_{i}" for j in self.data.columns])
            lagged_data = pd.concat([lagged_data, temp], axis=1)
        # Apply NaN Filter
        lagged_data = lagged_data.fillna(lagged_data.mean()) 
        self.transform_data = lagged_data
        return lagged_data
        
    def fit_transform(self):
        """
        Transform data to lagged dataframe and fit Multivariate Gaussian on lagged variables
        """
        transform_data = self.transform()
        self.model.fit(transform_data)
        self.fit = True 
        
    def sample(self, num_samples, cond_col, cond_lag):
        """
        Generate num_samples new samples of data given cond_col lagged by cond_lag time units 
        is equal to most recently seen observation of variable (sequentiality concept)
        
        In order to generate data from current-day lag variable, set cond_lag=0
        
        Returns: Tuple(dataframe of transformed data + new sample appended, new sample row)
        """
        conditional = f"{cond_col}_lagged_{cond_lag}"
        if conditional not in self.transform_data:
            raise ValueError("Conditional Column not found in transformed data")
        if self.fit == False:
            raise ValueError("Please fit model on data")
        new_sample = self.model.sample(num_samples, conditions={conditional: self.transform_data[conditional].iloc[-1]})
        self.transform_data = self.transform_data.append(new_sample)
        return (self.transform_data, new_sample)
    
    
    def series_sample(self, cond_col):
        """
        Generates chunk of samples equal to length (max_lag + 1) based off of lagging variables
        of chosen conditional feature/column
        """
        samples = pd.DataFrame(index=range(self.max_lag+1), columns=self.transform_data.columns)
        for i in range(self.max_lag+1):
            conditional = f"{cond_col}_lagged_{i}"
            new_sample = self.model.sample(1, conditions={conditional: self.transform_data[conditional].iloc[-1]})
            samples.iloc[i,:] = new_sample.iloc[0,:]
        return samples