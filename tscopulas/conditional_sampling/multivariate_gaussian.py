import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tscopulas.utils import *
from tscopulas.preprocessing import lag_transform as lt
from copulas.multivariate import GaussianMultivariate


def conditional_MultivariateGaussian(data, cols, cond_col, cond_lag, val, max_lag=5):
    
    # Apply lag transform to data with columns and maximum lag specified
    lagged_2d = lt.apply_lag_transform(data[cols + cond_col].values, max_lag)
    
    # Create 2D Pandas DataFrame with lagged columns to generate pairs
    lagged_data = pd.DataFrame()
    for i in range(max_lag+1):
        temp = pd.DataFrame(lagged_2d[...,i], columns=[f"{j}_lagged_{i}" for j in data.columns], index=data.index)
        lagged_data = pd.concat([lagged_data, temp], axis=1)
        
    # Apply NaN Filter
    lagged_data = lagged_data.fillna(lagged_data.mean())   
        
    # Fit Multivariate Gaussian on every pair    
    dist = GaussianMultivariate()
    dist.fit(lagged_data)

    # Generate conditional 
    return dist.sample(1000, conditions={f"{cond_col}_{cond_lag}": val})
