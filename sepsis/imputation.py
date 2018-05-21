__author__ = 'Aaron J Masino'

import pandas as pd
import numpy as np

def missing_percents(df):
    d = {}
    denom = float(len(df))
    for c in df.columns:
        d[c] = np.sum(pd.isnull(df[c]))/denom*100
    return d

# TODO
# add an imputation method that generates a normal random sample based on the sample mean and stdev in the
# non-missing data