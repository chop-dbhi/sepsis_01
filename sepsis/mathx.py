__author__ = 'Aaron J Masino'

import numpy as np

import sepsis.evaluation as evaluate
import scipy.stats
import scipy as sp
import numpy as np

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    s = np.std(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, s, m-h, m+h