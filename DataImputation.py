# -*- coding: utf-8 -*-
"""
@author: Prasad Gandham
"""
# Ref: https://scikit-learn.org/stable/modules/impute.html
#3 This will compute the missing values using various imputers in sklearn

import numpy as np
from sklearn.impute import SimpleImputer

##########################################
# this is simple imputer where the strategy can be mean or most frequent element
imp = SimpleImputer(np.nan, strategy='mean')
x=np.array([[1,2],[np.nan,3],[7,6]])
print(x)
imp.fit(x)
print(x)
x=imp.transform(x)
print(x)


############################################
#iterative imputer where it imputes based on other vars
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp=IterativeImputer(max_iter=10, sample_posterior=True)
x=np.array([[1,2],[3,6],[4,8],[np.nan,3],[7,np.nan],[11, np.nan]])
print(x)
imp.fit(x)
print(x)
x=imp.transform(x)
print(x)

#################################################
# similarly we can use KNNImputer
from sklearn.impute import KNNImputer
imp=KNNImputer(n_neighbors=2, weights='uniform')
x=np.array([[1,2],[3,6],[4,8],[np.nan,3],[7,np.nan],[11, np.nan]])
print(x)
#imp.fit(x)
print(x)
x=imp.fit_transform(x)
print(x)
