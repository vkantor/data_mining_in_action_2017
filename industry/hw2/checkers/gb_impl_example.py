from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


TREE_PARAMS_DICT = {}
TAU = 0.1


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.estimators = []
        curr_pred = 0
        for iter_num in xrange(self.iters):
            self.estimators.append(DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data))
        return self
    
    def predict(self, X_data):
        res = np.zeros(X_data.shape[0])
        for estimator in self.estimators:
            res += estimator.predict(X_data)
        res /= len(self.estimators)
        return res
