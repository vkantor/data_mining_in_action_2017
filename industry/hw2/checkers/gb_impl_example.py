from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


TREE_PARAMS_DICT = {'max_depth': 2}
TAU = 0.05


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)
        for iter_num in xrange(self.iters):
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data - curr_pred)
            self.estimators.append(algo)
            curr_pred += self.tau * algo.predict(X_data)
        return self
    
    def predict(self, X_data):
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        return res
