from sklearn.datasets import make_friedman1
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import imp
import signal


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def signal_handler(signum, frame):
    raise Exception("Timed out!")


class Checker(object):
    def __init__(self):
        self.X_data, self.y_data = make_friedman1(n_samples=1000, noise=10, n_features=10, random_state=777)

    def check(self, script_path):
        try:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(20)
            gb_impl = imp.load_source('gb_impl', script_path)
            algo = gb_impl.SimpleGB(
                tree_params_dict=gb_impl.TREE_PARAMS_DICT,
                iters=100,
                tau=gb_impl.TAU
            )
            return np.mean(cross_val_score(algo, self.X_data, self.y_data, cv=5, scoring='neg_mean_squared_error'))
        except:
            return None


if __name__ == '__main__':
    print Checker().check(SCRIPT_DIR + '/gb_impl_example.py')
