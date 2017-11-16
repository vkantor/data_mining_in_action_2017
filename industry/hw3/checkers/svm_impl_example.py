import numpy as np
from sklearn.base import BaseEstimator


SVM_PARAMS_DICT = {
    'C': 1000,
    'random_state': 42,
    'iters': 10000
}


class MySVM(BaseEstimator):
    def __init__(self, C, random_state, iters):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        # I'd recommend you to add your more own parameters

    # f(x) = <w,x> + w_0
    def f(self, x):
        return np.dot(self.w, x) + self.w0

    # a(x) = [f(x) > 0]
    def a(self, x):
        return 1 if self.f(x) > 0 else 0
    
    # predicting answers for X_test
    def predict(self, X_test):
        return np.array([self.a(x) for x in X_test])

    # l2-regularizator
    def reg(self):
        return 1.0 * sum(self.w ** 2) / (2.0 * self.C)

    # l2-regularizator derivative
    def der_reg(self):
        '''ToDo: fix this function'''
        return 0.0

    # hinge loss
    def loss(self, x, answer):
        return max([0, 1 - answer * self.f(x)])

    # hinge loss derivative
    def der_loss(self, x, answer):
        return -1.0 if 1 - answer * self.f(x) > 0 else 0.0

    # fitting w and w_0 with SGD
    def fit(self, X_train, y_train):
        random_gen = np.random.RandomState(self.random_state)
        dim = len(X_train[0])
        self.w = random_gen.rand(dim) # initial value for w
        self.w0 = random_gen.randn() # initial value for w_0
        
        for k in range(self.iters):  
            
            # random example choise
            rand_index = random_gen.choice(dim) # generating random index
            x = X_train[rand_index]
            y = y_train[rand_index]

            # simple heuristic for step size
            step = 0.5 * 0.9 ** k

            # w update
            '''ToDo: add w update with regularization'''
            
            # w_0 update
            '''ToDo: add w_0 update'''
        return self