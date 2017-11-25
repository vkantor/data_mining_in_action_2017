# coding=utf-8
import numpy as np
from sklearn.base import BaseEstimator


SVM_PARAMS_DICT = {
    'C': 10.,
    'random_state': 777,
    'iters': 10000,
    'batch_size': 100,
    'step': 0.015
}


class MySVM(BaseEstimator):
    def __init__(self, C, random_state, iters, batch_size, step):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step

    # будем пользоваться этой функцией для подсчёта <w, x>
    def __predict(self, X):
        return np.dot(X, self.w) + self.w0

    # sklearn нужно, чтобы predict возвращал классы, поэтому оборачиваем наш __predict в это
    def predict(self, X):
        res = self.__predict(X)
        res[res > 0] = 1
        res[res < 0] = 0
        return res

    # производная регуляризатора
    def der_reg(self):
        return 1. / self.C * self.w

    # будем считать стохастический градиент не на одном элементе, а сразу на пачке (чтобы было эффективнее)
    def der_loss(self, x, y):
        # s.shape == (batch_size, features)
        # y.shape == (batch_size,)

        # считаем производную по каждой координате на каждом объекте
        ders_w = - y[:, np.newaxis] * x
        der_w0 = - y

        # занулить производные там, где отступ > 1
        indices = (1. - y * self.__predict(x)) < 0.
        ders_w[indices, :] = 0.
        der_w0[indices] = 0.

        # для масштаба возвращаем средний градиент по пачке
        return ders_w.mean(axis=0), der_w0.mean()

    def fit(self, X_train, y_train):
        # RandomState для воспроизводитмости
        random_gen = np.random.RandomState(self.random_state)
        
        # получаем размерности матрицы
        size, dim = X_train.shape
        
        # случайная начальная инициализация
        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()

        for _ in range(self.iters):  
            # берём случайный набор элементов
            rand_indices = random_gen.choice(size, self.batch_size)
            x = X_train[rand_indices]
            y = y_train[rand_indices] * 2 - 1 # исходные метки классов это 0/1 а нам надо -1/1

            # считаем производные
            der_w, der_w0 = self.der_loss(x, y)
            der_w += self.der_reg()

            # обновляемся по антиградиенту
            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step

        # метод fit для sklearn должен возвращать self
        return self