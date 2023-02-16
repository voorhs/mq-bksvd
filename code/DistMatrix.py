# query and preprocess algorithms are taken from
# 'Faster Linear Algebra for Distance Matrices'
# Piotr Indyk, Sandeep Silwal, 2022
# https://openreview.net/forum?id=y--ZUTfbNB

import numpy as np
from numba import jit


class BaseDistMatrix:
    def __init__(self, X):
        self.X = X.copy()

    def dot(self, B):
        """
        Matrix multiplication with B
        """
        if B.shape[0] != self.shape[1]:
            raise ValueError()

        ans = np.empty((self.shape[0], B.shape[1]))
        for i in range(B.shape[1]):
            ans[:, i] = self._query(B[:, i].flatten())
        return ans

    def _query(self, y):
        """
        Implemetation of query algorithm
        """
        raise NotImplementedError()

    @property
    def shape(self):
        n = self.X.shape[0]
        return (n, n)


class L1DistMatrix(BaseDistMatrix):
    def preprocess(self):
        self.order1 = np.argsort(self.X, axis=0)
        self.order2 = np.argsort(np.argsort(self.X, axis=0), axis=0)

    def _query(self, y):
        n, d = self.X.shape
        B = np.take_along_axis((((self.X.T) * y).T),
                               self.order1, axis=0).cumsum(axis=0)
        C = (y[self.order1.T].T).cumsum(axis=0)

        return inner_loop(self.X, self.order2, B, C, n, d)


@jit(nopython=True)
def inner_loop(X, order, B, C, n, d):
    z = np.zeros(n)  # ответ
    for k in range(n):
        for i in range(d):
            # позиция i-го признака k-го обьекта по порядку T_i
            q = order[k, i]
            S1 = B[q, i]
            # разница куммулятивных сумм и есть то, что нам надо (из формулы)
            S2 = B[n - 1, i] - B[q, i]
            S3 = C[q, i]
            S4 = C[n - 1, i] - C[q, i]
            z[k] += X[k, i] * (S3 - S4) + S2 - S1

    return z


class L2_2DistMatrix(BaseDistMatrix):
    def _query(self, y):
        X_norm = np.sum(self.X ** 2, axis=1)
        S_1 = np.sum(y)
        S_2 = np.sum(X_norm * y)
        v = np.sum(self.X * y[:, None], axis=0)
        z = S_1 * X_norm + S_2 - 2 * self.X @ v
        return z
