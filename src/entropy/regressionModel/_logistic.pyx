import cython
import numpy as cnp
cimport numpy as cnp
from ..base import BaseEstimator

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


class LogisticRegression(BaseEstimator):
    cdef public double bias
    cdef public double[:] weight
    
    def __init__(self):
        self.bias = 0.0

    def fit(self, double[:, :] X, double[:, :] Y, int epoch, dobule lr):
        cdef int i, j, k
        cdef double z, yPred, error, dw, db

        n = X.shape[0]
        m = X.shape[1]
        o = Y.shape[1]
        
        self.weight = cnp.zeros(n, dtype=cnp.float64)
        
        for i in range(epoch):
            for i in range(n):
                for j in range(m):
                    z = self.weight * X[i, j] + self.bias
                    yPred = 1 / (1 + cnp.exp(-z))
                    error = yPred - Y[i, j]

            dw = np.sum(X * error) / n
            db = np.sum(error) / n

            self.weight -= lr * dw
            self.bias -= lr * db

    def predict(self, double[:] X):
        cdef int n = X.shape[0]
        nparr = cnp.zeros(n, dtype=cnp.float64)
        cdef double[:] yPred = nparr
        cdef int i
        for i in range(n):
            yPred[i] = (self.weight * X[i]) + self.bias
        return nparr