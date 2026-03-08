from abc import ABC, abstractmethod
import numpy as np


class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
