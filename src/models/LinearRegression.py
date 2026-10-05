from src.models.BasePredictor import BasePredictor

import numpy as np

class LinearRegression(BasePredictor):
    def __init__(self) -> None:
        self.__beta: np.ndarray = None
    

    def get_beta(self) -> np.ndarray:
        return self.__beta


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__beta = np.linalg.solve(X.T@X, X.T@y)


    def predict(self, X: np.ndarray) -> np.ndarray:
        return X@self.__beta

