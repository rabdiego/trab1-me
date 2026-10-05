from src.stats.BaseTransformer import BaseTransformer

import numpy as np

class MatrixInterceptor(BaseTransformer):
    def __init__(self) -> None:
        pass


    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(X.shape[0]), X]

