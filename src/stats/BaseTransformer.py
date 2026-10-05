from abc import ABC, abstractmethod
import numpy as np

class BaseTransformer:
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

