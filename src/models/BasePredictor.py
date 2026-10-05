from abc import ABC, abstractmethod
import numpy as np

class BasePredictor(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

