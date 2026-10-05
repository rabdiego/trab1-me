import numpy as np
from src.models.LinearRegression import LinearRegression

def residual_squared_sum(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> float:
    return y@y.T - model.get_beta().T@X.T@y


def regression_squared_sum(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> float:
    return model.get_beta().T@X.T@y - ((y.sum())**2)/y.shape[0]


def total_squared_sum(y: np.ndarray) -> float:
    return y.T@y - ((y.sum())**2)/y.shape[0]


def r_squared(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> float:
    return 1 - residual_squared_sum(X, y, model)/total_squared_sum(y)


def r_squared_adj(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> float:
    n = X.shape[0]
    p = X.shape[1]
    return 1 - ((n-1)/(n-p))*(residual_squared_sum(X, y, model)/total_squared_sum(y))

