import numpy as np
from scipy import stats

from src.models.LinearRegression import LinearRegression
from src.stats.MatrixInterceptor import MatrixInterceptor
from src.stats.statistics import regression_squared_sum, residual_squared_sum


def residual_variance(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> float:
    n = X.shape[0]
    p = X.shape[1]
    return residual_squared_sum(X, y, model) / (n - p)


def covariance_matrix(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> np.ndarray:
    return residual_variance(X, y, model) * np.linalg.solve(X.T@X, np.eye(X.shape[1]))


def standard_error(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> np.ndarray:
    return np.sqrt(np.diag(covariance_matrix(X, y, model)))


def t_statistic(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> np.ndarray:
    return model.get_beta() / standard_error(X, y, model)


def t_p_value(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> np.ndarray:
    n = X.shape[0]
    p = X.shape[1]
    return 2 * stats.t.sf(np.abs(t_statistic(X, y, model)), n - p)


def confidence_interval(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression,
    confidence_level: float = 0.95
) -> np.ndarray:
    n = X.shape[0]
    p = X.shape[1]
    critical_value = stats.t.ppf((1 + confidence_level) / 2, n - p)
    beta = model.get_beta()
    error = standard_error(X, y, model)
    return np.column_stack((beta - critical_value*error, beta + critical_value*error))


def mean_response_interval(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression,
    x0: np.ndarray,
    confidence_level: float = 0.95
) -> np.ndarray:
    n = X.shape[0]
    p = X.shape[1]
    x0_intercept = MatrixInterceptor().transform(x0.reshape(1, -1)).ravel()
    prediction = x0_intercept@model.get_beta()
    leverage = x0_intercept@np.linalg.solve(X.T@X, np.eye(X.shape[1]))@x0_intercept
    critical_value = stats.t.ppf((1 + confidence_level) / 2, n - p)
    half_width = critical_value*np.sqrt(residual_variance(X, y, model)*leverage)
    return np.array([prediction - half_width, prediction + half_width])


def prediction_interval(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression,
    x0: np.ndarray,
    confidence_level: float = 0.95
) -> np.ndarray:
    n = X.shape[0]
    p = X.shape[1]
    x0_intercept = MatrixInterceptor().transform(x0.reshape(1, -1)).ravel()
    prediction = x0_intercept@model.get_beta()
    leverage = x0_intercept@np.linalg.solve(X.T@X, np.eye(X.shape[1]))@x0_intercept
    critical_value = stats.t.ppf((1 + confidence_level) / 2, n - p)
    half_width = critical_value*np.sqrt(residual_variance(X, y, model)*(leverage + 1))
    return np.array([prediction - half_width, prediction + half_width])


def f_statistic(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> float:
    n = X.shape[0]
    p = X.shape[1]
    numerator = regression_squared_sum(X, y, model) / (p - 1)
    denominator = residual_squared_sum(X, y, model) / (n - p)
    return numerator / denominator


def f_p_value(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> float:
    n = X.shape[0]
    p = X.shape[1]
    return stats.f.sf(f_statistic(X, y, model), p - 1, n - p)
