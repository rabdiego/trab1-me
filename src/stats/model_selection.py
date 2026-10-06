import numpy as np
from scipy import stats

from src.models.LinearRegression import LinearRegression
from src.stats.statistics import residual_squared_sum


def sum_of_squares_extra(
    X_reduced: np.ndarray,
    X_full: np.ndarray,
    y: np.ndarray
) -> float:
    reduced = LinearRegression()
    reduced.fit(X_reduced, y)
    full = LinearRegression()
    full.fit(X_full, y)
    return residual_squared_sum(X_reduced, y, reduced) - residual_squared_sum(X_full, y, full)


def partial_f_statistic(
    X_reduced: np.ndarray,
    X_full: np.ndarray,
    y: np.ndarray
) -> float:
    full = LinearRegression()
    full.fit(X_full, y)
    n = X_full.shape[0]
    p = X_full.shape[1]
    r = p - X_reduced.shape[1]
    extra = sum_of_squares_extra(X_reduced, X_full, y)
    return (extra/r)/(residual_squared_sum(X_full, y, full)/(n - p))


def partial_f_p_value(
    X_reduced: np.ndarray,
    X_full: np.ndarray,
    y: np.ndarray
) -> float:
    n = X_full.shape[0]
    p = X_full.shape[1]
    r = p - X_reduced.shape[1]
    return stats.f.sf(partial_f_statistic(X_reduced, X_full, y), r, n - p)


def information_criteria(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> tuple[float, float]:
    n = X.shape[0]
    p = X.shape[1]
    rss = residual_squared_sum(X, y, model)
    aic = n*np.log(rss/n) + 2*p
    bic = n*np.log(rss/n) + p*np.log(n)
    return aic, bic
