import numpy as np
from scipy import stats

from src.models.LinearRegression import LinearRegression
from src.stats.inference import residual_variance
from src.stats.model_selection import partial_f_p_value, partial_f_statistic
from src.stats.statistics import r_squared


def fitted_values(X: np.ndarray, model: LinearRegression) -> np.ndarray:
    return model.predict(X)


def residuals(X: np.ndarray, y: np.ndarray, model: LinearRegression) -> np.ndarray:
    return y - model.predict(X)


def leverage(X: np.ndarray) -> np.ndarray:
    inverse = np.linalg.solve(X.T@X, np.eye(X.shape[1]))
    return (X@inverse*X).sum(axis=1)


def standardized_residuals(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> np.ndarray:
    return residuals(X, y, model)/np.sqrt(residual_variance(X, y, model))


def studentized_residuals(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> np.ndarray:
    hat = leverage(X)
    return residuals(X, y, model)/np.sqrt(residual_variance(X, y, model)*(1 - hat))


def cooks_distance(
    X: np.ndarray,
    y: np.ndarray,
    model: LinearRegression
) -> np.ndarray:
    p = X.shape[1]
    hat = leverage(X)
    studentized = studentized_residuals(X, y, model)
    return (studentized**2/p)*(hat/(1 - hat))


def variance_inflation_factors(X: np.ndarray) -> np.ndarray:
    predictors = X[:, 1:]
    count = predictors.shape[1]
    factors = np.empty(count)
    for j in range(count):
        others = [k for k in range(count) if k != j]
        design = np.column_stack([np.ones(X.shape[0]), predictors[:, others]])
        model = LinearRegression()
        model.fit(design, predictors[:, j])
        factors[j] = 1/(1 - r_squared(design, predictors[:, j], model))
    return factors


def chi_square_normality(
    studentized: np.ndarray,
    bins: int = 10
) -> tuple[float, float, np.ndarray, np.ndarray]:
    edges = stats.norm.ppf(np.linspace(0, 1, bins + 1))
    observed, _ = np.histogram(studentized, bins=edges)
    expected = np.full(bins, studentized.size/bins)
    statistic = ((observed - expected)**2/expected).sum()
    return statistic, stats.chi2.sf(statistic, bins - 1), observed, expected


def quadratic_term_tests(
    X: np.ndarray,
    y: np.ndarray
) -> list[tuple[int, float | None, float | None]]:
    results = []
    for j in range(1, X.shape[1]):
        if np.unique(X[:, j]).size <= 2:
            results.append((j, None, None))
            continue
        features = np.column_stack([X, X[:, j]**2])
        results.append(
            (j, partial_f_statistic(X, features, y), partial_f_p_value(X, features, y))
        )
    return results
