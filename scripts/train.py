import numpy as np
import pandas as pd
from pathlib import Path

from src.models.DataLoader import DataLoader
from src.models.LinearRegression import LinearRegression
from src.stats.inference import *
from src.stats.MatrixInterceptor import MatrixInterceptor
from src.stats.statistics import *

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "data.csv"

FEATURES = ['Pre_Semester_GPA', 'Weekly_GenAI_Hours']
TARGET = 'Post_Semester_GPA'

dl = DataLoader(pd.read_csv(DATA_PATH), FEATURES, TARGET)

lr = LinearRegression()

X, y = dl.get_partitions()

X = MatrixInterceptor().transform(X)

lr.fit(X, y)

beta = lr.get_beta()
standard_errors = standard_error(X, y, lr)
t_statistics = t_statistic(X, y, lr)
p_values = t_p_value(X, y, lr)
intervals = confidence_interval(X, y, lr)

labels = ['Intercept', *FEATURES]

print('Coefficient table')
for index, label in enumerate(labels):
    print(
        f'{label:<22} '
        f'beta={beta[index]: .5f} '
        f'se={standard_errors[index]:.5f} '
        f't={t_statistics[index]: .3f} '
        f'p={p_values[index]:.4g} '
        f'IC95=[{intervals[index, 0]: .5f}, {intervals[index, 1]: .5f}]'
    )

print(f'R2 = {r_squared(X, y, lr):.5f}')
print(f'R2 adjusted = {r_squared_adj(X, y, lr):.5f}')
print(f'Residual variance = {residual_variance(X, y, lr):.5f}')
print(f'F = {f_statistic(X, y, lr):.3f}')
print(f'F p-value = {f_p_value(X, y, lr):.4g}')

x0 = np.array([3.5, 5.0])

print(f'IC mean response = {mean_response_interval(X, y, lr, x0)}')
print(f'IP = {prediction_interval(X, y, lr, x0)}')
