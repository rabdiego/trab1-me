import pandas as pd
from pathlib import Path

from src.models.LinearRegression import LinearRegression
from src.models.DataLoader import DataLoader
from src.stats.statistics import *
from src.stats.StandardScaler import StandardScaler
from src.stats.MatrixInterceptor import MatrixInterceptor

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "data.csv"

dl = DataLoader(
    pd.read_csv(DATA_PATH),
    ['Pre_Semester_GPA', 'Weekly_GenAI_Hours'],
    'Post_Semester_GPA'
)

lr = LinearRegression()
scaler = StandardScaler()

X, y = dl.get_partitions()

X = MatrixInterceptor().transform(X)

lr.fit(X, y)
y_pred = lr.predict(X)

print(residual_squared_sum(X, y, lr))
print(regression_squared_sum(X, y, lr))
print(total_squared_sum(y))
print(r_squared(X, y, lr))
print(r_squared_adj(X, y, lr))