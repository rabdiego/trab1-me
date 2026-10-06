import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from src.models.DataLoader import DataLoader
from src.models.LinearRegression import LinearRegression
from src.plots import theme
from src.report import coefficient_table
from src.stats.inference import *
from src.stats.MatrixInterceptor import MatrixInterceptor
from src.stats.statistics import *

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'data' / 'data.csv'
REPORT_DIR = ROOT_DIR / 'reports' / 'ajuste_modelo'

FEATURES = [
    'Pre_Semester_GPA',
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Tool_Diversity',
    'Paid_Subscription',
]
TARGET = 'Post_Semester_GPA'

theme.apply_theme()
REPORT_DIR.mkdir(parents=True, exist_ok=True)

dl = DataLoader(pd.read_csv(DATA_PATH), FEATURES, TARGET)

model = LinearRegression()

X, y = dl.get_partitions()

X = MatrixInterceptor().transform(X)

model.fit(X, y)

beta = model.get_beta()
standard_errors = standard_error(X, y, model)
t_statistics = t_statistic(X, y, model)
p_values = t_p_value(X, y, model)
intervals = confidence_interval(X, y, model)

labels = ['Intercept', *FEATURES]

n = X.shape[0]
p = X.shape[1]

lines = []
lines.append('=' * 70)
lines.append('ETAPA 2 - AJUSTE DO MODELO E INFERENCIA GLOBAL')
lines.append('=' * 70)
lines.append('')
lines.append(f'Variavel resposta: {TARGET}')
lines.append(f'Regressoras: {", ".join(FEATURES)}')
lines.append(f'Observacoes (n): {n}')
lines.append(f'Parametros (p): {p}')
lines.append('')
lines.append('Tabela de coeficientes')
lines.extend(coefficient_table(labels, beta, standard_errors, t_statistics, p_values, intervals))
lines.append('')
lines.append('Somas de quadrados')
lines.append(f'SQRES = {residual_squared_sum(X, y, model):.6f}')
lines.append(f'SQREG = {regression_squared_sum(X, y, model):.6f}')
lines.append(f'SQTOT = {total_squared_sum(y):.6f}')
lines.append('')
lines.append('Qualidade do ajuste')
lines.append(f'R2 = {r_squared(X, y, model):.6f}')
lines.append(f'R2 ajustado = {r_squared_adj(X, y, model):.6f}')
lines.append(f'Variancia residual = {residual_variance(X, y, model):.6f}')
lines.append(f'Desvio padrao residual = {np.sqrt(residual_variance(X, y, model)):.6f}')
lines.append('')
lines.append('Teste F global do modelo')
lines.append('H0: todos os coeficientes, exceto o intercepto, sao nulos')
lines.append(f'F = {f_statistic(X, y, model):.4f}')
lines.append(f'Graus de liberdade = ({p - 1}, {n - p})')
lines.append(f'p-valor = {f_p_value(X, y, model):.6g}')
lines.append('')
lines.append('Graficos')
lines.append('coeficientes.png')
lines.append('ajustado_vs_observado.png')
lines.append('')

(REPORT_DIR / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')

positions = np.arange(len(labels))
errors = np.vstack([beta - intervals[:, 0], intervals[:, 1] - beta])

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(beta, positions, xerr=errors, fmt='o', color=theme.BLUE, ecolor=theme.MAGENTA, capsize=4)
ax.axvline(0, color=theme.RED, linestyle='--', linewidth=1)
ax.set_xscale('symlog', linthresh=1e-6)
ax.set_xlim(-1e-4, 1e1)
ax.set_xticks([-1e-4, -1e-5, 0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
ax.set_yticks(positions)
ax.set_yticklabels(labels)
ax.set_xlabel('Coeficiente (escala log simetrica)')
ax.set_title('Coeficientes com intervalo de confianca de 95%')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'coeficientes.png', dpi=150)
plt.close(fig)

fitted = model.predict(X)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(fitted, y, s=6, alpha=0.2, color=theme.BLUE)
limits = [min(fitted.min(), y.min()), max(fitted.max(), y.max())]
ax.plot(limits, limits, color=theme.RED, linewidth=1.5)
ax.set_xlabel('Valores ajustados')
ax.set_ylabel('Valores observados')
ax.set_title('Ajustados vs observados')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'ajustado_vs_observado.png', dpi=150)
plt.close(fig)
