import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats

from src.models.DataLoader import DataLoader
from src.models.LinearRegression import LinearRegression
from src.plots import theme
from src.stats.diagnostics import *
from src.stats.MatrixInterceptor import MatrixInterceptor
from src.stats.statistics import residual_squared_sum, total_squared_sum

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'data' / 'data.csv'
REPORT_DIR = ROOT_DIR / 'reports' / 'diagnostico'

FEATURES = [
    'Pre_Semester_GPA',
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Tool_Diversity',
    'Paid_Subscription',
]
SELECTED = [
    'Pre_Semester_GPA',
    'Traditional_Study_Hours',
    'Tool_Diversity',
    'Paid_Subscription',
]
TARGET = 'Post_Semester_GPA'

theme.apply_theme()
REPORT_DIR.mkdir(parents=True, exist_ok=True)

dl = DataLoader(pd.read_csv(DATA_PATH), FEATURES, TARGET)

X_all, y = dl.get_partitions()

selected = [FEATURES.index(feature) for feature in SELECTED]

X = MatrixInterceptor().transform(X_all[:, selected])

model = LinearRegression()
model.fit(X, y)

n = X.shape[0]
p = X.shape[1]

error = residuals(X, y, model)
fitted = fitted_values(X, model)
hat = leverage(X)
standardized = standardized_residuals(X, y, model)
studentized = studentized_residuals(X, y, model)
cooks = cooks_distance(X, y, model)

leverage_threshold = 2 * p / n
cook_threshold = 4 / n

within_two = 100 * (np.abs(standardized) < 2).mean()
above_two = int((np.abs(standardized) >= 2).sum())
above_three = int((np.abs(studentized) > 3).sum())
leverage_count = int((hat > leverage_threshold).sum())
cook_count = int((cooks > cook_threshold).sum())
top_cooks = np.argsort(cooks)[::-1][:5]

chi_statistic, chi_p_value, chi_observed, chi_expected = chi_square_normality(studentized)
quadratic = quadratic_term_tests(X, y)
vif = variance_inflation_factors(X)

lines = []
lines.append('=' * 78)
lines.append('ETAPA 5 - DIAGNOSTICO E VALIDACAO DOS PRESSUPOSTOS')
lines.append('=' * 78)
lines.append('')
lines.append('Ferramentas do livro texto: residuo bruto, padronizado e studentizado;')
lines.append('alavancagem; distancia de Cook; QQ-plot; teste chi-quadrado de aderencia;')
lines.append('F parcial para termos quadraticos; VIF; graficos de residuos.')
lines.append('')
lines.append(f'Variavel resposta: {TARGET}')
lines.append(f'Modelo diagnosticado (selecionado na etapa 3): {", ".join(SELECTED)}')
lines.append(f'Observacoes (n): {n}')
lines.append(f'Parametros (p): {p}')
lines.append(f'R2 = {1 - residual_squared_sum(X, y, model)/total_squared_sum(y):.6f}')
lines.append(f'MQRes = {residual_squared_sum(X, y, model)/(n - p):.6f}')
lines.append('')
lines.append('-' * 78)
lines.append('1. Residuos')
lines.append('-' * 78)
lines.append('Residuo bruto e_i = y_i - y_hat_i')
lines.append(f'  Media = {error.mean():.6f}')
lines.append(f'  Desvio padrao = {error.std(ddof=1):.6f}')
lines.append(f'  Minimo = {error.min():.6f}')
lines.append(f'  Maximo = {error.max():.6f}')
lines.append('')
lines.append('Residuo padronizado d_i = e_i / sqrt(MQRes)')
lines.append(f'  Dentro de (-2, +2): {within_two:.4f}% (esperado ~95%)')
lines.append(f'  Fora de (-2, +2): {above_two}')
lines.append('')
lines.append('Residuo studentizado r_i = e_i / sqrt(MQRes (1 - h_ii))')
lines.append(f'  Com |r_i| > 3: {above_three}')
lines.append('')
lines.append('-' * 78)
lines.append('2. Normalidade dos erros')
lines.append('-' * 78)
lines.append('Ferramenta grafica: QQ-plot dos residuos studentizados (qq_residuos.png)')
lines.append('Teste de aderencia chi-quadrado')
lines.append('H0: os residuos seguem distribuicao normal')
lines.append(f'  chi2 = {chi_statistic:.4f}')
lines.append(f'  Graus de liberdade = {chi_observed.size - 1}')
lines.append(f'  p-valor = {chi_p_value:.6g}')
lines.append('  Classes de igual probabilidade sob N(0, 1):')
for index in range(chi_observed.size):
    lines.append(f'    classe {index + 1:>2}: observado = {chi_observed[index]:>6}  esperado = {chi_expected[index]:.1f}')
lines.append('')
lines.append('-' * 78)
lines.append('3. Homocedasticidade')
lines.append('-' * 78)
lines.append('Ferramentas graficas: residuos vs valores ajustados e residuos vs regressores')
lines.append('(residuos_vs_ajustados.png e residuos_vs_preditores.png).')
lines.append('Padrao satisfatorio: dispersao aleatoria em torno de zero, sem funil.')
lines.append('Padrao em funil: variancia cresce com a resposta; pede transformacao em Y.')
lines.append('')
lines.append('-' * 78)
lines.append('4. Linearidade e inadequacao do modelo')
lines.append('-' * 78)
lines.append('Ferramenta grafica: residuos vs regressores e residuos vs ajustados.')
lines.append('Teste F parcial para o termo quadratico x_j^2 (soma de quadrados extra)')
lines.append(f'{"Regressor":<28}{"F":>12}{"p-valor":>14}')
for j, f_statistic, p_value in quadratic:
    label = SELECTED[j - 1]
    if f_statistic is None:
        lines.append(f'{label:<28}{"binaria":>12}{"nao se aplica":>14}')
    else:
        lines.append(f'{label:<28}{f_statistic:>12.4f}{p_value:>14.6g}')
lines.append('')
lines.append('-' * 78)
lines.append('5. Independencia dos erros')
lines.append('-' * 78)
lines.append('Ferramenta grafica: residuos em sequencia de coleta (residuos_sequencia.png).')
lines.append('Nota: o dataset nao tem ordem natural (amostra embaralhada), portanto os')
lines.append('padroes ciclicos ou blocos de mesmo sinal nao sao interpretaveis aqui.')
lines.append('')
lines.append('-' * 78)
lines.append('6. Multicolinearidade')
lines.append('-' * 78)
lines.append('Fator de inflacao da variancia FIV_j = 1 / (1 - R_j^2)')
lines.append(f'{"Variavel":<28}{"R2 auxiliar":>14}{"FIV":>10}')
for index, label in enumerate(SELECTED):
    lines.append(f'{label:<28}{1 - 1/vif[index]:>14.6f}{vif[index]:>10.4f}')
lines.append('Referencia do livro: FIV > 5 ou 10 indica multicolinearidade severa.')
lines.append('')
lines.append('-' * 78)
lines.append('7. Pontos de alavancagem e influentes')
lines.append('-' * 78)
lines.append('Alavancagem h_ii (diagonal de H = X (X^T X)^-1 X^T)')
lines.append(f'  Minimo = {hat.min():.8f}')
lines.append(f'  Maximo = {hat.max():.8f}')
lines.append(f'  h_ii > 2p/n = {leverage_threshold:.8f}: {leverage_count}')
lines.append('')
lines.append('Distancia de Cook D_i = (r_i^2 / p) * h_ii / (1 - h_ii)')
lines.append(f'  D_i > 4/n = {cook_threshold:.8f}: {cook_count} ({100*cook_count/n:.4f}%)')
lines.append('  Cinco maiores distancias:')
lines.append(f'{"indice":>8}{"h_ii":>12}{"r studentizado":>16}{"Cook":>12}')
for index in top_cooks:
    lines.append(
        f'{index:>8}{hat[index]:>12.6f}{studentized[index]:>16.6f}{cooks[index]:>12.6f}'
    )
lines.append('')
lines.append('-' * 78)
lines.append('8. Conclusao')
lines.append('-' * 78)
if chi_p_value < 0.05:
    lines.append(f'Normalidade: rejeitada pelo teste chi-quadrado (p = {chi_p_value:.4g});')
    lines.append(f'  {within_two:.2f}% dos residuos padronizados em (-2, +2) (esperado ~95%).')
else:
    lines.append('Normalidade: nao rejeitada pelo teste chi-quadrado.')
lines.append('Homocedasticidade: avaliar visualmente os graficos de residuos.')
lines.append('Linearidade: ver discriminacao dos termos quadraticos acima.')
if vif.max() > 10:
    lines.append('Multicolinearidade: FIV acima de 10 em alguma variavel.')
elif vif.max() > 5:
    lines.append('Multicolinearidade: FIV entre 5 e 10 em alguma variavel.')
else:
    lines.append('Multicolinearidade: ausente (FIV abaixo de 5).')
lines.append(f'Influencia: {cook_count} observacoes com D_i > 4/n ({100*cook_count/n:.4f}%).')
lines.append('')
lines.append('Graficos')
lines.append('residuos_vs_ajustados.png')
lines.append('residuos_vs_preditores.png')
lines.append('qq_residuos.png')
lines.append('qui_quadrado.png')
lines.append('residuos_sequencia.png')
lines.append('influencia.png')
lines.append('cooks.png')
lines.append('vif.png')
lines.append('')

(REPORT_DIR / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(fitted, error, s=6, alpha=0.25, color=theme.BLUE)
ax.axhline(0, color=theme.RED, linewidth=1.2)
ax.set_xlabel('Valores ajustados')
ax.set_ylabel('Residuos brutos')
ax.set_title('Residuos vs valores ajustados')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'residuos_vs_ajustados.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, index in zip(axes.ravel(), range(len(SELECTED))):
    ax.scatter(X[:, index + 1], error, s=5, alpha=0.2, color=theme.BLUE)
    ax.axhline(0, color=theme.RED, linewidth=1)
    ax.set_xlabel(SELECTED[index])
    ax.set_ylabel('Residuos brutos')
    ax.set_title(f'Residuos vs {SELECTED[index]}')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'residuos_vs_preditores.png', dpi=150)
plt.close(fig)

(osm, osr), (slope, intercept, _) = stats.probplot(studentized, dist='norm')

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(osm, osr, s=5, alpha=0.3, color=theme.BLUE)
ax.plot(osm, slope*osm + intercept, color=theme.RED, linewidth=1.5)
ax.set_xlabel('Quantis teoricos')
ax.set_ylabel('Residuos studentizados ordenados')
ax.set_title('QQ-plot dos residuos studentizados')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'qq_residuos.png', dpi=150)
plt.close(fig)

positions = np.arange(chi_observed.size)
width = 0.4

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(positions - width/2, chi_observed, width, color=theme.BLUE, label='Observado')
ax.bar(positions + width/2, chi_expected, width, color=theme.RED, label='Esperado')
ax.set_xticks(positions)
ax.set_xticklabels([f'{index + 1}' for index in positions])
ax.set_xlabel('Classe de igual probabilidade')
ax.set_ylabel('Frequencia')
ax.set_title(f'Teste chi-quadrado de aderencia (chi2 = {chi_statistic:.1f}, p = {chi_p_value:.3g})')
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / 'qui_quadrado.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(np.arange(n), error, s=5, alpha=0.2, color=theme.BLUE)
ax.axhline(0, color=theme.RED, linewidth=1.2)
ax.set_xlabel('Ordem de coleta (indice da observacao)')
ax.set_ylabel('Residuos brutos')
ax.set_title('Residuos em sequencia de coleta')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'residuos_sequencia.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
sizes = 15 + 900*cooks/cooks.max()
ax.scatter(hat, studentized, s=sizes, alpha=0.4, color=theme.BLUE)
ax.axhline(0, color=theme.FOREGROUND, linewidth=0.8)
ax.axhline(2, color=theme.RED, linestyle='--', linewidth=1)
ax.axhline(-2, color=theme.RED, linestyle='--', linewidth=1)
ax.axvline(leverage_threshold, color=theme.YELLOW, linestyle='--', linewidth=1)
ax.set_xlabel('Alavancagem (h_ii)')
ax.set_ylabel('Residuo studentizado')
ax.set_title('Alavancagem vs residuo studentizado (tamanho = distancia de Cook)')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'influencia.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(np.arange(n), cooks, s=7, alpha=0.4, color=theme.BLUE)
ax.axhline(cook_threshold, color=theme.RED, linestyle='--', linewidth=1, label='4/n')
ax.set_xlabel('Observacao')
ax.set_ylabel('Distancia de Cook')
ax.set_title('Distancia de Cook por observacao')
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / 'cooks.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(SELECTED, vif, color=theme.BLUE)
ax.axvline(5, color=theme.YELLOW, linestyle='--', linewidth=1, label='5')
ax.axvline(10, color=theme.RED, linestyle='--', linewidth=1, label='10')
ax.set_xlabel('FIV')
ax.set_title('Fator de inflacao da variancia')
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / 'vif.png', dpi=150)
plt.close(fig)
