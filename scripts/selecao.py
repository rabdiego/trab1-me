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
from src.stats.model_selection import *
from src.stats.statistics import *

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'data' / 'data.csv'
REPORT_DIR = ROOT_DIR / 'reports' / 'selecao'

FEATURES = [
    'Pre_Semester_GPA',
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Tool_Diversity',
    'Paid_Subscription',
]
SHORT = ['PreGPA', 'GenAIH', 'EstudoH', 'Divers', 'Assin']
TARGET = 'Post_Semester_GPA'
PRE_GPA = FEATURES.index('Pre_Semester_GPA')
ALPHA_ENTER = 0.05
ALPHA_REMOVE = 0.10

theme.apply_theme()
REPORT_DIR.mkdir(parents=True, exist_ok=True)

dl = DataLoader(pd.read_csv(DATA_PATH), FEATURES, TARGET)

X, y = dl.get_partitions()

n = X.shape[0]
q = X.shape[1]


def fit_columns(columns: list[int]) -> tuple[np.ndarray, LinearRegression]:
    design = MatrixInterceptor().transform(X[:, columns])
    model = LinearRegression()
    model.fit(design, y)
    return design, model


def metrics(columns: list[int]) -> dict:
    design, model = fit_columns(columns)
    aic, bic = information_criteria(design, y, model)
    return {
        'columns': list(columns),
        'aic': aic,
        'bic': bic,
        'r2': r_squared(design, y, model),
        'r2_adj': r_squared_adj(design, y, model),
    }


all_subsets = [metrics([j for j in range(q) if mask & (1 << j)]) for mask in range(2 ** q)]

has_pregpa = np.array([PRE_GPA in item['columns'] for item in all_subsets])

selected: list[int] = []
steps: list[dict] = []

for _ in range(2 * q + 1):
    current_design, _ = fit_columns(selected)

    best = None
    for j in range(q):
        if j in selected:
            continue
        candidate = sorted(selected + [j])
        candidate_design, _ = fit_columns(candidate)
        p_value = partial_f_p_value(current_design, candidate_design, y)
        if best is None or p_value < best['p']:
            best = {
                'j': j,
                'p': p_value,
                'f': partial_f_statistic(current_design, candidate_design, y),
                'extra': sum_of_squares_extra(current_design, candidate_design, y),
            }

    worst = None
    for j in selected:
        reduced = [c for c in selected if c != j]
        reduced_design, _ = fit_columns(reduced)
        p_value = partial_f_p_value(reduced_design, current_design, y)
        if worst is None or p_value > worst['p']:
            worst = {
                'j': j,
                'p': p_value,
                'f': partial_f_statistic(reduced_design, current_design, y),
                'extra': sum_of_squares_extra(reduced_design, current_design, y),
            }

    if best is not None and best['p'] < ALPHA_ENTER:
        selected.append(best['j'])
        selected.sort()
        action = f'entrou {FEATURES[best["j"]]}'
        record = best
    elif worst is not None and worst['p'] > ALPHA_REMOVE:
        selected.remove(worst['j'])
        action = f'saiu {FEATURES[worst["j"]]}'
        record = worst
    else:
        break

    current = metrics(selected)
    steps.append(
        {
            'action': action,
            'extra': record['extra'],
            'f': record['f'],
            'p': record['p'],
            'r2': current['r2'],
            'r2_adj': current['r2_adj'],
            'aic': current['aic'],
            'bic': current['bic'],
            'selected': list(selected),
        }
    )

initial = metrics([])
final = metrics(selected)
full = metrics(list(range(q)))

final_design, final_model = fit_columns(selected)
full_design, full_model = fit_columns(list(range(q)))

final_beta = final_model.get_beta()
final_standard_errors = standard_error(final_design, y, final_model)
final_t_statistics = t_statistic(final_design, y, final_model)
final_p_values = t_p_value(final_design, y, final_model)
final_intervals = confidence_interval(final_design, y, final_model)
final_labels = ['Intercept', *[FEATURES[j] for j in selected]]

extra = sum_of_squares_extra(final_design, full_design, y)
restrictions = full_design.shape[1] - final_design.shape[1]
partial_f = partial_f_statistic(final_design, full_design, y)
partial_p = partial_f_p_value(final_design, full_design, y)
excluded = [FEATURES[j] for j in range(q) if j not in selected]

best_aic = min(all_subsets, key=lambda item: item['aic'])
best_bic = min(all_subsets, key=lambda item: item['bic'])

lines = []
lines.append('=' * 78)
lines.append('ETAPA 3 - SELECAO DE MODELO E TESTE DE HIPOTESES')
lines.append('=' * 78)
lines.append('')
lines.append(f'Variavel resposta: {TARGET}')
lines.append(f'Candidatas: {", ".join(FEATURES)}')
lines.append(f'Observacoes (n): {n}')
lines.append('Metodo: forward-stepwise guiado por teste F parcial')
lines.append(f'alpha de entrada: {ALPHA_ENTER}')
lines.append(f'alpha de saida: {ALPHA_REMOVE}')
lines.append('AIC = n ln(SQRes/n) + 2p')
lines.append('BIC = n ln(SQRes/n) + p ln(n)')
lines.append('')
lines.append('-' * 78)
lines.append('1. Modelo completo')
lines.append('-' * 78)
lines.append(f'R2 = {full["r2"]:.6f}')
lines.append(f'R2 ajustado = {full["r2_adj"]:.6f}')
lines.append(f'AIC = {full["aic"]:.2f}')
lines.append(f'BIC = {full["bic"]:.2f}')
lines.append(f'F global = {f_statistic(full_design, y, full_model):.4f}')
lines.append('')
lines.append('-' * 78)
lines.append('2. Subconjuntos avaliados (32 modelos), ordenados por AIC')
lines.append('-' * 78)
lines.append(f'{"Regressoras":<38}{"R2":>10}{"R2aj":>10}{"AIC":>12}{"BIC":>12}')
lines.append('-' * 82)
for item in sorted(all_subsets, key=lambda value: value['aic']):
    names = '+'.join(SHORT[j] for j in item['columns']) if item['columns'] else '(nulo)'
    lines.append(
        f'{names:<38}{item["r2"]:>10.4f}{item["r2_adj"]:>10.4f}'
        f'{item["aic"]:>12.2f}{item["bic"]:>12.2f}'
    )
lines.append('')
lines.append(f'Melhor AIC: {", ".join(FEATURES[j] for j in best_aic["columns"]) or "(nulo)"} ({best_aic["aic"]:.2f})')
lines.append(f'Melhor BIC: {", ".join(FEATURES[j] for j in best_bic["columns"]) or "(nulo)"} ({best_bic["bic"]:.2f})')
lines.append('')
lines.append('-' * 78)
lines.append('3. Caminho stepwise')
lines.append('-' * 78)
lines.append(
    f'{"passo":>5}  {"acao":<32}{"SQExtra":>12}{"F":>12}{"p":>12}'
    f'{"R2":>10}{"R2aj":>10}{"AIC":>12}{"BIC":>12}'
)
for index, step in enumerate(steps):
    lines.append(
        f'{index + 1:>5}  {step["action"]:<32}{step["extra"]:>12.6f}'
        f'{step["f"]:>12.3f}{step["p"]:>12.4g}{step["r2"]:>10.4f}'
        f'{step["r2_adj"]:>10.4f}{step["aic"]:>12.2f}{step["bic"]:>12.2f}'
    )
lines.append('')
lines.append('-' * 78)
lines.append('4. Modelo selecionado')
lines.append('-' * 78)
lines.append(f'Regressoras: {", ".join(FEATURES[j] for j in selected) or "(nulo)"}')
lines.extend(coefficient_table(final_labels, final_beta, final_standard_errors, final_t_statistics, final_p_values, final_intervals))
lines.append('')
lines.append(f'R2 = {final["r2"]:.6f}')
lines.append(f'R2 ajustado = {final["r2_adj"]:.6f}')
lines.append(f'AIC = {final["aic"]:.2f}')
lines.append(f'BIC = {final["bic"]:.2f}')
lines.append(f'F global = {f_statistic(final_design, y, final_model):.4f}')
lines.append(f'p-valor do F global = {f_p_value(final_design, y, final_model):.6g}')
lines.append('')
lines.append('-' * 78)
lines.append('5. Teste F parcial: modelo selecionado vs modelo completo')
lines.append('-' * 78)
lines.append('H0: os coeficientes das variaveis excluidas sao conjuntamente nulos')
lines.append(f'Excluidas: {", ".join(excluded) or "(nenhuma)"}')
lines.append(f'Restricoes (r): {restrictions}')
lines.append(f'SQExtra = {extra:.6f}')
lines.append(f'F = {partial_f:.4f}')
lines.append(f'p-valor = {partial_p:.6g}')
if partial_p < ALPHA_ENTER:
    lines.append('Conclusao: rejeita H0, as variaveis excluidas agregam informacao.')
else:
    lines.append('Conclusao: nao rejeita H0, as variaveis excluidas nao agregam informacao.')
lines.append('')
lines.append('-' * 78)
lines.append('6. Comparacao de criterios')
lines.append('-' * 78)
lines.append(f'Nulo        AIC = {initial["aic"]:.2f}  BIC = {initial["bic"]:.2f}')
lines.append(f'Selecionado AIC = {final["aic"]:.2f}  BIC = {final["bic"]:.2f}')
lines.append(f'Completo    AIC = {full["aic"]:.2f}  BIC = {full["bic"]:.2f}')
lines.append('')
lines.append('Graficos')
lines.append('criterios.png')
lines.append('caminho_stepwise.png')
lines.append('r2_subconjuntos.png')
lines.append('coeficientes_final.png')
lines.append('')

(REPORT_DIR / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')

sizes = np.array([len(item['columns']) for item in all_subsets])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, key, title in zip(axes, ['aic', 'bic'], ['AIC', 'BIC']):
    values = np.array([item[key] for item in all_subsets])
    ax.scatter(sizes[has_pregpa], values[has_pregpa], s=30, alpha=0.75, color=theme.BLUE, label='com Pre_Semester_GPA')
    ax.scatter(sizes[~has_pregpa], values[~has_pregpa], s=30, alpha=0.75, color=theme.RED, label='sem Pre_Semester_GPA')
    path_x = [0] + [len(step['selected']) for step in steps]
    path_y = [initial[key]] + [step[key] for step in steps]
    ax.plot(path_x, path_y, color=theme.YELLOW, marker='o', linewidth=1.6, label='Caminho stepwise')
    best = min(all_subsets, key=lambda item: item[key])
    ax.scatter([len(best['columns'])], [best[key]], s=170, marker='*', color=theme.GREEN, label=f'Melhor {title}')
    ax.set_xlabel('Numero de regressoras')
    ax.set_ylabel(title)
    ax.set_title(title)
axes[0].legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / 'criterios.png', dpi=150)
plt.close(fig)

if steps:
    step_f = np.array([step['f'] for step in steps])
    colors = [theme.GREEN if step['action'].startswith('entrou') else theme.RED for step in steps]

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(steps) + 2))
    ax.barh(np.arange(len(steps)), step_f, color=colors)
    ax.set_xscale('log')
    ax.set_xlim(step_f.min()/3, step_f.max()*8)
    ax.set_yticks(np.arange(len(steps)))
    ax.set_yticklabels([f'{index + 1}. {step["action"]}' for index, step in enumerate(steps)])
    for index, step in enumerate(steps):
        label = 'p ~ 0' if step['p'] == 0 else f'p = {step["p"]:.2g}'
        ax.text(step['f'] * 1.15, index, label, va='center', color=theme.FOREGROUND, fontsize=9)
    ax.set_xlabel('Estatistica F parcial (escala log)')
    ax.set_title('Contribuicao de cada variavel no caminho stepwise')
    fig.tight_layout()
    fig.savefig(REPORT_DIR / 'caminho_stepwise.png', dpi=150)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
r2_values = np.array([item['r2'] for item in all_subsets])
r2_adj_values = np.array([item['r2_adj'] for item in all_subsets])
ax.scatter(sizes[has_pregpa], r2_values[has_pregpa], s=30, color=theme.BLUE, marker='o', label='R2 com PreGPA')
ax.scatter(sizes[~has_pregpa], r2_values[~has_pregpa], s=30, color=theme.RED, marker='o', label='R2 sem PreGPA')
ax.scatter(sizes[has_pregpa], r2_adj_values[has_pregpa], s=45, color=theme.BLUE, marker='^', facecolors='none', label='R2aj com PreGPA')
ax.scatter(sizes[~has_pregpa], r2_adj_values[~has_pregpa], s=45, color=theme.RED, marker='^', facecolors='none', label='R2aj sem PreGPA')
ax.set_xlabel('Numero de regressoras')
ax.set_ylabel('R2')
ax.set_title('R2 por subconjunto')
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / 'r2_subconjuntos.png', dpi=150)
plt.close(fig)

positions = np.arange(len(final_labels))
errors = np.vstack([final_beta - final_intervals[:, 0], final_intervals[:, 1] - final_beta])

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(final_beta, positions, xerr=errors, fmt='o', color=theme.BLUE, ecolor=theme.MAGENTA, capsize=4)
ax.axvline(0, color=theme.RED, linestyle='--', linewidth=1)
ax.set_xscale('symlog', linthresh=1e-4)
ax.set_xlim(0, 1e1)
ax.set_xticks([0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
ax.set_yticks(positions)
ax.set_yticklabels(final_labels)
ax.set_xlabel('Coeficiente (escala log simetrica)')
ax.set_title('Modelo selecionado: coeficientes com IC 95%')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'coeficientes_final.png', dpi=150)
plt.close(fig)
