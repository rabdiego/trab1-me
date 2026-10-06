import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.plots import theme

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / 'data' / 'data.csv'
REPORT_DIR = ROOT_DIR / 'reports' / 'eda'

PREDICTORS = [
    'Pre_Semester_GPA',
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Tool_Diversity',
    'Paid_Subscription',
]
TARGET = 'Post_Semester_GPA'

theme.apply_theme()
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
study = df[PREDICTORS + [TARGET]].astype(float)
correlation = study.corr()
target_correlation = correlation[TARGET]

summary = study.describe().T

lines = []
lines.append('=' * 78)
lines.append('ETAPA 1 - ANALISE EXPLORATORIA DOS DADOS')
lines.append('=' * 78)
lines.append('')
lines.append(f'Variavel resposta: {TARGET}')
lines.append(f'Preditores: {", ".join(PREDICTORS)}')
lines.append(f'Observacoes (n): {len(df)}')
lines.append('')
lines.append('-' * 78)
lines.append('Estatisticas descritivas')
lines.append('-' * 78)
lines.append(f'{"Variavel":<28}{"media":>10}{"desvio":>10}{"min":>10}{"q1":>10}{"mediana":>10}{"q3":>10}{"max":>10}')
for name in PREDICTORS + [TARGET]:
    row = summary.loc[name]
    lines.append(
        f'{name:<28}{row["mean"]:>10.4f}{row["std"]:>10.4f}{row["min"]:>10.4f}'
        f'{row["25%"]:>10.4f}{row["50%"]:>10.4f}{row["75%"]:>10.4f}{row["max"]:>10.4f}'
    )
lines.append('')
lines.append('-' * 78)
lines.append(f'Correlacao (Pearson) com {TARGET}')
lines.append('-' * 78)
for name in PREDICTORS:
    lines.append(f'{name:<28}{target_correlation[name]:>10.4f}')
lines.append('')
lines.append('-' * 78)
lines.append('Matriz de correlacao')
lines.append('-' * 78)
lines.append(f'{"":<28}' + ''.join(f'{name[:8]:>10}' for name in PREDICTORS + [TARGET]))
for name in PREDICTORS + [TARGET]:
    values = ''.join(f'{correlation.loc[name, other]:>10.4f}' for other in PREDICTORS + [TARGET])
    lines.append(f'{name:<28}{values}')
lines.append('')
lines.append('Graficos')
lines.append('preditores_vs_alvo.png')
lines.append('correlacoes.png')
lines.append('distribuicoes.png')
lines.append('')

(REPORT_DIR / 'report.txt').write_text('\n'.join(lines), encoding='utf-8')

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, name in zip(axes.ravel(), PREDICTORS):
    x = study[name].to_numpy()
    y = study[TARGET].to_numpy()
    ax.scatter(x, y, s=5, alpha=0.15, color=theme.BLUE)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, slope*grid + intercept, color=theme.RED, linewidth=1.5)
    ax.set_xlabel(name)
    ax.set_ylabel(TARGET)
    ax.set_title(f'{name} (r = {target_correlation[name]:.2f})')

target_ax = axes.ravel()[len(PREDICTORS)]
target_ax.hist(study[TARGET], bins=60, color=theme.BLUE)
target_ax.set_xlabel(TARGET)
target_ax.set_ylabel('Frequencia')
target_ax.set_title(f'Distribuicao de {TARGET}')

fig.tight_layout()
fig.savefig(REPORT_DIR / 'preditores_vs_alvo.png', dpi=150)
plt.close(fig)

cmap = LinearSegmentedColormap.from_list('gruvbox', [theme.RED, theme.BACKGROUND, theme.BLUE])
names = PREDICTORS + [TARGET]

fig, ax = plt.subplots(figsize=(9, 8))
image = ax.imshow(correlation.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(names)))
ax.set_yticks(np.arange(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f'{correlation.iloc[i, j]:.2f}', ha='center', va='center', color=theme.FOREGROUND)
ax.set_title('Matriz de correlacao')
ax.tick_params(axis='both', labelsize=9)
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(REPORT_DIR / 'correlacoes.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, name in zip(axes.ravel(), PREDICTORS + [TARGET]):
    ax.hist(study[name], bins=60, color=theme.BLUE)
    ax.set_xlabel(name)
    ax.set_ylabel('Frequencia')
    ax.set_title(f'Distribuicao de {name}')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'distribuicoes.png', dpi=150)
plt.close(fig)
