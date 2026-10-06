import numpy as np


def coefficient_table(
    labels: list[str],
    beta: np.ndarray,
    standard_errors: np.ndarray,
    t_statistics: np.ndarray,
    p_values: np.ndarray,
    intervals: np.ndarray
) -> list[str]:
    header = (
        f'{"Variavel":<22}{"beta":>12}{"erro-padrao":>14}'
        f'{"t":>12}{"p-valor":>14}{"IC95%":>30}'
    )
    lines = [header, '-' * 104]
    for index, label in enumerate(labels):
        interval = f'[{intervals[index, 0]: .5f}; {intervals[index, 1]: .5f}]'
        lines.append(
            f'{label:<22}{beta[index]:>12.5f}{standard_errors[index]:>14.5f}'
            f'{t_statistics[index]:>12.3f}{p_values[index]:>14.4g}{interval:>30}'
        )
    return lines
