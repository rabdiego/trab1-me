import matplotlib.pyplot as plt

BACKGROUND = '#ebdbb2'
FOREGROUND = '#1d2021'
GRID = '#3c3836'
RED = '#cc241d'
GREEN = '#98971a'
YELLOW = '#d79921'
BLUE = '#458588'
MAGENTA = '#b16286'
AQUA = '#689d6a'
ORANGE = '#d65d0e'
BRIGHT_RED = '#fb4934'
BRIGHT_GREEN = '#b8bb26'
BRIGHT_YELLOW = '#fabd2f'
BRIGHT_BLUE = '#83a598'
BRIGHT_MAGENTA = '#d3869b'
BRIGHT_AQUA = '#8ec07c'

PALETTE = [
    BLUE,
    RED,
    GREEN,
    YELLOW,
    MAGENTA,
    AQUA,
    ORANGE,
]


def apply_theme() -> None:
    plt.rcParams.update(
        {
            'figure.facecolor': BACKGROUND,
            'axes.facecolor': BACKGROUND,
            'savefig.facecolor': BACKGROUND,
            'text.color': FOREGROUND,
            'axes.labelcolor': FOREGROUND,
            'axes.edgecolor': GRID,
            'xtick.color': FOREGROUND,
            'ytick.color': FOREGROUND,
            'grid.color': GRID,
            'axes.grid': True,
            'grid.alpha': 0.35,
            'grid.linestyle': '--',
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.titleweight': 'bold',
            'legend.facecolor': BACKGROUND,
            'legend.edgecolor': GRID,
            'axes.prop_cycle': plt.cycler(color=PALETTE),
        }
    )
