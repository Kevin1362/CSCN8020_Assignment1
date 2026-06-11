"""
utils.py
Utility functions for logging, tables, and grid display.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

State = Tuple[int, int]


def setup_logger(log_path: str = "logs/sample_execution.log") -> logging.Logger:
    """Create a logger that writes to file and console."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("CSCN8020_Assignment1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def values_to_dataframe(values: Dict[State, float], rows: int, cols: int) -> pd.DataFrame:
    """Convert state-value dictionary to grid DataFrame."""
    data = np.zeros((rows, cols))
    for (r, c), value in values.items():
        data[r, c] = value
    return pd.DataFrame(data, index=[f"row {i}" for i in range(rows)], columns=[f"col {j}" for j in range(cols)])


def policy_to_dataframe(policy: Dict[State, str], rows: int, cols: int) -> pd.DataFrame:
    """Convert policy dictionary to grid DataFrame with arrows."""
    arrows = {"right": "→", "down": "↓", "left": "←", "up": "↑", "G": "G"}
    data = [["" for _ in range(cols)] for _ in range(rows)]
    for (r, c), action in policy.items():
        data[r][c] = arrows.get(action, action)
    return pd.DataFrame(data, index=[f"row {i}" for i in range(rows)], columns=[f"col {j}" for j in range(cols)])


def plot_value_grid(values: Dict[State, float], rows: int, cols: int, title: str = "Value Function") -> None:
    """Plot value function grid with numbers."""
    grid = values_to_dataframe(values, rows, cols).values
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(grid)

    for r in range(rows):
        for c in range(cols):
            ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title(title)
    plt.show()


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
