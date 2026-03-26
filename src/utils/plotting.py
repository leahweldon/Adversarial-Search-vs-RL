from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt


def moving_average(values: List[float], window_size: int) -> List[float]:
    """Compute simple moving average"""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = values[start : i + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed


def plot_training_curve(
    values: List[float],
    title: str,
    y_label: str,
    output_path: str | Path,
    smoothing_window: Optional[int] = None,
) -> None:
    """ Save single training curve to disk"""
    if not values:
        raise ValueError("Cannot plot an empty list of values.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = list(range(1, len(values) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, values, label="Raw values")

    if smoothing_window is not None and smoothing_window > 1:
        smoothed = moving_average(values, smoothing_window)
        plt.plot(x_values, smoothed, label=f"Moving average ({smoothing_window})")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()