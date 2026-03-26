from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std_dev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def aggregate_rows(
    rows: List[dict],
    group_keys: List[str],
    metric_keys: List[str],
) -> List[dict]:
    """Groups rows by supplied keys and computes mean/std for metric keys"""
    grouped: Dict[tuple, List[dict]] = defaultdict(list)

    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped[key].append(row)

    aggregated_rows = []
    for group_values, group_rows in grouped.items():
        result = {key: value for key, value in zip(group_keys, group_values)}
        result["runs"] = len(group_rows)

        for metric in metric_keys:
            metric_values = [float(row[metric]) for row in group_rows]
            result[f"{metric}_mean"] = mean(metric_values)
            result[f"{metric}_std"] = std_dev(metric_values)

        aggregated_rows.append(result)

    return aggregated_rows