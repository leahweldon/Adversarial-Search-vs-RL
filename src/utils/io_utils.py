from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def save_dict_rows_to_csv(rows: list[dict], output_path: str | Path) -> None:
    """Save list of dictionaries to a CSV file"""
    if not rows:
        raise ValueError("Cannot save an empty list of rows.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)