"""Load plain text (.txt) or CSV files.

The Kaggle "AWS Case Studies and Blogs" dataset ships as CSV, so this loader
also knows how to turn a CSV row into a single text blob (all columns joined,
labelled by column name) which then flows through the same chunker as every
other document type.
"""

import csv


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_csv_rows(file_path: str) -> list[dict]:
    """Return each CSV row as a dict: {"source": <file>:<row idx>, "text": <joined columns>}."""
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Join every column as "column: value" so no information is lost
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            rows.append({"source": f"{file_path}#row{i}", "text": text})
    return rows
