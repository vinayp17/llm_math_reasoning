#!/usr/bin/env python

from dataclasses import dataclass
import pandas as pd
from typing import List


@dataclass
class Model:
    category: str
    model_name: str
    version: str
    notes: str


def load_models_from_csv(path: str) -> List[Model]:
    """
    Loads LLM models from a CSV and returns a list of Model instances.

    Args:
        path (str): Path to the CSV file.

    Returns:
        List[Model]: List of Model objects.
    """
    df = pd.read_csv(path)

    required_columns = {"Category", "Model Name", "Version", "Notes"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    return [
        Model(
            category=row["Category"],
            model_name=row["Model Name"],
            version=row["Version"],
            notes=row["Notes"]
        )
        for _, row in df.iterrows()
    ]
