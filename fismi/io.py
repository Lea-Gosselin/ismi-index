from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_config(country):
    """Load configuration .yaml for a country."""

    country = country.upper()
    path = ROOT / "config" / f"{country}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config["country"] != country:
        raise ValueError(
            f"Country mismatch in {path}"
        )

    return config


def save_input(country, prices, weights):
    """Save standardized input datasets"""

    folder = ROOT / "input"
    folder.mkdir(parents=True, exist_ok=True)

    prices.to_parquet(folder / f"{country.upper()}IPC.parquet")
    weights.to_parquet(folder / f"{country.upper()}Weights.parquet")

    return folder

def save_ismi(country, results):
    folder = ROOT / "results"
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{country.upper()}Ismi.parquet"
    results.to_parquet(path)

    return path

def load_input(country):
    folder = ROOT / "input"

    prices = pd.read_parquet(folder / f"{country.upper()}IPC.parquet")
    weights = pd.read_parquet(folder / f"{country.upper()}Weights.parquet")

    return prices, weights


import json
import numpy as np


def save_persistence(country, persistence):
    """Save persistence results and estimated parameters"""

    folder = ROOT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    country = country.upper()

    # Filtered time series
    path = folder / f"{country}Persistence.parquet"
    persistence["results"].to_parquet(path)

    # Parameters and diagnostics
    metadata = {
        "model": persistence["model"],
        "parameters": np.asarray(
            persistence["parameters"]
        ).tolist(),
        "loglikelihood": float(
            persistence["loglikelihood"]
        ),
    }

    with open(
        folder / f"{country}Persistence.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=4)

    return path
