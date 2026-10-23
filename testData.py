import numpy as np
import pandas as pd

from fismi.io import load_config, save_input
from fismi.data import collect_data


# 1. Load country configuration
for country in ["EA", "US"]:
    config = load_config(country)

    print("Country:", config["country"])
    print("Disaggregation level:", config["data"]["level"])

    # 2. Collect and prepare data
    prices, weights = collect_data(country)

    print("\n--- DATA COLLECTION ---")
    print("Prices:", prices.shape)
    print("Weights:", weights.shape)
    print("Period:", prices.index.min(), "to", prices.index.max())


    # 7. Save standardized inputs
    folder = save_input(country, prices, weights)

    print("\nFiles saved to:", folder)
    print(f"{country.upper()} collection test passed!")


    print("Data successfully processed!")
    print(f"Prices (incl. aggregate price index): {prices.shape}")
    print(f"Weights: {weights.shape}")
    print(f"Weight sums: {weights.sum(axis=1).describe()}")