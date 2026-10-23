from pathlib import Path
import pandas as pd

from fismi.data import prepare_data

root = Path.cwd()

prices = pd.read_parquet(
    root / "input" / "EArawItemsLvl3.parquet"
)

weights = pd.read_parquet(
    root / "input" / "EArawWeightsLvl3.parquet"
)

prices = prices.rename(columns={"INX": "HEADLINE"})

prices, weights = prepare_data(
    prices=prices,
    weights=weights
)

output = root / "Input" / "EA"
output.mkdir(parents=True, exist_ok=True)

prices.to_parquet(output / "IPC.parquet")
weights.to_parquet(output / "Weights.parquet")

print("Data successfully processed!")
print(f"Prices: {prices.shape}")
print(f"Weights: {weights.shape}")
print(f"Weight sums: {weights.sum(axis=1).describe()}")