from fismi.io import load_input, save_persistence, load_config
from fismi.models import estimate_persistence


for country in ["EA", "US"]:

    print(f"\n--- {country} ---")

    config = load_config(country)
    prices, weights = load_input(country)

    persistence = estimate_persistence(
        prices=prices,
        config=config,
    )

    path = save_persistence(country, persistence)

    print(persistence["results"].tail())
    print("Estimated parameters:", persistence["parameters"])
    print("Log-likelihood:", persistence["loglikelihood"])
    print("Saved:", path)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# US estimation
final = persistence["filtered"]

# Dates aligned with Kalman outputs
dates = prices["HEADLINE"].dropna().index[
    config["persistence"]["inflation_lag"] + 1:
]

# Results
df = pd.DataFrame({
    "Observed": final["Y"],
    "Fitted": final["FittedY"],
    "Rho": final["Rho"]
}, index=dates)

fig, axes = plt.subplots(
    2, 1, figsize=(16, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# 1. Observed vs fitted inflation
axes[0].plot(df.index, df["Observed"] * 100,
             color="darkgrey", label="Observed inflation")
axes[0].plot(df.index, df["Fitted"] * 100,
             color="darkblue", label="Fitted inflation")

axes[0].set_ylabel("Inflation (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Time-varying persistence
axes[1].plot(df.index, df["Rho"],
             color="purple", label=r"$\rho_t$")

axes[1].axhline(1, color="red", linestyle="--", linewidth=1)
axes[1].set_ylabel(r"Persistence ($\rho_t$)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()