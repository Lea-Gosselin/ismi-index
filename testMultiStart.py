from fismi.io import load_input, load_config
import numpy as np
import pandas as pd
from fismi.models import kalAlgoSimple
import matplotlib.pyplot as plt

# Select country
country = "US"

# Start estimates
print(f"\n--- {country} ---")
config = load_config(country)
startDate = config["persistence"]["startDate"]
prices, weights = load_input(country)
    
rng = np.random.default_rng(42)

inflation = np.log(
    prices["HEADLINE"].dropna()
).diff(
    config["persistence"]["inflation_lag"]
).dropna()

inflation = inflation[
    inflation.index >= startDate
].to_numpy()

runs = []

for i in range(500):
    x0 = np.array([
        rng.uniform(-0.01, 0.01),                   # mu
        rng.uniform(0, 1),                          # rho_bar
        rng.uniform(-1, 1),                         # F
        rng.uniform(10**-6, 1),                     # D
        rng.uniform(10**-6, 1)                      # B
    ])

    try:
        res = kalAlgoSimple.optNegLogLike(
            x0,
            inflation,
            cMu=True,
            cRho=True
        )

        nll = kalAlgoSimple.kalman_filter_inflation(
            res.x,
            inflation,
            cMu=True,
            cRho=True,
            return_loglik=True
        )

        runs.append({
            "run": i,
            "success": res.success,
            "objective": res.fun,
            "loglik": -nll,
            "mu": res.x[0],
            "rho_bar": res.x[1],
            "F": res.x[2],
            "D": res.x[3],
            "B": res.x[4],
            "lambda": (res.x[4] / res.x[3]) ** 2
        })

    except (ValueError, FloatingPointError, OverflowError):
        continue

dfMultiStart = pd.DataFrame(runs)

dfMultiStart = dfMultiStart.sort_values(
    "objective"
).reset_index(drop=True)

print(dfMultiStart.head(20).to_string(index=False))

# Plot the distribution of each parameter
params = ["mu", "rho_bar", "F", "D", "B", "lambda"]

# Best solution according to the objective function
successful = dfMultiStart.loc[(dfMultiStart["success"]==True)]
best = successful.loc[successful["objective"].idxmin()]

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 8),
    constrained_layout=True
)

for ax, c in zip(axes.flat, params):

    values = dfMultiStart[c].dropna()

    ax.hist(
        values,
        bins=30,
        color="skyblue",
        edgecolor="black",
        alpha=0.7
    )

    ax.axvline(
        best[c],
        color="darkred",
        linestyle="--",
        linewidth=1.5,
        label="Best optimum"
    )

    ax.set_title(c.upper(), fontsize=13)
    ax.set_xlabel("Estimated value")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.show()
