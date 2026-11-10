from fismi.io import load_input, save_persistence, load_config
from fismi.models import estimate_persistence
from fismi.utils import olsToLatex
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

for country in ["US"]:

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

# US estimation
filtered = persistence["filtered"]
smoothed = persistence["smoothed"]

# Dates aligned with Kalman outputs
dates = prices["HEADLINE"].dropna().index[
config["persistence"]["inflation_lag"] + 1:
]

# Results
df = persistence["results"]

## 1) Estimate an AR(p) on the full sample
p = 1

# AR(1) à coefficient constant
Y = df["Inflation"][p:].to_numpy()
X = sm.add_constant(df["Inflation"][:-p].to_numpy())
model = sm.OLS(Y, X).fit()
print(model.summary())
resid = model.resid
olsToLatex(model,save=True)

# ACF and PACF
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

plot_acf(Y, lags=36, ax=axes[0, 0])
axes[0, 0].set_title("ACF")

plot_pacf(Y, lags=36, ax=axes[0, 1], method="ywm")
axes[0, 1].set_title("PACF") 

plot_acf(resid, lags=36, ax=axes[1, 0])
axes[1, 0].set_title("Residuals (ACF)")

plot_pacf(resid, lags=36, ax=axes[1, 1], method="ywm")
axes[1, 1].set_title("Residuals (PACF)")

for ax in axes.flat:
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# 2) Observed vs fitted inflation
fig, axes = plt.subplots(
    2, 1,
    figsize=(16, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)
axes[0].plot(df.index, df["Inflation"] * 100,
            color="darkgrey", label="Observed inflation")
axes[0].plot(df.index, df["FittedInflation"] * 100,
            color="darkblue", label="Fitted inflation")

axes[0].set_ylabel("Inflation (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Time-varying persistence and IC

## Smoothed
# 90% confidence interval
z = norm.ppf(0.95)

RhoSmoothed = df["RhoSmoothed"]
std = np.sqrt(smoothed["smoothedVarState"])
df["RhoLower90"] = RhoSmoothed - z * std
df["RhoUpper90"] = RhoSmoothed + z * std

axes[1].fill_between(
    df.index,
    df["RhoLower90"],
    df["RhoUpper90"],
    color="grey",
    alpha=0.20,
    linewidth=0,
    label="90% confidence interval"
)

axes[1].plot(
    df.index, df["RhoSmoothed"],
    color="black",
    linewidth=1.5,
    label=r"Smoothed $\rho_t$"
)

## Filtered
axes[1].plot(df.index, df["RhoFiltered"],
            color="lightblue", label=rf"Filtered $\rho_t$")

axes[1].axhline(0.48, color="red", linestyle="--", linewidth=1)
axes[1].set_ylabel(r"Persistence ($\rho_t$)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
