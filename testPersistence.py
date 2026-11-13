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

# ["US", "EA", "UK"]

for country in ["US"]:

    print(f"\n--- {country} ---")

    config = load_config(country)
    prices, weights = load_input(country)
    startDate = config["persistence"]["startDate"]
    pi = prices.loc[prices.index >= startDate, "HEADLINE"].diff().dropna()
    
    ## 1) Estimate an AR(p) on the full sample
    p = 1

    # AR(1) à coefficient constant
    Y = pi[p:].to_numpy()
    X = sm.add_constant(pi[:-p].to_numpy())
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

    # Start estimation...
    persistence = estimate_persistence(
        prices=prices,
        config=config,
    )

    path = save_persistence(country, persistence)

    print("----- Log-likelihood -----")    
    print(f"AR(1) full data, constant persistence : {model.llf}")
    print(f"KF and MLE, varying persistence : {-persistence['optimization']['fun']}")
    
    print("----- Initial parameters -----")
    initParams = config['persistence']['initial_params']
    print(f"Mu bar = {initParams[0]}")
    print(f"Rho bar = {initParams[1]}")
    print(f"Alpha = {initParams[2]}")
    print(f"Std. inflation innovations = {initParams[3]}")
    print(f"Std. state innovations = {initParams[4]}")
    
    print("----- Estimated parameters -----")
    estParams = persistence["parameters"]
    print(f"Mu bar = {estParams[0]}")
    print(f"Rho bar = {estParams[1]}")
    print(f"Alpha = {estParams[2]}")
    print(f"Std. inflation innovations = {estParams[3]}")
    print(f"Std. state innovations = {estParams[4]}")

    print("Saved:", path)

# US estimation
filtered = persistence["filtered"]
smoothed = persistence["smoothed"]

# Dates aligned with Kalman outputs
dates = pi.index[
config["persistence"]["inflation_lag"] + 1:
]

# Results
df = persistence["results"]


# 1. Observed vs fitted inflation
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(df.index, df["Inflation"] * 100,
        color="darkgrey", label="Observed inflation")
ax.plot(df.index, df["FittedInflation"] * 100,
        color="darkblue", label="Fitted inflation")

ax.set_ylabel("Inflation (%)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# 2. Time-varying persistence and confidence interval
fig, ax = plt.subplots(figsize=(16, 5))

z = norm.ppf(0.95)  # 90% bilateral confidence interval

RhoSmoothed = df["RhoSmoothed"]
RhoFiltered = df["RhoFiltered"]
std = np.sqrt(smoothed["smoothedVarState"])

df["RhoLower90"] = RhoSmoothed - z * std
df["RhoUpper90"] = RhoSmoothed + z * std

ax.fill_between(
    df.index,
    df["RhoLower90"],
    df["RhoUpper90"],
    color="grey",
    alpha=0.20,
    linewidth=0,
    label="90% confidence interval"
)

ax.plot(df.index, RhoSmoothed,
        color="darkblue", linewidth=1.5,
        label=r"Smoothed $\rho_t$")

ax.plot(df.index, RhoFiltered,
        color="orange",
        label=r"Filtered $\rho_t$")

ax.axhline(0.48, color="red", linestyle="--", linewidth=1)

ax.set_ylabel(r"Persistence ($\rho_t$)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Constante... Time-varying...
cFiltered = estParams[0]*(1-RhoFiltered)
cSmoothed = estParams[0]*(1-RhoSmoothed)

fig, ax = plt.subplots(
    figsize=(16, 8)
)
ax.plot(df.index, df["Inflation"] * 100,
            color="darkgrey", label="Monthly inflation")
ax.plot(df.index, cFiltered * 100,
            color="orange", label=r"Filtered time-varying constant $c_{t|t-1} = \mu (1-\rho_{t|t-1})$")
ax.plot(df.index, cSmoothed * 100,
            color="darkblue", label=r"Smoothed time-varying constant $c_{t|t} = \mu (1-\rho_{t|t})$")

ax.legend()
ax.grid(alpha=0.3)

# Standardized one-step-ahead Kalman innovations
z = (
    np.asarray(filtered["Err"])
    / np.sqrt(np.asarray(filtered["forecastVarY"]))
)

# ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(z, lags=36, ax=axes[0])
axes[0].set_title("Standardized KF Innovations (ACF)")

plot_pacf(z, lags=36, ax=axes[1], method="ywm")
axes[1].set_title("Standardized KF Innovations (PACF)")

axes[0].grid(alpha=0.3)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
