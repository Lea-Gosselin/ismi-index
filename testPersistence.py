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
    print(f"Std. inflation innovations = {initParams[2]}")
    print(f"Std. state innovations = {initParams[3]}")
    
    print("----- Estimated parameters -----")
    estParams = persistence["parameters"]
    print(f"Mu bar = {estParams[0]}")
    print(f"Rho bar = {estParams[1]}")
    print(f"Std. inflation innovations = {estParams[2]}")
    print(f"Std. state innovations = {estParams[3]}")

    print("Saved:", path)


# ============================================================
# 0. PREPARE DATA
# ============================================================

df = persistence["results"].copy()
filtered = persistence["filtered"]
smoothed = persistence["smoothed"]

# The estimation must use monthly inflation
if config["persistence"]["inflation_lag"] != 1:
    raise ValueError(
        "Set inflation_lag=1 and re-estimate the model "
        "to display fitted MoM inflation."
    )

# Actual inflation (log differences, in %)
headline = prices["HEADLINE"].sort_index()
mom = 100 * np.log(headline).diff(1)
yoy = 100 * np.log(headline).diff(12)

# Align with Kalman estimation dates
dates = df.index
mom = mom.reindex(dates)
yoy = yoy.reindex(dates)

# Kalman estimates
momFitted = 100 * df["FittedInflation"]
rhoFiltered = df["RhoFiltered"]
rhoSmoothed = df["RhoSmoothed"]

# 90% interval for smoothed persistence
z90 = norm.ppf(0.95) # Oui, vraie valeur 1.64
stdRho = np.sqrt(
    np.maximum(df["VarStateSmoothed"], 0)
)

rhoLower = rhoSmoothed - z90 * stdRho
rhoUpper = rhoSmoothed + z90 * stdRho

# Diagnostics
Kt = df["KalmanGain"]
Err = df["Residuals"]
forecastStd = np.sqrt(df["forecastVarY"])

# Standardized one-step-ahead innovations
innovationErr = Err / forecastStd

# Colors
BLUE = "#173F81"
ORANGE = "#EC9227"
GREY = "#777777"
LIGHT_GREY = "#B5B5B5"

# ============================================================
# 1. INFLATION AND TIME-VARYING PERSISTENCE
# ============================================================

fig, axes = plt.subplots(
    2, 1, figsize=(16, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1]},
    constrained_layout=True
)

# Panel 1: Inflation
ax = axes[0]
ax2 = ax.twinx()

l1, = ax.plot(
    dates, mom,
    color=LIGHT_GREY, lw=1.2, alpha=0.8,
    label="Observed"
)

l2, = ax.plot(
    dates, momFitted,
    color=BLUE, lw=1.5,
    label=r"Forecast $\pi_{t|t-1}$ (Kalman)"
)

l3, = ax2.plot(
    dates, yoy,
    color="black", lw=1.2, ls="--", alpha=0.65,
    label="Observed g.a"
)

ax.set_ylabel("Monthly inflation (%)")
ax2.set_ylabel("Annual inflation (%)")
ax.set_title("Inflation: observed and fitted")
ax.legend(
    handles=[l1, l2, l3],
    loc="upper left", ncol=3
)
ax.grid(alpha=0.25)

# Panel 2: Persistence
ax = axes[1]

ax.fill_between(
    dates, rhoLower, rhoUpper,
    color=BLUE, alpha=0.15,
    linewidth=0,
    label="90% smoothed interval"
)

ax.plot(
    dates, rhoFiltered,
    color=ORANGE, lw=1.2, alpha=0.85,
    label=r"Filtered $\rho_{t|t}$"
)

ax.plot(
    dates, rhoSmoothed,
    color=BLUE, lw=1.8,
    label=r"Smoothed $\rho_{t|T}$"
)

ax.axhline(
    0, color="black", lw=0.8, alpha=0.5
)

ax.set_ylabel(r"Persistence ($\rho_t$)")
ax.set_title("Time-varying inflation persistence")
ax.legend(loc="upper left", ncol=3)
ax.grid(alpha=0.25)

plt.show()


# ============================================================
# 2. KALMAN GAIN AND FORECAST ERRORS
# ============================================================

fig, axes = plt.subplots(
    2, 1, figsize=(16, 8),
    sharex=True,
    constrained_layout=True
)

# Panel 1: Kalman gain
axes[0].plot(
    dates, Kt,
    color=BLUE, lw=1.3,
    label=r"Kalman gain $K_t$"
)

axes[0].axhline(
    0, color="black", lw=0.8, alpha=0.5
)

axes[0].set_title("Kalman gain")
axes[0].set_ylabel(r"$K_t$")
axes[0].legend(loc="upper left")
axes[0].grid(alpha=0.25)

# Panel 2: Forecast errors
axes[1].plot(
    dates, Err * 100,
    color=ORANGE, lw=1.1,
    label=r"Forecast error $e_t$"
)

axes[1].axhline(
    0, color="black", lw=0.8, alpha=0.5
)

axes[1].set_title("One-step-ahead forecast errors")
axes[1].set_ylabel("Error (percentage points)")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.25)

plt.show()


# ============================================================
# 3. RAW AND STANDARDIZED KALMAN INNOVATIONS
# ============================================================

fig, axes = plt.subplots(
    2, 1, figsize=(16, 8),
    sharex=True,
    constrained_layout=True
)

# Panel 1: Raw innovations
axes[0].plot(
    dates, Err * 100,
    color=GREY, lw=1.1,
    label=r"Raw innovation $e_t$"
)

axes[0].axhline(
    0, color="black", lw=0.8, alpha=0.5
)

axes[0].set_title("Raw Kalman innovations")
axes[0].set_ylabel("Innovation (percentage points)")
axes[0].legend(loc="upper left")
axes[0].grid(alpha=0.25)

# Panel 2: Standardized innovations
axes[1].plot(
    dates, innovationErr,
    color=BLUE, lw=1.1,
    label=r"Standardized innovation $z_t$"
)

axes[1].axhline(
    0, color="black", lw=0.8, alpha=0.5
)

for bound in [-1.96, 1.96]:
    axes[1].axhline(
        bound,
        color="red", ls="--", lw=1,
        alpha=0.7
    )

axes[1].set_title("Standardized Kalman innovations")
axes[1].set_ylabel(r"$e^y_t / \sqrt{P^y_{t|t-1}}$")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.25)

plt.show()
