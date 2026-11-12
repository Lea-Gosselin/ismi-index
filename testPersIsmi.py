from fismi.io import load_input, save_persistence, load_config
from fismi.models import estimate_persistence
from fismi.utils import olsToLatex
from fismi.models import kalAlgoSimple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ["US", "EA", "UK"]
country = "US"
config = load_config(country)
params = config["persistence"]
sDate = params["startDate"]
model_name = params["model"].lower()
lag = params["inflation_lag"]

# 1. ISMI
ismi = pd.read_parquet("/Users/lea_gosselin/GitRepo/ismi-index/results/USIsmi.parquet")
pi = ismi["ISMI"].to_numpy()

x0 = np.asarray(params["initial_params"], dtype=float)
estimation = kalAlgoSimple.optNegLogLike(
    x0,
    pi,
    cMu=params["estimate_mu"],
    cRho=params["estimate_rho_bar"],
)

violations = np.asarray(
    kalAlgoSimple.persistence_constraint(estimation.x, pi, params["estimate_mu"], params["estimate_rho_bar"])
)

if not estimation.success:
    raise RuntimeError(
        f"Kalman optimization failed: {estimation.message}",
        f"Status: {estimation.status}"
        f"Objective function: {estimation.fun}"
        f"Parameters: {estimation.x}"
        f"Minimum constraint: {max(0, -np.nanmin(violations))}"
        f"Maximum violation: {np.nanmin(violations)}"
    )
    print("Status:", estimation.status)
    print("Message:", estimation.message)
    print("Objective:", estimation.fun)
    print("Parameters:", estimation.x)

# 4. Run Kalman filter with estimated parameters
filtered = kalAlgoSimple.kalman_filter_inflation(
    estimation.x,
    pi,
    cMu=params["estimate_mu"],
    cRho=params["estimate_rho_bar"]
)

# 5. Convert filtered series to DataFrame
dates = ismi.index[1:]

results = pd.DataFrame(
    {
        "Inflation": filtered["Y"],
        "FittedInflation": filtered["FittedY"],
        "ForecastInflation": filtered["ForecastY"],
        "RhoFiltered": filtered["Rho"],
        "Residuals": filtered["Err"],
        "forecastVarState": filtered["forecastVarState"],
        "filteredVarState": filtered["filteredVarState"],
        "forecastVarY": filtered["forecastVarY"],
        "forecastState":filtered["forecastState"],
        "filteredState": filtered["filteredState"],
        "KalmanGain": filtered["KalmanGain"],
        "LogLikelihood": filtered["LogLikelihood"]
    },
    index=dates,
)

# 6. Smooth estimates
smoothed = kalAlgoSimple.kalman_smoother(
    filtered,
    estimation.x,
    cMu=params["estimate_mu"],
    cRho=params["estimate_rho_bar"]
    )

rhoSmoothed = pd.DataFrame(
    {
        "RhoSmoothed": smoothed["smoothedRho"],
        "VarStateSmoothed": smoothed["smoothedVarState"]
    },
    index=dates,
)

# 7. Merge with results
results = pd.merge(
    results,
    rhoSmoothed,
    right_index = True,
    left_index = True
)


# 2) Observed vs fitted inflation
dates = ismi.index[1:]
fig, axes = plt.subplots(
    2, 1,
    figsize=(16, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)
axes[0].plot(dates, results["Inflation"] * 100,
            color="darkgrey", label="Observed ISMI")
axes[0].plot(dates, results["FittedInflation"] * 100,
            color="darkblue", label="Fitted ISMI")

axes[0].set_ylabel("ISMI (pp)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Time-varying persistence and IC

## Smoothed
# 90% confidence interval
z = norm.ppf(0.95)

RhoSmoothed = results["RhoSmoothed"]
std = np.sqrt(smoothed["smoothedVarState"])
results["RhoLower90"] = RhoSmoothed - z * std
results["RhoUpper90"] = RhoSmoothed + z * std

axes[1].fill_between(
    dates,
    results["RhoLower90"],
    results["RhoUpper90"],
    color="grey",
    alpha=0.20,
    linewidth=0,
    label="90% confidence interval"
)

axes[1].plot(
    dates, results["RhoSmoothed"],
    color="black",
    linewidth=1.5,
    label=r"Smoothed $\rho_t$"
)

## Filtered
axes[1].plot(dates, results["RhoFiltered"],
            color="lightblue", label=rf"Filtered $\rho_t$")

axes[1].axhline(0.48, color="red", linestyle="--", linewidth=1)
axes[1].set_ylabel(r"Persistence ($\rho_t$)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


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
