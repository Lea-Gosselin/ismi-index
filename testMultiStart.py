from fismi.io import load_input, load_config
import numpy as np
import pandas as pd
from fismi.models import kalAlgoSimple
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from fismi.maths import rmse

# Select country
country = "US"
cMu = True
cRho = False

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
        rng.uniform(-0.5, 0.5),                              # mu
        # rng.uniform(-0.99, 0.99),                            # rho_bar
        rng.uniform(-0.9, 0.99),                             # F
        rng.uniform(1e-6, 1),                           # D
        rng.uniform(1e-6, 1)                            # B
    ])

    try:
        res = kalAlgoSimple.optNegLogLike(
            x0,
            inflation,
            cMu=cMu,
            cRho=cRho
        )

        nll = kalAlgoSimple.kalman_filter_inflation(
            res.x,
            inflation,
            cMu=cMu,
            cRho=cRho,
            return_loglik=True
        )

        runs.append({
            "run": i,
            "success": res.success,
            "objective": res.fun,
            "loglik": -nll,
            "mu": res.x[0],
            # "rho_bar": res.x[1],
            "F": res.x[1],
            "D": res.x[1],
            "B": res.x[3],
            "lambda": (res.x[2] / res.x[3]) ** 2
        })

    except (ValueError, FloatingPointError, OverflowError):
        continue

dfMultiStart = pd.DataFrame(runs)

dfMultiStart = dfMultiStart.sort_values(
    "objective"
).reset_index(drop=True)

print(dfMultiStart.head(20).to_string(index=False))

# Plot the distribution of each parameter
# params = ["mu", "rho_bar", "F", "D", "B", "lambda"]
params = ["mu", "F", "D", "B", "lambda"]

# Best solution according to the objective function
successful = dfMultiStart.loc[(dfMultiStart["success"]==True)]
best = successful.loc[successful["objective"].idxmin()]

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 8),
    constrained_layout=True
)

for ax, c in zip(axes.flat, params):

    values = successful[c].dropna()

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


# Plot some of the successful results...
# x0 = successful.iloc[98,-6:-1].values
x0 = dfMultiStart.iloc[0, -6:-1].values
result = kalAlgoSimple.kalman_filter_inflation(
    x0,
    inflation,
    cMu=cMu,
    cRho=cRho,
    return_loglik=False
)

df = pd.DataFrame(result)

# 1. Observed vs fitted inflation
rMse = rmse(df["Y"], df["FittedY"])
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(df.index, df["Y"] * 100,
        color="darkgrey", label="Observed inflation")
ax.plot(df.index, df["FittedY"] * 100,
        color="darkblue", label=f"Fitted inflation ({rMse*100:.2f} pp)")

ax.set_ylabel("Inflation (%)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# 2. Time-varying persistence and confidence interval
fig, ax = plt.subplots(figsize=(16, 5))

z = norm.ppf(0.95)  # 90% bilateral confidence interval

RhoFiltered = df["Rho"]
std = np.sqrt(df["forecastVarState"])

df["RhoLower90"] = RhoFiltered - z * std
df["RhoUpper90"] = RhoFiltered + z * std

ax.fill_between(
    df.index,
    df["RhoLower90"],
    df["RhoUpper90"],
    color="grey",
    alpha=0.20,
    linewidth=0,
    label="90% confidence interval"
)

ax.plot(df.index, RhoFiltered,
        color="orange",
        label=r"Filtered $\rho_t$")

ax.axhline(x0[1], color="red", linestyle="--", linewidth=1)

ax.set_ylabel(r"Persistence ($\rho_t$)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Constante... Time-varying...
cFiltered = x0[0]*(1-RhoFiltered)

fig, ax = plt.subplots(
    figsize=(16, 8)
)
ax.plot(df.index, df["Y"] * 100,
            color="darkgrey", label="Monthly inflation")
ax.plot(df.index, cFiltered * 100,
            color="orange", label=r"Filtered time-varying constant $c_{t|t-1} = \mu (1-\rho_{t|t-1})$")

ax.legend()
ax.grid(alpha=0.3)


# Etude des erreurs...
errors = df["Err"].to_numpy()

print("Mean innovation:", np.mean(errors))
print("Innovation std:", np.std(errors))
print("RMSE forecast:", rmse(df["Y"], df["ForecastY"]))

fig, ax = plt.subplots(figsize=(16, 4))

ax.plot(df.index, errors * 100, color="darkblue")
ax.axhline(0, color="black", linestyle="--")
ax.axhline(np.mean(errors) * 100,
           color="red", linestyle=":",
           label="Mean innovation")

ax.set_ylabel("Forecast error (pp)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Standardized one-step-ahead Kalman innovations
z = (
    np.asarray(df["Err"])
    / np.sqrt(np.asarray(df["forecastVarY"]))
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
