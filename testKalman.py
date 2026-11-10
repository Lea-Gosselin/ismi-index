from fismi.io import load_input, save_persistence, load_config
from fismi.models import estimate_persistence
from fismi.utils import olsToLatex
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import statsmodels.api as sm 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Country
country = "US"
print(f"\n--- {country} ---")

config = load_config(country)
prices, weights = load_input(country)

# Load results
ROOT = Path(__file__).resolve().parents[0]
path = ROOT / "results" / f"{country.upper()}Persistence.parquet"
persistence = pd.read_parquet(path)
parameters = pd.read_json(ROOT / "results" / f"{country.upper()}Persistence.json")
alpha = parameters["parameters"][2]
sigU = parameters["parameters"][4]
sigEpsilon = parameters["parameters"][3]
mu = parameters["parameters"][0]
rhoBar = parameters["parameters"][1]

# Dates aligned with Kalman outputs
dates = prices["HEADLINE"].dropna().index[
config["persistence"]["inflation_lag"] + 1:
]

# Test 1 : Are innovations exactly equal to forecast errors ? 
np.testing.assert_allclose(
    persistence["Inflation"] - persistence["ForecastInflation"],
    persistence["Residuals"]
)

## Step : Inflation prediction and error
fig, ax = plt.subplots(figsize=(16, 5))

# Filtered inflation (left axis)
l1, = ax.plot(
    persistence.index,
    persistence["FittedInflation"] * 100,
    color="darkblue",
    linewidth=1.5,
    label=r"Filtered inflation ($\hat \pi_{t|t}$)"
)

# Forecast errors (right axis)
axBis = ax.twinx()
z = persistence["Residuals"] / np.sqrt(persistence["forecastVarY"])
l2, = axBis.plot(
    persistence.index,
    z,
    color="darkgrey",
    linewidth=1,
    alpha=0.8,
    label=r"Standardised forecast errors ($\frac{\tilde y_t - \tilde y_{t|t-1}}{\sqrt{P_{t|t-1}^{\tilde y}}}$)"
)

# Zero reference line
axBis.axhline(
    0, color="darkgrey",
    linewidth=0.8, linestyle="--"
)

# Axis labels
ax.set_ylabel("Filtered inflation (%)", color="darkblue")
axBis.set_ylabel("Forecast errors (pp)", color="darkgrey")

ax.tick_params(axis="y", labelcolor="darkblue")
axBis.tick_params(axis="y", labelcolor="darkgrey")

# Combined legend
ax.legend(
    handles=[l1, l2],
    loc="upper left",
    frameon=False
)

ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()
print("le modèle rencontre des difficultés à anticiper les mouvements brusques de l'inflation. Ce n'est pas caractéristique d'une mauvaise spécification de la persistance.")

## Step : Effect of measurement errors standard deviation
fig, ax = plt.subplots(figsize=(16, 5))

# Left axis
l1, = ax.plot(
    persistence.index,
    persistence["Residuals"],
    color="lightgrey",
    linewidth=1.5,
    label=r"Forecast errors ($\tilde y_t - \tilde y_{t|t-1}$)"
)

# Right axis
axBis = ax.twinx()
l2, = axBis.plot(
    persistence.index,
    np.sqrt(persistence["forecastVarY"]),
    color="orange",
    linewidth=1,
    alpha=0.8,
    label=r"Forecast errors std. dev. ($\sqrt{P_{t|t-1}^{\tilde y}}$)"
)
# Zero reference line
axBis.axhline(
    sigEpsilon, color="orange",
    linewidth=0.8, linestyle="--"
)

# Axis labels
ax.set_ylabel("Forecast errors")
axBis.set_ylabel("Measurement error std. dev.", color="orange")

ax.tick_params(axis="y", color="lightgrey")
axBis.tick_params(axis="y", labelcolor="orange")

# Combined legend
ax.legend(
    handles=[l1, l2],
    loc="upper left",
    frameon=False
)

ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()
print(r"La variance des erreurs de prévision est principalement déterminée par le bruit de l'équation d'observation : $\pi_{t-1}^2 P_{t|t-1}^{\xi} + \sigma_{\epsilon}^2$. Les fluctuations de la persistance latente contribuent relativement peu à l'incertitude prédictive lorsque l'inflation retardée est proche de zéro.")


## Step : State inference
## How much the persistence coefficient reflects forecast errors ?
T = persistence["Inflation"].shape[0]
KtErr = persistence["KalmanGain"]*persistence["Residuals"]

# Test 2 : Are updated states exactly equal to the forecasts + KtErr ? 
np.testing.assert_allclose(
    persistence["filteredState"] - persistence["forecastState"],
    KtErr
)
Error = persistence["filteredState"] - persistence["forecastState"] - KtErr
print(f"Test 2 : {Error.max()}")

# Plot
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(persistence["forecastState"], color="steelblue",
        linewidth=1.5, label=r"$\hat{\xi}_{t|t-1}$ (Forecast)")

ax.plot(persistence["filteredState"], color="black",
        linewidth=1.5, label=r"$\hat{\xi}_{t|t}$ (Filtered)")

ax.plot(KtErr, color="darkorange",
        linewidth=1.2, alpha=0.8,
        label=r"$K_t e_t$ (Kalman correction)")

ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

ax.set_title("State inference and Kalman correction")
ax.set_xlabel("Time")
ax.set_ylabel(r"Latent state $\xi_t$")
ax.legend(frameon=False)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()


# Plot différence entre filtered et forecast
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot((persistence["filteredState"] - persistence["forecastState"]), color="steelblue",
        linewidth=1.5, label=r"$\hat{\xi}_{t|t} - \hat{\xi}_{t|t-1}$")

ax.plot(KtErr, color="lightgrey", linestyle="--",
        linewidth=1.5, label=r"$K_t e_t^{\tilde y}")

ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

ax.set_title("Difference between filtered and forecast states")
ax.set_xlabel("Time")
ax.set_ylabel(r"Kalman gain times forecast errors $K_te_t^{\tilde y}$")
ax.legend(frameon=False)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()


## Test des erreurs standardisées (au carré)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(z, color="steelblue", linewidth=1.2)
axes[0, 0].axhline(0, color="grey", linestyle="--")
axes[0, 0].set_title("Standardized innovations")

plot_acf(z**2, lags=24, ax=axes[0, 1])
axes[0, 1].set_title("ACF of squared innovations")

plot_pacf(z**2, lags=24, ax=axes[1, 0])
axes[1, 0].set_title("PACF of squared innovations")

axes[1, 1].hist(
    z, bins=35, density=True,
    color="steelblue", alpha=0.6
)

x = np.linspace(-4, 4, 300)
axes[1, 1].plot(
    x, norm.pdf(x), color="darkgrey",
    linestyle="--", label="N(0,1)"
)
axes[1, 1].legend()
axes[1, 1].set_title("Innovation distribution")

for ax in axes.flat:
    ax.grid(alpha=0.5, linestyle="--")
plt.tight_layout()
plt.show()
print("Ces graphiques suggèrent une dépendence de court-terme entre les surprises inflationnistes. La variance conditionnelle n'est pas correctement capturée.")



## Step 3 : Décomposition du gain de Kalman
lagged_pi = persistence["Inflation"].shift(1)

## Les révisions de l'état interviennent-elles majoritairement quand l'inflation retardée est proche de 0 ?
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(persistence["KalmanGain"][1:], color="darkgrey", linewidth=1.2)
axes[0, 0].set_title("Kalman Gain ($K_t$)")

axes[0, 1].plot(persistence["forecastVarY"][1:], color="orange", linewidth=1.2)
axes[0, 1].axhline(sigEpsilon**2, color="darkgrey", linestyle="--")
axes[0, 1].set_title(r"Forecast errors variance ($P_{t|t-1}^{\tilde y}$)")

axes[1, 0].plot(persistence["forecastVarState"][1:], color="darkblue", linewidth=1.2)
axes[1, 0].axhline(sigU**2, color="darkgrey", linestyle="--")
axes[1, 0].set_title(r"Forecast state variance ($P_{t|t-1}^{\xi}$)")

axes[1, 1].plot(lagged_pi, color="steelblue", linewidth=1.2)
axes[1, 1].axhline(mu, color="darkgrey", linestyle="--")
axes[1, 1].set_title(r"Lagged inflation ($\pi_{t-1}$)")

for ax in axes.flat:
    ax.grid(alpha=0.5, linestyle="--")
plt.tight_layout()
plt.show()
