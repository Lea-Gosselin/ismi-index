import requests
from pathlib import Path
from dotenv import load_dotenv
from fismi.io import load_input, save_persistence, load_config
import pandas as pd
import numpy as np
import os
from io import BytesIO
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Country
country = "US"
startDate = "1995-01-01"

# Path
ROOT = Path(__file__).resolve().parents[0]

# Load private environment variables
config = load_config(country)
prices, weights = load_input(country)

# Load Ismi
path = ROOT / "results" / f"{country.upper()}Ismi.parquet"
dfIsmi = pd.read_parquet(path)

# Divide by 100
dfIsmi = dfIsmi/100
dfIsmi = dfIsmi[dfIsmi.index >= startDate]

# Keep only rows with numeric values
headline = prices["HEADLINE"].dropna()
mom = np.log(headline).diff(1).dropna()
mom = mom[mom.index >= startDate]
yoy = np.log(headline).diff(12).dropna()
yoy = yoy[yoy.index >= startDate]

# Date limits
startDate = np.min(yoy.index)
endDate = np.max(yoy.index)

### Plot monthly and yearly inflation rate
fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

# MoM inflation (left axis)
ax1.plot(
    mom.index, mom*100,
    color="steelblue",
    linewidth=1,
    label="MoM inflation"
)
ax1.set_ylabel("MoM inflation (%)", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

# YoY inflation (right axis)
ax2.plot(
    yoy.index, yoy*100,
    color="darkred",
    linewidth=1.5,
    label="YoY inflation"
)
ax2.set_ylabel("YoY inflation (%)", color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")

ax1.set_xlim(pd.Timestamp(startDate), pd.Timestamp(endDate))

# Adjust Y-axis limits to visible observations
for ax, series in [(ax1, mom), (ax2, yoy)]:
    visible = series.loc[startDate:endDate].dropna() * 100

    ymin, ymax = visible.min(), visible.max()
    margin = 0.05 * (ymax - ymin)

    ax.set_ylim(ymin - margin, ymax + margin)

# Grid
ax1.grid(
    True,
    which="major",
    axis="both",
    linestyle="--",
    linewidth=0.6,
    alpha=0.3
)

ax1.set_axisbelow(True)

plt.tight_layout()
plt.show()

### AR(1) sufficient ? 
p = 1

# AR(1) à coefficient constant
Y = mom.values[p:]
X = sm.add_constant(mom.values[:-p])
model = sm.OLS(Y, X).fit()
print(model.summary())
resid = model.resid

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


## AR(1) avec ISMI pour distinguer persistance des chocs d'inertie de l'inflation ?
dfIsmi = dfIsmi.join(mom.rename("MoM"))
Y = dfIsmi["MoM"].values[p:]
x = pd.DataFrame([dfIsmi["MoM"].values[:-p], dfIsmi["ISMI"].values[p:]]).transpose()
X = sm.add_constant(x.values)
model = sm.OLS(Y, X).fit()
print(model.summary())
resid = model.resid

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