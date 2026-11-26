from fismi.io import load_input, load_config
import numpy as np
import pandas as pd
from fismi.models import kalAlgoRw
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from fismi.maths import rmse, rolling_largest_ar_root, adfSummary, residualDiagnostics
from scipy.optimize import minimize
import matplotlib.dates as mdates
import statsmodels.api as sm 
import statistics as stat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import acf, pacf

from itertools import product
import warnings

########################################################
# Settings
# Select country
BLEU = "#2B3A67"
ORANGE = "#F6AE2D"
country = "US"
cMu = True
cRho = True

print(f"\n--- {country} ---")
config = load_config(country)
startDate = config["persistence"]["startDate"]
prices, weights = load_input(country)

# Resize on startDate
# prices = prices[prices.index >= "1994-12-12"]

inflationGa = np.log(
    prices["HEADLINE"].dropna()
).diff(
    12
).dropna()

# Glissement annuel
inflation = np.log(
    prices["HEADLINE"].dropna()
).diff(
    1
).dropna().reindex(inflationGa.index)

# Dates
dates = inflationGa.index

########################################################
# Table 1 : Statistiques descriptives
inflation_dict = {
    "MoM": inflation,
    "YoY": inflationGa
}

periods = {
    "1960–2026": ("1960-01-01", "2026-12-31"),
    "1995–2026": ("1995-01-01", "2026-12-31")
}

lags = [1, 3, 6, 12, 24, 36]

# Set to 100 if inflation is stored in decimal units
# Set to 1 if already expressed in percentage points
scale = 100

# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_stats(series, lags=lags):

    x = series.dropna().astype(float).sort_index() * scale
    n = len(x)
    
    acf_values = acf(
        x, nlags=max(lags),
        adjusted=False, fft=True
    )

    pacf_values = pacf(
        x, nlags=max(lags),
        method="ywm"
    )

    jb = jarque_bera(x)

    result = {
        "Observations": n,
        "Mean (%)": x.mean(),
        "Median (%)": x.median(),
        "Minimum (%)": x.min(),
        "Maximum (%)": x.max(),
        "Variance (pp²)": x.var(ddof=1),
        "Std. deviation (pp)": x.std(ddof=1),
        "Skewness": skew(x, bias=False),
        "Excess kurtosis": kurtosis(x, fisher=True, bias=False)
    }

    for lag in lags:
        result[f"ACF({lag})"] = acf_values[lag]

    for lag in lags:
        result[f"PACF({lag})"] = pacf_values[lag]

    # result["Jarque-Bera p-value"] = jb.pvalue

    return pd.Series(result)


# ============================================================
# 3. BUILD COMPARISON TABLE
# ============================================================
results = {}

for period, (start, end) in periods.items():

    for frequency, series in inflation_dict.items():

        series = series.copy()
        series.index = pd.to_datetime(series.index)
        series.index = series.index.to_period("M").to_timestamp()
        series = series.sort_index()

        sample = series.loc[start:end]

        results[(period, frequency)] = descriptive_stats(
            sample, lags=lags
        )

stats_table = pd.DataFrame(results)

stats_table.columns = pd.MultiIndex.from_tuples(
    stats_table.columns,
    names=["Sample", "Frequency"]
)

print(stats_table.round(4).to_string())

# Export to LaTeX
latex_table = stats_table.to_latex(
    float_format="%.4f",
    multicolumn=True,
    multicolumn_format="c",
    escape=True,
    caption="Descriptive statistics of US PCE inflation",
    label="tab:inflation_descriptive",
    column_format="lcccc"
)

with open("inflation_descriptive.tex", "w", encoding="utf-8") as f:
    f.write(latex_table)


########################################################
# Chart 1
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(16, 8),
    sharex=True,
    constrained_layout=True
)

# Panel 1 : Inflation YoY
ax1.plot(
    dates,
    inflationGa*100,
    color=BLEU,
    linewidth=1.8
)

ax1.axhline(0, color="black", lw=0.8, alpha=0.6)
ax1.set_title("Year-on-year PCE inflation", loc="left",
              fontsize=14, fontweight="bold", pad=12)
# ax1.legend(loc="upper left", frameon=False, fontsize=14)

# Panel 2 : Inflation MoM
ax2.plot(
    dates,
    inflation*100,
    color=ORANGE,
    linewidth=1.5
)

ax2.axhline(0, color="black", lw=0.8, alpha=0.6)
ax2.set_title("Month-on-month PCE inflation", loc="left",
              fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("")

# ax2.legend(loc="upper left", frameon=False,  fontsize=14)

# Shared formatting
for ax in (ax1, ax2):
    ax.grid(True, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tick labels
    ax.tick_params(axis="both", labelsize=12)

    # Date formatting
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    # Exact date boundaries
    ax.set_xlim(dates[0], dates[-1])
    ax.margins(x=0)

# Show dates on both panels
ax1.tick_params(axis="x", labelbottom=True)

# Rotate dates for readability
for ax in (ax1, ax2):
    ax.tick_params(axis="x", labelrotation=45)

fig.suptitle(
    "United States",
    fontsize=17,
    fontweight="bold"
)

plt.show()

########################################################
# Diagnosis

# 1) Plot the series
# Plot...
fig, ax1 = plt.subplots(figsize=(16, 5))
ax2 = ax1.twinx()

# Persistence — left axis
ax1.plot(
    dates,
    inflation*100,
    color="grey",
    linewidth=1.5,
    label=r"Monthly inflation rate ($\pi_t$)"
)

ax1.axhline(
    0, color="grey", linestyle=":", linewidth=1
)
ax1.axhline(
    1, color="grey", linestyle=":", linewidth=1
)

ax1.set_ylabel(r"Monthly inflation rate")
ax1.tick_params(axis="y")

# Intercept — right axis
ax2.plot(
    dates,
    inflationGa*100,
    color="darkblue",
    alpha=0.7,
    linewidth=1.5,
    label=r"Monthly inflation rate ($\pi^Y_t$)"
)

ax2.set_ylabel("Inflation (%)", color="darkblue")
ax2.tick_params(axis="y", labelcolor="darkblue")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# ACF, PACF
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Mom inflation
plot_acf(inflation, lags=36, ax=axes[0,0])
axes[0,0].set_title("Monthly inflation (ACF)")

plot_pacf(inflation, lags=36, ax=axes[0,1], method="ywm")
axes[0,1].set_title("Monthly inflation (PACF)")

axes[0,0].grid(alpha=0.3)
axes[0,1].grid(alpha=0.3)

# YoY inflation
plot_acf(inflationGa, lags=36, ax=axes[1,0])
axes[1,0].set_title("Year-on-year inflation (ACF)")

plot_pacf(inflationGa, lags=36, ax=axes[1,1], method="ywm")
axes[1,1].set_title("Year-on-year inflation (PACF)")

axes[1,0].grid(alpha=0.3)
axes[1,1].grid(alpha=0.3)

for ax in axes.flat:
    ax.grid(True, linestyle="--", alpha=0.4)
    
plt.tight_layout()
plt.show()


# Stratégie séquentielle simplifiée ADF
Data = {
    'MoM': inflation,
    'YoY': inflationGa,
    'deltaYoY': inflationGa.diff().dropna()
}

for inf in Data.keys():
    print(f"Run ADF on {inf} series...")
    dfADF = adfSummary(Data[inf])
    print(dfADF.round(4).to_string(index=False))


########################################################
# Quel processus ARMA suit l'inflation mensuelle ?
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Mom inflation
plot_acf(Data["MoM"], lags=36, ax=axes[0,0])
axes[0,0].set_title("Monthly inflation (ACF)")

plot_pacf(Data["MoM"], lags=36, ax=axes[0,1], method="ywm")
axes[0,1].set_title("Monthly inflation (PACF)")

axes[0,0].grid(alpha=0.3)
axes[0,1].grid(alpha=0.3)

# YoY inflation
plot_acf(Data["deltaYoY"], lags=36, ax=axes[1,0])
axes[1,0].set_title("Delta year-on-year inflation (ACF)")

plot_pacf(Data["deltaYoY"], lags=36, ax=axes[1,1], method="ywm")
axes[1,1].set_title("Delta year-on-year inflation (PACF)")

axes[1,0].grid(alpha=0.3)
axes[1,1].grid(alpha=0.3)


for ax in axes.flat:
    ax.grid(True, linestyle="--", alpha=0.4)
    
plt.tight_layout()
plt.show()


# Best ARMA(p,q) process
# Maximum orders
p_max = 12
q_max = 12
allSeriesArima = {}
bestModels = {}
IC = "BIC"

for inf in ["MoM", "YoY","deltaYoY"]:

    results = []
    series = Data[inf].dropna()

    for p, q in product(range(p_max + 1), range(q_max + 1)):

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(
                    series,
                    order=(p, 0, q),
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True
                ).fit()

            if not model.mle_retvals.get("converged", False):
                continue

            bestModels[(inf, p, q)] = model
            results.append({
                "p": p,
                "q": q,
                "AIC": model.aic,
                "BIC": model.bic,
                "NLL": -model.llf
            })

        except (ValueError, np.linalg.LinAlgError):
            continue

    # Store results for each inflation series
    allSeriesArima[inf] = (
        pd.DataFrame(results)
        .sort_values(IC)
        .reset_index(drop=True)
    )


# Best specifications
bestMom = allSeriesArima["MoM"].iloc[0,:]
bestYoy = allSeriesArima["YoY"].iloc[0,:]
bestDeltaYoy = allSeriesArima["deltaYoY"].iloc[0,:]

print(f"Best ARMA for monthly inflation according to {IC}:")
print(bestMom)

print(f"\nBest ARMA for year-on-year inflation according to {IC}:")
print(bestYoy)

print(f"\nBest ARMA for delta year-on-year inflation according to {IC}:")
print(bestDeltaYoy)

# ARMA(1,12) specification (pas de racine unitaire ici, cf. ADF)
model = ARIMA(
    Data['YoY'],
    order=(1, 0, 12),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
).fit()

allSeriesArima["YoY"].loc[len(allSeriesArima["YoY"])] = {
    "p": p,
    "q": q,
    "AIC": model.aic,
    "BIC": model.bic,
    "NLL": -model.llf
}

bestModels['YoY',1,12] = model

# Residual diagnostics: meilleur modèle MoM et YoY (1,12)
arimaModels = {
    'MoM':bestModels[('MoM',bestMom['p'],bestMom['q'])],
    'YoY':bestModels[('YoY', 1, 12)],
    # 'deltaYoY':bestModels[('deltaYoY',bestDeltaYoy['p'],bestDeltaYoy['q'])]
}

pOpt = {
    'MoM': bestMom['p'],
    'YoY': 1,
    # 'deltaYoY':bestDeltaYoy['p']
}

qOpt = {
    'MoM': bestMom['q'],
    'YoY': 12,
    # 'deltaYoY':bestDeltaYoy['q']
}

window = 120

for inf in arimaModels.keys():
    # Standard deviation
    std = np.std(Data[inf])
    
    # Selection
    mod = arimaModels[inf]
    p_opt = pOpt[inf]
    q_opt = qOpt[inf]
    
    # Start analysis... Some charts
    print("\n")
    print(f"Residual diagnostics for {inf}...")
    mod.plot_diagnostics(figsize=(13, 8))
    plt.tight_layout()
    plt.show()
    
    # Some tests on standardised one-step ahead forecast errors
    # Standardized one-step-ahead forecast errors
    resid = mod.filter_results.standardized_forecasts_error[0]
    resid = np.asarray(resid)
    resid = resid[np.isfinite(resid)]
    print(residualDiagnostics(resid))
    
    # RMSE et NRMSE
    errors = np.asarray(mod.resid)
    rmse = np.sqrt(np.mean(errors**2))
    nrmse = rmse / std

    print(f"RMSE:  {rmse:.6f}")
    print(f"NRMSE: {nrmse:.4f}")


    # AR roots: stationarity and causality
    ar_roots = mod.arroots
    ma_roots = mod.maroots

    if len(ar_roots) > 0:
        rho_max = 1 / np.min(np.abs(ar_roots))

        if np.all(np.abs(ar_roots) > 1):
            print(f"ARMA({p_opt},{q_opt}) is stationary and causal")
        else:
            print(f"ARMA({p_opt},{q_opt}) is not causal/stationary under the usual ARMA representation")
    else:
        rho_max = 0.0

    # MA roots: invertibility
    if len(ma_roots) > 0:
        phi_max = 1 / np.min(np.abs(ma_roots))

        if np.all(np.abs(ma_roots) > 1):
            print(f"ARMA({p_opt},{q_opt}) is invertible")
        else:
            print(f"ARMA({p_opt},{q_opt}) is non-invertible")
    else:
        phi_max = 0.0

    print(f"Selected ARMA({p_opt},{q_opt})")
    print(f"Largest AR eigenvalue modulus: {rho_max:.4f}")
    print(f"Largest MA eigenvalue modulus: {phi_max:.4f}")


## Largest Autoregressive Root
# A partir de l'ARMA identifié, LAR comme estimation de la persistance du processus, sur une fenêtre de 10 ans
results = rolling_largest_ar_root(
    Data['MoM'] * 100,
    window=window,
    p=pOpt['MoM'],
    q=qOpt['MoM'],
    trend='c'
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    results.index,
    results["LargestRoot"],
    color="darkblue",
    linewidth=1.8,
    label="Largest inverse AR root modulus"
)

ax.axhline(
    1,
    color="darkred",
    linestyle="--",
    linewidth=1,
    label="Unit root boundary"
)

ax.set(
    title=f"US Inflation Persistence — Rolling {window/12:.0f}-year ARMA({pOpt['YoY']},{qOpt['YoY']})",
    ylabel="Largest inverse AR root modulus",
    xlabel=""
)

ax.legend(frameon=False)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()


########################################################
# Y a t'il preuve de changement de moyenne post-Covid ? 
# Covid dummy
covidDummy = pd.Series(
    (inflation.index >= pd.Timestamp("2020-03-01")).astype(int),
    index=inflation.index,
    name="PostCovid"
)

# Model with Covid dummy
model1 = ARIMA(
    inflation,
    exog=covidDummy,
    order=(p_opt, 0, q_opt),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
).fit()

print(model1.summary())


########################################################
# Le changement de niveau s'est-il résorbé ?
# Hypothesized break dates
T1 = "2021-04-01"
T2 = "2024-01-01"

exog = pd.DataFrame(index=inflation.index)

exog["D1"] = (
    inflation.index >= pd.Timestamp(T1)
).astype(int)

exog["D2"] = (
    inflation.index >= pd.Timestamp(T2)
).astype(int)

model = ARIMA(
    inflation,
    exog=exog,
    order=(p_opt, 0, q_opt),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
).fit()

print(model.summary())

# Mean shift after the second break
delta_post = model.params["D1"] + model.params["D2"]

# Variance of the linear combination
cov = model.cov_params()

var_post = (
    cov.loc["D1", "D1"]
    + cov.loc["D2", "D2"]
    + 2 * cov.loc["D1", "D2"]
)

se_post = np.sqrt(var_post)

# Wald test: H0 = delta1 + delta2 = 0
z_stat = delta_post / se_post
p_value = 2 * norm.sf(abs(z_stat))

ci_low = delta_post - 1.96 * se_post
ci_high = delta_post + 1.96 * se_post

print(f"Post-period mean shift: {delta_post:.3f} pp")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Wald test p-value: {p_value:.4f}")



