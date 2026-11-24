from fismi.io import load_input, load_config
from fismi.maths import residual_diagnostics, plot_residual_diagnostics
import numpy as np
import pandas as pd
from fismi.models import kalAlgoRw
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from fismi.maths import rmse, rolling_largest_ar_root, adfSummary, residualDiagnostics
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from statsmodels.stats.stattools import jarque_bera
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm 
import statistics as stat
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import arma_acf, arma_pacf
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from itertools import product
import warnings
import numpy as np
from statsmodels.tsa.arima_process import arma_impulse_response

########################################################
# Settings
# United States
BLUE = "#2B3A67"
ORANGE = "#F6AE2D"
RED = "#BB4350"
GREEN = "#268477"
INK = "#202A38"
MUTED = "#808A99"

COLORS = [BLUE, GREEN, ORANGE, RED, INK]

########################################################
# Load ARMA specifications
folder = "./results/2026-11-23 Comparisons/"
AR1 = pd.read_parquet(folder + "AR-1.parquet")
AR1_1960 = pd.read_parquet(folder + "AR-1-1960.parquet")

ARMA13 = pd.read_parquet(folder + "ARMA-1_3.parquet")
# ARMA16 = pd.read_parquet(folder + "ARMA-1_6.parquet")
ARMA112 = pd.read_parquet(folder + "ARMA-1_12.parquet")
# ARMA112_1995 = pd.read_parquet(folder + "ARMA-1_12-1995.parquet")

# Load ARMA parameters
folder = "./results/2026-11-23 Comparisons/"
paramAR1 = pd.read_json(folder + "AR-1.json")
paramAR1_1960 = pd.read_json(folder + "AR-1-1960.json")
paramARMA13 = pd.read_json(folder + "ARMA-1_3.json")
paramARMA112 = pd.read_json(folder + "ARMA-1_12.json")

mom = ARMA13['Inflation']
yoy = ARMA112['Inflation']
dates = mom.index
start_date = min(dates)
end_date = max(dates)

# Reindex on MoM...
ARMA13.reindex(dates)
AR1.reindex(dates)

# Models
# ============================================================
# CONSTANT ARMA BENCHMARKS
# ============================================================

specifications = {
    "TVP-AR(1) on monthly inflation since 1995":
        (AR1, 1, 0, 'MoM', paramAR1),
        
    "TVP-AR(1) on monthly inflation since 1960":
        (AR1_1960, 1, 0, 'MoM', paramAR1_1960),

    # "TVP-ARMA(1,6) on monthly inflation since 1960":
    #     (ARMA16, 1, 6, 'MoM'),

    "TVP-ARMA(1,3) on monthly inflation since 1960":
        (ARMA13, 1, 3, 'MoM', paramARMA13),

    "TVP-ARMA(1,12) on year-on-year inflation since 1960":
        (ARMA112, 1, 12, 'YoY', paramARMA112),
        
    # "TVP-ARMA(1,12) on year-on-year inflation since 1995":
    #     (ARMA112_1995, 1, 12, 'YoY')
}

arma_fix = {}

# Estimate constant-coefficient ARMA models
for name, (df, p, q, freq, params) in specifications.items():

    arma_fix[name] = ARIMA(
        df["Inflation"].dropna(),
        order=(p, 0, q),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True
    ).fit()

# ============================================================
# FIGURE 1 — FILTERED PERSISTENCE VS CONSTANT ARMA
# ============================================================
fig, ax = plt.subplots(figsize=(17, 6.5))

for i, (name, (df, p, q, freq, params)) in enumerate(specifications.items()):

    color = COLORS[i]
    rhoConstant = arma_fix[name].arparams[0]

    # Time-varying persistence
    ax.plot(
        df["RhoFiltered"].index,
        df["RhoFiltered"],
        color=color,
        linewidth=1.8,
        label=name,
        zorder=3
    )

    # Constant ARMA benchmark
    ax.axhline(
        rhoConstant,
        color=color,
        linestyle=(0, (5, 3)),
        linewidth=1.2,
        alpha=0.8,
        label=f"ARMA({p},{q}) fixed: {rhoConstant:.3f}",
        zorder=2
    )

# ============================================================
# AXES AND DATES
# ============================================================

# Exact date boundaries from the data
ax.set_xlim(start_date, end_date)
ax.margins(x=0)

# Major ticks every 5 years
ax.xaxis.set_major_locator(
    mdates.YearLocator(3)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)

# Minor ticks every year
ax.xaxis.set_minor_locator(
    mdates.YearLocator(1)
)

ax.tick_params(
    axis="x",
    which="major",
    labelsize=11,
    rotation=0,
    length=5
)

ax.tick_params(
    axis="x",
    which="minor",
    length=2.5
)

# ============================================================
# REFERENCE LINES
# ============================================================

ax.axhline(
    1,
    color="#626B76",
    linestyle=":",
    linewidth=1.1,
    alpha=0.8,
    label="Unit-root threshold"
)

# ============================================================
# STYLING
# ============================================================

ax.set_title(
    "United States : Inflation persistence (filtered estimates)",
    loc="left",
    fontsize=16,
    fontweight="bold",
    color="#202A38",
    pad=18
)

ax.set_ylabel(
    r"Persistence coefficient $\hat{\rho}_{t|t}$",
    fontsize=12
)

ax.set_xlabel("")

# Horizontal grid
ax.grid(
    axis="y",
    which="major",
    linestyle="-",
    linewidth=0.6,
    alpha=0.18
)

# Vertical grid every 5 years
ax.grid(
    axis="x",
    which="major",
    linestyle="--",
    linewidth=0.6,
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="y",
    labelsize=11
)

# Legend
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
    fontsize=10,
    columnspacing=2.5,
    handlelength=3
)

plt.tight_layout()
plt.show()



# ============================================================
# FIGURE 2 — SMOOTHED PERSISTENCE WITH 90% CI
# ============================================================
z = norm.ppf(0.95)  # Two-sided 90% CI = 1.64485

fig, ax = plt.subplots(figsize=(17, 6.5))
i=0
for key, model in specifications.items():

    color = COLORS[i]
    df = model[0]

    # Smoothed persistence and variance
    rho = df["RhoSmoothed"].dropna()
    variance = df['VarStateSmoothed'].dropna()

    # Confidence intervals
    se = np.sqrt(variance)

    lower = rho - z * se
    upper = rho + z * se

    # Smoothed persistence
    ax.plot(
        rho.index,
        rho,
        color=color,
        linewidth=1.8,
        label=key,
        zorder=3
    )

    # 90% confidence interval
    ax.fill_between(
        rho.index,
        lower,
        upper,
        color=color,
        alpha=0.12,
        linewidth=0,
        zorder=2
    )

    # Long-run persistence / initial mean
    # Replace with your estimated rho_bar values
    rho_bar = np.mean(df["RhoFiltered"])

    ax.axhline(
        rho_bar,
        color=color,
        linestyle=(0, (5, 3)),
        linewidth=1.3,
        alpha=0.8,
        label=rf"Empirical mean $E(\hat{{\rho}})$: {rho_bar:.2f}"
    )
    
    i+=1

# ============================================================
# AXES AND DATES
# ============================================================
ax.set_xlim(start_date, end_date)
ax.margins(x=0)

# Major ticks every 5 years
ax.xaxis.set_major_locator(mdates.YearLocator(3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Minor ticks every year
ax.xaxis.set_minor_locator(mdates.YearLocator(1))

ax.tick_params(axis="x", which="major", labelsize=11)
ax.tick_params(axis="x", which="minor", length=2.5)

# ============================================================
# REFERENCE LINE
# ============================================================

ax.axhline(
    1,
    color="#626B76",
    linestyle=":",
    linewidth=1,
    alpha=0.7,
    label="Unit-root threshold"
)

# ============================================================
# STYLING
# ============================================================

ax.set_title(
    "United States : Inflation persistence (smoothed estimates)",
    loc="left",
    fontsize=16,
    fontweight="bold",
    color="#202A38",
    pad=18
)

ax.set_ylabel(
    r"Smoothed persistence $\hat{\rho}_{t|T}$",
    fontsize=12
)

ax.set_xlabel("")

ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.6,
    alpha=0.18
)

ax.grid(
    axis="x",
    which="major",
    linestyle="--",
    linewidth=0.6)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False,
    fontsize=10,
    columnspacing=2.5,
    handlelength=3
)

plt.tight_layout()
plt.show()

# ============================================================
# FIGURES — ACF(1) = PACF(1)
# ============================================================

def finite_horizon_acf(rho, maParams, lag=1, H=12):
    """
    Finite-horizon model-implied autocorrelation
    for an ARMA(1,q) with frozen parameters.

    Parameters
    ----------
    rho : float
        Autoregressive coefficient.
    maParams : array
        MA polynomial [1, theta_1, ..., theta_q].
    lag : int
        Autocorrelation lag.
    H : int
        Finite propagation horizon.

    Returns
    -------
    float
        Corr(x_H, x_{H-lag}).
    """

    if not 1 <= lag <= H:
        raise ValueError("lag must be between 1 and H")

    if not np.isfinite(rho):
        return np.nan

    # Impulse response coefficients: psi_0, ..., psi_H
    psi = arma_impulse_response(
        ar=np.array([1.0, -rho]),
        ma=maParams,
        leads=H+1
    )

    # Covariance (up to sigma_e²)
    covariance = np.dot(
        psi[:-lag],
        psi[lag:]
    )

    # Variances (up to sigma_e²)
    variance_H = np.sum(psi**2)
    variance_H_lag = np.sum(psi[:-lag]**2)

    return covariance / np.sqrt(
        variance_H * variance_H_lag
    )
    
keyAr = 'TVP-AR(1) on monthly inflation since 1960'
keyArma = 'TVP-ARMA(1,3) on monthly inflation since 1960'

keys = [keyAr, keyArma]

H = 1
lag = 1

for key in keys:

    df = specifications[key][0]
    param = specifications[key][-1]["parameters"]
    q = specifications[key][2]

    # MA polynomial
    maParams = np.r_[1.0, param.iloc[2:2+q].values]

    for state in ["Filtered", "Smoothed"]:

        df[f"FiniteACF{lag}_{H}m_{state}"] = (
            df[f"Rho{state}"].apply(
                lambda rho: finite_horizon_acf(
                    rho, maParams, lag=lag, H=H
                )
            )
        )

# ============================================================
# COMPUTE AND PLOT FILTERED / SMOOTHED ACF
# ============================================================
colors = ["#344E73", "#C16B5A"]
labels = ["TVP-AR(1)", "TVP-ARMA(1,3)"]

mom = specifications[keyArma][0]["Inflation"]

# Empirical rolling first-order correlation
window = 60

empiricalACF = mom.rolling(window).corr(
    mom.shift(lag)
)

for state in ["Filtered", "Smoothed"]:

    fig, ax = plt.subplots(figsize=(16, 5))

    for key, color, label in zip(keys, colors, labels):

        df = specifications[key][0]

        ax.plot(
            df.index,
            df[f"FiniteACF{lag}_{H}m_{state}"],
            color=color,
            lw=1.8,
            label=label
        )

    # Empirical benchmark
    ax.plot(
        empiricalACF.index,
        empiricalACF,
        color="#808A99",
        linestyle="--",
        lw=1.3,
        alpha=0.85,
        label=f"Empirical ACF({lag}) — {window}m rolling"
    )

    ax.axhline(0, color="black", lw=0.8)

    ax.set_title(
        f"US inflation — Finite-horizon ACF({lag}) "
        f"at {H} months ({state.lower()})",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set(
        ylabel=f"ACF({lag}, H={H})",
        xlabel="",
        xlim=(mom.index.min(), mom.index.max())
    )

    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    ax.tick_params(axis="both", labelsize=11)

    ax.grid(axis="y", lw=0.6, alpha=0.25)
    ax.grid(axis="x", lw=0.6, alpha=0.35, linestyle="--")

    ax.set_axisbelow(True)
    ax.margins(x=0, y=0.08)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    plt.show()
    
# ============================================================
# FIGURES — ONE-STEP-AHEAD FIT AND CROSS-FREQUENCY RECONSTRUCTION
# ============================================================

BLUE = "#344E73"
RED = "#C16B5A"
GREEN = "#388477"
ORANGE = "#D5A044"
INK = "#202A38"
MUTED = "#808A99"

COLORS = [BLUE, RED, GREEN, ORANGE]
# ------------------------------------------------------------
# 4. RECONSTRUCTION FUNCTIONS
# ------------------------------------------------------------

def reconstruct_yoy(forecast_mom, mom):
    """
    One-step-ahead YoY forecast from MoM forecast.

    YoY_hat(t|t-1) =
        MoM_hat(t|t-1) + sum(MoM(t-j), j=1,...,11)
    """

    # Strictly lagged observed monthly inflation
    past_11 = (
        mom.shift(1)
        .rolling(window=11, min_periods=11)
        .sum()
    )

    return forecast_mom + past_11.reindex(
        forecast_mom.index
    )


def reconstruct_mom(forecast_yoy, yoy, mom):
    """
    One-step-ahead MoM forecast from YoY forecast.

    MoM_hat(t|t-1) =
        YoY_hat(t|t-1)
        - YoY(t-1)
        + MoM(t-12)
    """

    return (
        forecast_yoy
        - yoy.shift(1).reindex(forecast_yoy.index)
        + mom.shift(12).reindex(forecast_yoy.index)
    )

def calculate_metrics(observed, forecast):

    data = pd.concat(
        [observed.rename("observed"),
         forecast.rename("forecast")],
        axis=1
    ).dropna()

    if data.empty:
        return np.nan, np.nan

    errors = data["observed"] - data["forecast"]

    rmse = np.sqrt(np.mean(errors**2))
    nrmse = rmse / np.std(observed)
    mae = np.mean(np.abs(errors))

    return rmse, nrmse, mae


# ------------------------------------------------------------
# 6. CREATE THE THREE FIGURES
# ------------------------------------------------------------
i=0
for model_name, (model, p, q, freq, params) in specifications.items():

    df = model.copy()
    frequency = freq
    color = COLORS[i]

    # Forecast produced by Kalman before observing pi_t
    forecast = df["ForecastInflation"].astype(float)

    if frequency == "MoM":
        observed_native = mom.reindex(df.index)
        observed_other = yoy.reindex(df.index)
        forecast_other = reconstruct_yoy(
            forecast.dropna(),
            mom.dropna()
        )

        native_label = "Month-on-month"
        other_label = "Year-on-year"

    else:
        observed_native = yoy.reindex(df.index)
        observed_other = mom.reindex(df.index)
        forecast_other = reconstruct_mom(
            forecast.dropna(),
            yoy.dropna(),
            mom.dropna()
        )

        native_label = "Year-on-year"
        other_label = "Month-on-month"

    # --------------------------------------------------------
    # ALIGN DATA
    # --------------------------------------------------------

    native = pd.concat(
        [
            observed_native.rename("Observed"),
            forecast.rename("Forecast")
        ],
        axis=1
    ).dropna()

    other = pd.concat(
        [
            observed_other.rename("Observed"),
            forecast_other.rename("Forecast")
        ],
        axis=1
    ).dropna()

    if native.empty or other.empty:
        raise ValueError(
            f"No overlapping observations for {model_name}"
        )

    # Convert log differences into percentage points
    native *= 100
    other *= 100

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    rmse_native, nrmse_native, mae_native = calculate_metrics(
        native["Observed"],
        native["Forecast"]
    )

    rmse_other, nrmse_other, mae_other = calculate_metrics(
        other["Observed"],
        other["Forecast"]
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2, 1,
        figsize=(17, 9),
        sharex=True,
        constrained_layout=True
    )

    panels = [
        (
            axes[0],
            native,
            native_label,
            rmse_native,
            nrmse_native,
            mae_native
        ),
        (
            axes[1],
            other,
            other_label,
            rmse_other,
            nrmse_other,
            mae_other
        )
    ]
    
    i+=1

    for ax, data, label, rmse_val, nrmse_val, mae_val in panels:

        x = data.index

        # Observed inflation
        ax.plot(
            x,
            data["Observed"],
            color=MUTED,
            linewidth=1.6,
            alpha=0.85,
            linestyle=':',
            label="Observed inflation",
            zorder=2
        )

        # One-step-ahead forecast
        ax.plot(
            x,
            data["Forecast"],
            color=color,
            linewidth=1.5,
            alpha=0.95,
            label="One-step-ahead forecast",
            zorder=3
        )

        ax.set_title(
            f"{label} inflation",
            loc="left",
            fontsize=14,
            fontweight="bold",
            color=INK,
            pad=15
        )

        ax.set_ylabel(
            "Inflation (%)",
            fontsize=12
        )

        # Forecast accuracy
        ax.text(
            0.99,
            0.96,
            f"RMSE  {rmse_val:.3f} pp  |  "
            f"NRMSE  {nrmse_val:.3f} pp  |  "
            f"MAE  {mae_val:.3f} pp",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            color=color,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
                pad=5
            )
        )

        ax.legend(
            loc="upper left",
            frameon=False,
            ncol=2,
            fontsize=10
        )

        # Exact date boundaries from the data
        ax.set_xlim(start_date, end_date)
        ax.margins(x=0)

        # Major ticks every 5 years
        ax.xaxis.set_major_locator(
            mdates.YearLocator(3)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y")
        )

        # Minor ticks every year
        ax.xaxis.set_minor_locator(
            mdates.YearLocator(1)
        )

        ax.tick_params(
            axis="x",
            which="major",
            labelsize=11,
            rotation=0,
            length=5
        )

        ax.tick_params(
            axis="x",
            which="minor",
            length=2.5
        )
        
        # Horizontal grid
        ax.grid(
            axis="y",
            which="major",
            linestyle="-",
            linewidth=0.6,
            alpha=0.18
        )

        # Vertical grid every 5 years
        ax.grid(
            axis="x",
            which="major",
            linestyle="--",
            linewidth=0.6,
        )


    # --------------------------------------------------------
    # COMMON DATE AXIS
    # --------------------------------------------------------
    # Show dates on both panels
    axes[0].tick_params(
        axis="x",
        labelbottom=True
    )

    fig.suptitle(
        f"US PCE inflation — {model_name}",
        fontsize=17,
        fontweight="bold",
        color=INK
    )

    plt.show()

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(f"\n{model_name}")
    print(
        f"Native frequency ({native_label}): "
        f"RMSE = {rmse_native:.4f} pp, "
        f"NRMSE = {nrmse_native:.4f} pp, "
        f"MAE = {mae_native:.4f} pp"
    )

    print(
        f"Reconstructed frequency ({other_label}): "
        f"RMSE = {rmse_other:.4f} pp, "
        f"NRMSE = {nrmse_other:.4f} pp, "
        f"MAE = {mae_other:.4f} pp"
    )


# ============================================================
# 4. KALMAN GAIN AND FORECAST ERRORS
# ============================================================
for model_name, (model, p, q, freq, params) in specifications.items():
    # Variables
    Kt = model["KalmanGain"]
    
    Err = model["Residuals"]
    stdY = np.sqrt(model["forecastVarY"])
    innovationErr = Err / stdY  # Standardized one-step-ahead innovations
    dates = model.index
    
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 8),
        sharex=True,
        constrained_layout=True
    )

    # Panel 1: Kalman gain for persistence
    axes[0].plot(
        dates, Kt,
        color=BLUE, lw=1.3,
        label=r"Kalman gain for persistence $K_t^\rho$"
    )

    axes[0].axhline(
        0, color="black", lw=0.8, alpha=0.5
    )

    axes[0].set_title("Kalman gain")
    axes[0].set_ylabel(r"$K_t^\rho$")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    # Panel 2: Standardized innovations
    axes[1].plot(
        dates, innovationErr,
        color=ORANGE, lw=1.1,
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

    for ax in axes:
        ax.grid(alpha=0.35)

        # Larger fonts
        ax.tick_params(axis="both", labelsize=14)
        ax.title.set_fontsize(18)
        ax.xaxis.label.set_fontsize(15)
        ax.yaxis.label.set_fontsize(15)
        ax.legend(loc="upper left", fontsize=13)

        # Dates every 2 years
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Display dates on both panels
        ax.tick_params(axis="x", labelbottom=True)

        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right"
        )

    fig.suptitle(
        f"Kalman Gain and standardised errors - {model_name}",
        fontsize=17,
        fontweight="bold",
        color=INK
    )
    plt.show()




# ========================================================
# Diagnostics sur les résidus
# ========================================================

# Store name, residuals from TVP and ARMA specifications
# ========================================================
# STANDARDIZED INNOVATIONS — ALL SPECIFICATIONS
# ========================================================

residuals = {}

for name, (df, p, q, freq, params) in specifications.items():

    # ----------------------------------------------------
    # 1. TVP specifications
    # ----------------------------------------------------

    errors = df["Residuals"].astype(float)
    variance = df["forecastVarY"].astype(float)

    # Standardized one-step-ahead innovations
    z_tvp = errors / np.sqrt(variance)

    residuals[(name, "q")] = (z_tvp, q)

    # ----------------------------------------------------
    # 2. Constant-parameter specifications
    # ----------------------------------------------------

    model = arma_fix[name]
    fixed_name = name.removeprefix("TVP-")

    # Standardized Kalman forecast errors
    z_fixed = pd.Series(model.filter_results.standardized_forecasts_error[0])
    residuals[(fixed_name, "q")] = (z_fixed, q)





results = {}
for (name, _), (resid, q) in residuals.items():

    results[name] = residual_diagnostics(
        residuals=resid,
        p=1,
        q=q
    )
    
    plot_residual_diagnostics(
        z=resid,
        name=name,
        lags=36
    )

dfDiagnostics = pd.DataFrame.from_dict(
    results
)
print(dfDiagnostics.round(4))



# L'AR(1) est-il stationnaire ? 
resid = specifications['TVP-AR(1) on monthly inflation since 1960'][0]['Residuals']
dfADF = adfSummary(resid)
print(dfADF.round(4).to_string(index=False))


# Attention : la réponse impulsionnelle des chocs dépend aussi de la composante MA ; c'est le rapport des racines qui compte

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# 1. MODELS — CONSTANT COEFFICIENTS
# ============================================================

# Re-estimate all models on the same monthly sample for a meaningful comparison (1960).
orders = {
    "AR(1)": (1, 0, 0),
    "ARMA(1,3)": (1, 0, 3),
    "ARMA(1,6)": (1, 0, 6)
}

fixed_models = {}

for name, order in orders.items():

    fixed_models[name] = ARIMA(
        mom,
        order=order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True
    ).fit()

# ============================================================
# 2. IMPULSE RESPONSE FUNCTIONS
# ============================================================

H = 60
horizons = np.arange(H + 1)

colors = {
    "AR(1)": "#344E73",
    "ARMA(1,3)": "#C16B5A",
    "ARMA(1,6)": "#388477"
}

def arma_irf(rho, ma_coefficients, H=60):

    ma = np.asarray(ma_coefficients, dtype=float)

    psi = np.zeros(H + 1)
    psi[0] = 1.0

    for h in range(1, H + 1):

        a_h = ma[h - 1] if h <= len(ma) else 0.0

        psi[h] = rho * psi[h - 1] + a_h

    return psi


irfs = {}
summary_rows = []

for name, model in fixed_models.items():

    rho = float(model.arparams[0])
    ma = np.asarray(model.maparams)

    psi = arma_irf(rho, ma, H)

    irfs[name] = psi

    # Cumulative impulse response
    cumulative = np.cumsum(psi)

    # Absolute half-life: first horizon where the response
    # stays below half its initial magnitude thereafter.
    below_half = np.abs(psi) <= 0.5

    half_life = np.nan

    for h in range(H + 1):
        if np.all(below_half[h:]):
            half_life = h
            break

    # Long-run multiplier, if stationary
    long_run = (1 + np.sum(ma)) / (1 - rho)

    summary_rows.append({
        "Model": name,
        "AR coefficient": rho,
        "IRF h=1": psi[1],
        "IRF h=3": psi[3],
        "IRF h=12": psi[12],
        "IRF h=24": psi[24],
        "IRF h=60": psi[60],
        "Cumulative IRF h=12": cumulative[12],
        "Cumulative IRF h=60": cumulative[60],
        "Long-run multiplier": long_run,
        "Half-life (months)": half_life
    })

summary = pd.DataFrame(summary_rows).set_index("Model")

print(summary.round(4).to_string())

# ============================================================
# 3. FIGURE 1 — IMPULSE RESPONSES
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5.5))

for name, psi in irfs.items():

    ax.plot(
        horizons,
        psi,
        color=colors[name],
        linewidth=2,
        label=name
    )

ax.axhline(0, color="black", linewidth=0.8)

ax.axhline(
    0.5,
    color="#808A99",
    linestyle="--",
    linewidth=1,
    label="50% of initial shock"
)

ax.set_title(
    "US PCE inflation — Impulse response to a unit innovation",
    loc="left",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Months after shock")
ax.set_ylabel(r"Impulse response $\psi_h$")

ax.set_xlim(0, H)
ax.set_xticks(np.arange(0, H + 1, 6))

ax.grid(alpha=0.2)
ax.legend(frameon=False, ncol=3)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# ============================================================
# 4. FIGURE 2 — CUMULATIVE IMPULSE RESPONSES
# ============================================================

fig, ax = plt.subplots(figsize=(14, 5.5))

for name, psi in irfs.items():

    ax.plot(
        horizons,
        np.cumsum(psi),
        color=colors[name],
        linewidth=2,
        label=name
    )

ax.set_title(
    "US PCE inflation — Cumulative impulse responses",
    loc="left",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Months after shock")
ax.set_ylabel(r"Cumulative response $\sum_{j=0}^{h}\psi_j$")

ax.set_xlim(0, H)
ax.set_xticks(np.arange(0, H + 1, 6))

ax.grid(alpha=0.2)
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# ============================================================
# 5. FIGURE 3 — AR AND MA ROOTS
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

theta = np.linspace(0, 2*np.pi, 500)

ax.plot(
    np.cos(theta),
    np.sin(theta),
    color="#808A99",
    linestyle="--",
    linewidth=1.2,
    label="Unit circle"
)

for name, model in fixed_models.items():

    color = colors[name]

    ar_roots = np.asarray(model.arroots)
    ma_roots = np.asarray(model.maroots)

    # Inverse roots: stationarity/invertibility requires
    # their moduli to be strictly below one.
    if len(ar_roots):

        inv_ar = 1 / ar_roots

        ax.scatter(
            inv_ar.real,
            inv_ar.imag,
            color=color,
            marker="x",
            s=110,
            linewidths=2,
            label=f"{name} — AR"
        )

    if len(ma_roots):

        inv_ma = 1 / ma_roots

        ax.scatter(
            inv_ma.real,
            inv_ma.imag,
            facecolors="none",
            edgecolors=color,
            marker="o",
            s=95,
            linewidths=1.8,
            label=f"{name} — MA"
        )

ax.axhline(0, color=INK, linewidth=0.5)
ax.axvline(0, color=INK, linewidth=0.5)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect("equal", adjustable="box")

ax.set_title(
    "Inverse AR and MA roots",
    loc="left",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Real part")
ax.set_ylabel("Imaginary part")

ax.grid(alpha=0.2)

ax.legend(
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.show()

# ============================================================
# 6. ROOT CANCELLATION DIAGNOSTIC
# ============================================================

print("\nROOT CANCELLATION DIAGNOSTIC")

for name, model in fixed_models.items():

    ar_roots = np.asarray(model.arroots)
    ma_roots = np.asarray(model.maroots)

    if len(ar_roots) == 0 or len(ma_roots) == 0:
        continue

    distances = np.abs(
        ar_roots[:, None] - ma_roots[None, :]
    )

    i, j = np.unravel_index(
        np.argmin(distances),
        distances.shape
    )

    print(f"\n{name}")
    print(f"AR root: {ar_roots[i]}")
    print(f"Closest MA root: {ma_roots[j]}")
    print(f"Distance: {distances[i,j]:.6f}")


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------------------------------------
# Parameters
# -------------------------------------------------
H = 12

# Fixed MA coefficients from your ARMA(1,3)
a1, a2, a3 = specifications['TVP-ARMA(1,3) on monthly inflation since 1960'][-1]['parameters'][2:5].values

# Fixed MA coefficients from ARMA(1,12)
a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 = specifications['TVP-ARMA(1,12) on year-on-year inflation since 1960'][-1]['parameters'][2:14].values


# -------------------------------------------------
# AR(1): IRF at horizon H
# -------------------------------------------------
rho_ar1 = AR1["RhoFiltered"]

irf_ar1 = rho_ar1 ** H

# -------------------------------------------------
# ARMA(1,3): IRF at horizon H
# -------------------------------------------------
rho_arma = ARMA13["RhoFiltered"]

psi1 = rho_arma + a1
psi2 = rho_arma * psi1 + a2
psi3 = rho_arma * psi2 + a3

if H == 0:
    irf_arma = rho_arma * 0 + 1
elif H == 1:
    irf_arma = psi1
elif H == 2:
    irf_arma = psi2
else:
    irf_arma = rho_arma ** (H - 3) * psi3

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(
    irf_ar1.index,
    irf_ar1,
    color="#344E73",
    linewidth=1.8,
    label=f"TVP-AR(1) — IRF({H})"
)

ax.plot(
    irf_arma.index,
    irf_arma,
    color="#C16B5A",
    linewidth=1.8,
    label=f"TVP-ARMA(1,3) — IRF({H})"
)

ax.axhline(0, color="black", linewidth=0.8)

ax.set_title(
    f"US inflation — Persistence of a unit shock after {H} months",
    loc="left",
    fontsize=15,
    fontweight="bold"
)

ax.set_ylabel(f"Impulse response at horizon {H}")
ax.set_xlabel("")

ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.grid(alpha=0.2)
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
