import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from statsmodels.stats.stattools import jarque_bera
import warnings
from scipy.stats import norm, gaussian_kde


# Simplified ADF test
def adfSummary(inflation, CI='BIC', lags=12):

    results = []

    for name, reg in [
        ("M1: No constant", "n"),
        ("M2: Constant", "c"),
        ("M3: Constant + Trend", "ct")
    ]:
        model = adfuller(
            inflation,
            regression=reg,
            autolag=CI,
            maxlag=lags,
            regresults=True
        )

        ols = model[-1].resols

        results.append({
            "Model": name,
            "ADF statistic": model[0],
            "P-value": model[1],
            "Critical 5%": model[2]["5%"],
            "Constant p-value": (
                ols.pvalues[-2] if reg == "ct"
                else ols.pvalues[-1] if reg == "c"
                else float("nan")
            ),
            "Trend p-value": (
                ols.pvalues[-1] if reg == "ct"
                else float("nan")
            ),
            "Reject H0 (5%)": model[1] < 0.05
        })

    return pd.DataFrame(results)


def residual_diagnostics(z, p=1, q=0):

    # Ljung-Box
    lb = acorr_ljungbox(
        z,
        lags=[2, 6, 12, 24, 36],
        model_df=p+q,
        return_df=True
    )

    # ARCH-LM
    arch = het_arch(z, nlags=12, ddof=p+q)

    # Jarque-Bera
    jb = jarque_bera(z)

    return {
        "LB(2) stat": lb.loc[2, "lb_stat"],
        "LB(2) p-value": lb.loc[2, "lb_pvalue"],
        "LB(6) stat": lb.loc[6, "lb_stat"],
        "LB(6) p-value": lb.loc[6, "lb_pvalue"],
        "LB(12) stat": lb.loc[12, "lb_stat"],
        "LB(12) p-value": lb.loc[12, "lb_pvalue"],
        "LB(24) stat": lb.loc[24, "lb_stat"],
        "LB(24) p-value": lb.loc[24, "lb_pvalue"],
        "LB(36) stat": lb.loc[36, "lb_stat"],
        "LB(36) p-value": lb.loc[36, "lb_pvalue"],
        "ARCH(12) stat": arch[0],
        "ARCH(12) p-value": arch[1],
        "JB stat": jb[0],
        "JB p-value": jb[1],
        "Skewness": jb[2],
        "Kurtosis": jb[3]
    }


def plot_residual_diagnostics(z, name, lags=36):
    BLUE = "#344E73"
    RED = "#C16B5A"

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 9),
        constrained_layout=True
    )

    fig.suptitle(
        f"Residual diagnostics — {name}",
        fontsize=17,
        fontweight="bold"
    )

    # ========================================================
    # PANEL 1 — STANDARDIZED RESIDUALS
    # ========================================================

    ax = axes[0, 0]

    ax.plot(
        z.index, z,
        color=BLUE,
        lw=0.9
    )

    ax.axhline(0, color="black", lw=0.8)

    for bound in [-1.96, 1.96]:
        ax.axhline(
            bound,
            color=RED,
            linestyle="--",
            lw=1,
            alpha=0.8
        )

    ax.set_title("Standardized residuals", loc="left")
    ax.set_ylabel("Standard deviations")

    if isinstance(z.index, pd.DatetimeIndex):
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.margins(x=0)

    # ========================================================
    # PANEL 2 — HISTOGRAM AND DENSITY
    # ========================================================

    ax = axes[0, 1]

    x = np.linspace(
        min(z.min(), -4),
        max(z.max(), 4),
        500
    )

    ax.hist(
        z,
        bins=35,
        density=True,
        color=BLUE,
        alpha=0.30,
        edgecolor="white",
        label="Residuals"
    )

    ax.plot(
        x,
        gaussian_kde(z)(x),
        color=BLUE,
        lw=1.8,
        label="Empirical density"
    )

    ax.plot(
        x,
        norm.pdf(x),
        color=RED,
        linestyle="--",
        lw=1.8,
        label="N(0,1)"
    )

    ax.set_title("Histogram and estimated density", loc="left")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    # ========================================================
    # PANEL 3 — NORMAL Q-Q PLOT
    # ========================================================

    ax = axes[1, 0]

    sm.qqplot(
        z,
        line="45",
        ax=ax,
        marker="o",
        markersize=3,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        alpha=0.65
    )

    ax.lines[-1].set_color(RED)
    ax.lines[-1].set_linestyle("--")
    ax.lines[-1].set_linewidth(1.5)

    ax.set_title("Normal Q-Q plot", loc="left")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")

    # ========================================================
    # PANEL 4 — RESIDUAL AUTOCORRELATION
    # ========================================================

    ax = axes[1, 1]

    plot_acf(
        z,
        lags=min(lags, len(z)-1),
        alpha=0.05,
        zero=False,
        bartlett_confint=False,
        ax=ax,
        color=BLUE,
        vlines_kwargs={"colors": BLUE, "linewidth": 1.2}
    )

    ax.set_title("Residual autocorrelation", loc="left")
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("ACF")

    # ========================================================
    # COMMON STYLING
    # ========================================================

    for ax in axes.flat:

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            linestyle="-",
            linewidth=0.6,
            alpha=0.20
        )

        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=10)

        ax.title.set_fontsize(12)
        ax.title.set_fontweight("bold")

    plt.show()

    return fig


def residualDiagnostics(resid, lags=12):

    # Standardized one-step-ahead forecast errors

    # Ljung-Box
    lb = acorr_ljungbox(
        resid, lags=[12, 18, 24, 36], model_df=12, return_df=True
    )

    # Jarque-Bera
    jb = jarque_bera(resid)

    # Engle ARCH-LM
    arch = het_arch(resid, nlags=lags)

    results = pd.DataFrame({
        "Test": [
            "Ljung-Box",
            "Jarque-Bera",
            "ARCH-LM"
        ],
        "Statistic": [
            lb["lb_stat"].iloc[0],
            jb.statistic,
            arch[0]
        ],
        "P-value": [
            lb["lb_pvalue"].iloc[0],
            jb.pvalue,
            arch[1]
        ]
    })

    results["Reject H0 (5%)"] = (
        results["P-value"] < 0.05
    )

    return results


# Largest autoregressive root of inflation
def rolling_largest_ar_root(
    series,
    window=120,
    p=1,
    q=0,
    trend="c"
):
    """
    Rolling AR(p) estimation of monthly inflation persistence.

    Parameters
    ----------
    inflation : Time series, i.e monthly inflation (e.g. 100 * log(P_t/P_{t-1}))
    window : Number of monthly observations kept per window
    p : Autoregressive order

    Returns
    -------
    Largest companion eigenvalue modulus,
    AR coefficients, and stationarity indicator.
    """
    series = series.dropna()

    results = []

    for t in range(window, len(series) + 1):

        sample = series.iloc[t-window:t]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                mod = ARIMA(
                    sample,
                    order=(p, 0, q),
                    trend=trend,
                    enforce_stationarity=True,
                    enforce_invertibility=True
                ).fit(
                    method_kwargs={"maxiter": 500}
                )

            converged = mod.mle_retvals.get("converged", False)

            if converged and len(mod.arroots) > 0:
                lar = 1 / np.min(np.abs(mod.arroots))
            else:
                lar = np.nan

            results.append({
                "Date": sample.index[-1],
                "LargestRoot": lar,
                "Converged": converged,
                "AIC": mod.aic,
                "BIC": mod.bic
            })

        except (ValueError, np.linalg.LinAlgError):
            results.append({
                "Date": sample.index[-1],
                "LargestRoot": np.nan,
                "Converged": False,
                "AIC": np.nan,
                "BIC": np.nan
            })

    return pd.DataFrame(results).set_index("Date")


# RMSE between actual and fitted
def rmse(act,fit):
    meanSquaredError = np.nanmean((fit - act) ** 2)
    rmse = np.sqrt(meanSquaredError)
    return rmse

# Projections
def localProjections(Y, X, h=np.arange(61)):
    coefs = []
    Err = []
    IC90 = []
    IC10 = []
    SDPlus = []
    SDMinus = []

    for horz in h:
        y = Y.iloc[:, horz].values

        # OLS 
        model = sm.OLS(y, X.values).fit(
        # cov_type="HAC",
        # cov_kwds={"maxlags": max(1, horz)}
        ).get_robustcov_results(cov_type='HAC', maxlags=max(1, horz))
        coef = model.params

        # Contemporaneous coefficient
        B0 = coef[0]
        seB0 = model.bse[0]

        coefs.append(coef)
        Err.append(model.resid)

        IC10.append(B0 - 1.645 * seB0)
        IC90.append(B0 + 1.645 * seB0)
        SDMinus.append(B0 - 1 * seB0)
        SDPlus.append(B0 + 1 * seB0)
        
    finalCoefs = pd.DataFrame(coefs, columns=X.columns)
    return finalCoefs, IC10, IC90, SDPlus, SDMinus