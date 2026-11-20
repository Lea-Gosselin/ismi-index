import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from scipy.stats import jarque_bera
import warnings

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