from fismi.io import load_input, save_persistence, load_config
from fismi.models import estimate_persistence
from fismi.maths import rmse
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from fismi.maths import residual_diagnostics, plot_residual_diagnostics
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from scipy.stats import norm
import statsmodels.api as sm 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

# ["US", "EA", "UK", "JP"]

for country in ["US", "EA", "UK", "JP"]:

    print(f"\n--- {country} ---")

    config = load_config(country)
    prices, weights = load_input(country)
    startDate = config["persistence"]["startDate"]
    p = config["persistence"]["p_lag"]
    q = config["persistence"]["q_lag"]
    inf_lag = config["persistence"]["inflation_lag"]
    pi = np.log(prices.loc[prices.index >= startDate, "HEADLINE"]).diff(inf_lag).dropna()
    model_name = config['persistence']['model']
        
    ## 1) Estimate an ARMA(p,q) on the full sample
    model = ARIMA(
        pi,
        order=(p, 0, q),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True
    ).fit()

    # ACF and PACF
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    plot_acf(pi, lags=36, ax=axes[0, 0])
    axes[0, 0].set_title("ACF")

    plot_pacf(pi, lags=36, ax=axes[0, 1], method="ywm")
    axes[0, 1].set_title("PACF") 

    plot_acf(model.resid, lags=36, ax=axes[1, 0])
    axes[1, 0].set_title("Residuals (ACF)")

    plot_pacf(model.resid, lags=36, ax=axes[1, 1], method="ywm")
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
    
    print(f"------- Estimation success -------")
    print(f"Message : {persistence['optimization']['success']}")

    print("----- Log-likelihood and IC -----")    
    print(f"ARMA({p},{q}) full data, constant parameters : NLL = {model.llf}, AIC = {model.aic}, BIC = {model.bic}")
    print(f"KF and MLE, varying persistence : {-persistence['optimization']['fun']}")
    
    print("----- Initial parameters -----")
    initParams = config['persistence']['initial_params']
    print(f"Mu bar = {initParams[0]}")
    print(f"Rho bar = {initParams[1]}")
    for i in range(2,2+q):
        print(f"MA({i-1}) coefficient = {initParams[i]}")
    print(f"Std. inflation innovations = {initParams[-2]}")
    print(f"Std. state innovations = {initParams[-1]}")
    
    print("----- Estimated parameters -----")
    estParams = persistence["parameters"]
    print(f"Mu bar = {estParams[0]}")
    print(f"Rho bar = {estParams[1]}")
    for i in range(2,2+q):
        print(f"MA({i-1}) coefficient = {estParams[i]}")
    print(f"Std. inflation innovations = {estParams[-2]}")
    print(f"Std. state innovations = {estParams[-1]}")

    print("Saved:", path)


    # ============================================================
    # 0. PREPARE DATA
    # ============================================================
    df = persistence["results"].copy()

    # Actual inflation (log differences, in %)
    headline = prices["HEADLINE"].sort_index()
    mom = 100 * np.log(headline).diff(1)
    yoy = 100 * np.log(headline).diff(12)

    # Align with Kalman estimation dates
    dates = df.index
    mom = mom.reindex(dates)
    yoy = yoy.reindex(dates)
    
    # RMSE MoM
    forecastInflation = df['FittedInflation'].values*100
    rmseMoM = rmse(mom.values/100, forecastInflation/100)


    # ============================================================
    # 2. PERSISTENCE
    # ============================================================
    rhoFiltered = df["RhoFiltered"]
    rhoSmoothed = df["RhoSmoothed"]

    # 90% interval for smoothed persistence
    z90 = norm.ppf(0.95) # Oui, vraie valeur 1.64

    varRho = np.asarray(
        df['VarStateSmoothed']
    )

    z90 = norm.ppf(0.95)

    stdRho = np.sqrt(
        np.maximum(varRho, 0)
    )

    rhoLower = rhoSmoothed - z90 * stdRho
    rhoUpper = rhoSmoothed + z90 * stdRho

    # ============================================================
    # 3. DIAGNOSIS
    # ============================================================
    Kt = df["KalmanGain"]
    Err = df["Residuals"]
    stdY = np.sqrt(df["forecastVarY"])
    innovationErr = Err / stdY  # Standardized one-step-ahead innovations


    # ============================================================
    # RESEARCH FIGURE STYLE
    # ============================================================

    RED = "#F06449"
    BLUE = "#2255E8"
    ORANGE = "#F4A340"
    TEAL = "#27A59A"
    INK = "#263445"
    MUTED = "#9AA8B5"
    GRID = "#DFE5EA"
    BAND = "#DCE8F3"


    def save_research_figure(fig, name):
        fig.savefig(
            f"{name}.pdf",
            bbox_inches="tight"
        )
        fig.savefig(
            f"{name}.png",
            dpi=300,
            bbox_inches="tight"
        )

    # ============================================================
    # 1. MOM INFLATION AND FORECAST ERRORS
    # ============================================================

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 8),
        sharex=True,
        constrained_layout=True
    )

    # Forecast errors (percentage points)
    errorMoM = np.asarray(mom.reindex(dates)) - np.asarray(forecastInflation)

    # Panel 1: MoM inflation
    axes[0].plot(
        dates, mom.reindex(dates),
        color=MUTED, lw=1.8,
        label="Observed MoM inflation"
    )

    axes[0].plot(
        dates, forecastInflation,
        color=BLUE, lw=1.5,
        linestyle="--",
        label=f"Reconstructed forecast (RMSE = {rmseMoM:.3f} pp)"
    )

    axes[0].set_title(f"{country} inflation (monthly)")
    axes[0].set_ylabel("Inflation (%)")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)

    # Panel 2: Forecast errors
    axes[1].plot(
        dates, errorMoM,
        color=ORANGE, lw=1.3,
        label="Monthly inflation forecast errors"
    )

    axes[1].fill_between(
        dates, errorMoM, 0,
        color=ORANGE, alpha=0.12
    )

    axes[1].axhline(
        0, color="black", lw=0.8, alpha=0.5
    )

    axes[1].set_title("Monthly inflation forecast errors")
    axes[1].set_ylabel("Error (pp)")
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

    plt.show()


    # ============================================================
    # 3. TIME-VARYING INFLATION PERSISTENCE
    # ============================================================

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 8),
        sharex=True,
        constrained_layout=True
    )

    # Empirical mean persistence
    rho_mean = np.nanmean(rhoSmoothed)

    # Panel 1: Filtered and smoothed persistence
    axes[0].fill_between(
        dates, rhoLower, rhoUpper,
        color=BAND, alpha=0.75,
        linewidth=0,
        label="90% smoothed interval"
    )

    axes[0].plot(
        dates, rhoFiltered,
        color=ORANGE, lw=1.3,
        alpha=0.8,
        label=r"Filtered $\rho_{t|t}$"
    )

    axes[0].plot(
        dates, rhoSmoothed,
        color=BLUE, lw=2,
        label=r"Smoothed $\rho_{t|T}$"
    )

    axes[0].axhline(
        rho_mean,
        color=TEAL, lw=1.5,
        linestyle="--",
        label=rf"Sample mean $\bar{{\rho}}$ = {rho_mean:.3f}"
    )

    axes[0].axhline(
        1, color="darkred", lw=1.2,
        linestyle=":",
        label="Unit persistence"
    )

    axes[0].set_title(f"{country} inflation: time-varying persistence")
    axes[0].set_ylabel(r"Persistence ($\rho_t$)")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].grid(alpha=0.25)

    # Panel 2: Smoothed persistence uncertainty
    axes[1].plot(
        dates, rhoSmoothed,
        color=BLUE, lw=2,
        label=r"Smoothed $\rho_{t|T}$"
    )

    axes[1].fill_between(
        dates, rhoLower, rhoUpper,
        color=BAND, alpha=0.75,
        linewidth=0,
        label="90% confidence interval"
    )

    axes[1].axhline(
        rho_mean,
        color=TEAL, lw=1.5,
        linestyle="--",
        label=rf"Sample mean $\bar{{\rho}}$ = {rho_mean:.3f}"
    )

    axes[1].set_title("Smoothed persistence and estimation uncertainty")
    axes[1].set_ylabel(r"Persistence ($\rho_t$)")
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

    plt.show()


    # ============================================================
    # 4. KALMAN GAIN AND FORECAST ERRORS
    # ============================================================

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

    plt.show()


    # Diagnostiques sur les résidus ...
    results = residual_diagnostics(
        z=innovationErr,
        p=p,
        q=q
    )
        
    plot_residual_diagnostics(
        z=innovationErr,
        name=f"{country} : ARMA({p},{q})",
        lags=36
    )
    
    dfDiagnostics = pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["TVP-AR(1)"]
    )
    print(dfDiagnostics.round(4))