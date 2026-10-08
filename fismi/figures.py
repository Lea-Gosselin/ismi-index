import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from fismi import utils
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf

# 1) Aggrégé versus officiel
def plotAggregatedIndex(dfW,dfICP,col="Personal consumption expenditures",niv="4"):
    # Construct 
    # Comparaison des séries agrégées et headline
    aggregated = ((dfW*dfICP).sum(axis=1).to_frame(name="AggIndex"))

    aggregated = aggregated.merge(
        dfICP[col],
        how="left",
        right_index=True,
        left_index=True
    )
    
    # Plot
    xAxis = aggregated.index
    fig, ax = plt.subplots(figsize=(20, 6))

    ax.plot(
        xAxis,
        aggregated["AggIndex"],
        label=f"Moy. pondérée des catégories Niveau {niv}",
        color="royalblue",
        linewidth=1.5,
    )

    ax.plot(
        aggregated.index,
        aggregated[col],
        label=col,
        color="black",
        linewidth=1.5,
        linestyle="--",
    )

    ax.set_ylabel("Indice aggrégé", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    
# 2) ISMI versus reference index
def ismiVersusIndex(ismIndex, headline, index="PCE"):
    # Plot
    fig, ax2 = plt.subplots(figsize=(20, 6))

    # --- Left axis ---
    ax2.plot(
        ismIndex.loc[ismIndex.index, "ISMIndex"],
        label="ISM Index",
        color="royalblue",
        linewidth=1,
    )

    ax2.set_ylabel("ISM Index", fontsize=12)
    ax2.set_ylim(-1, 1)
    ax2.tick_params(axis="y")

    # --- Right axis ---
    ax1 = ax2.twinx()

    ax1.plot(
        headline,
        label=f"{index} Inflation (12-month)",
        color="black",
        linewidth=1,
    )

    ax1.set_xlabel("Dates", fontsize=12)
    ax1.set_ylim(-12, 12)
    ax1.set_ylabel("Inflation (percent)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Title, legend ---
    plt.title(f"Inflation Shock Momentum Index ({index})", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

    plt.tight_layout()
    plt.show()
    
# 3) Positive versus negative contributions
def posVersusNeg(posIsmi,negIsmi,index="PCE"):
    # Positive and negative contributions (5 last years)
    fig, ax = plt.subplots(figsize=(20, 6))
    x_axis = posIsmi.index 
    ax.plot(
        x_axis,
        posIsmi.values, 
        color="orange",
        label="Share of categories with 3 positive residuals in a row",
        linewidth=1,
    )

    ax.plot(
        x_axis,
        negIsmi.values,
        label="Share of categories with 3 negative residuals in a row",
        color="royalblue",
        linewidth=1
    )

    ax.legend()
    ax.set_xlabel("Dates", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.title(f"Components of ISM Index ({index})", fontsize=14, fontweight="bold")
    plt.show()
    

# 4) FED style chart

def FedStyleChart(ismIndex):
    ismi = ismIndex["ISMIndex"].copy()

    # Dates mensuelles et 24 dernières observations disponibles
    ismi.index = pd.to_datetime(ismi.index)
    ismi = ismi.sort_index().dropna().tail(24).round(3)

    x = np.arange(len(ismi))
    colors = np.where(ismi.ge(0), "#315D87", "#85CAE2")

    fig, ax = plt.subplots(figsize=(16, 8), facecolor="#FFFEFC")
    ax.set_facecolor("#FFFEFC")

    ax.bar(x, ismi, width=0.8, color=colors, edgecolor="none")

    ax.axhline(0, color="#D5DADF", linewidth=1)
    ax.grid(axis="y", color="#E4E8EB", linewidth=1)
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ismi.index.strftime("%Y-%m"),
        rotation=-90,
        fontsize=12,
        color="#44484C",
    )
    ax.set_xlim(-0.5, len(ismi) - 0.5)
    ax.set_ylabel("Index", fontsize=16, color="#44484C")
    ax.tick_params(axis="y", labelsize=12, colors="#44484C", length=0)
    ax.tick_params(axis="x", length=0)

    ax.set_title(
        "Figure 1: ISMI for past 24 months",
        loc="left",
        fontsize=21,
        fontweight="bold",
        pad=35,
    )

    ax.legend(
        handles=[
            Patch(facecolor="#315D87", label="Above zero"),
            Patch(facecolor="#85CAE2", label="Below zero"),
        ],
        loc="best",
        bbox_to_anchor=(1, 1.14),
        ncol=2,
        frameon=False,
        fontsize=14,
    )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#D5DADF")

    plt.tight_layout()
    plt.show()
    

# 5) Appendix chart A1

def figA1Paper(headline, PCE, index="PCE"):
    # Plot
    fig, ax1 = plt.subplots(figsize=(20, 6))

    # --- Left axis (HICP inflation - 12-month) ---
    ax1.plot(
        headline,
        label=f"{index} Inflation (12-month)",
        color="darkgrey",
        linewidth=1,
    )

    ax1.set_xlabel("Dates", fontsize=12)
    ax1.set_ylabel(f"{index} Inflation")
    ax1.set_ylim(-4, 12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Right axis (AR coefficients) ---
    ax2 = ax1.twinx()

    ax2.plot(
        PCE["A"],
        label=r'$\hat \alpha$ coefficient',
        color="darkgreen",
        linewidth=1,
    )

    ax2.plot(
        PCE["Rho"],
        label=r'$\hat \rho$ coefficient',
        color="darkblue",
        linewidth=1,
    )

    ax2.set_ylabel("AR(1) coefficients value", fontsize=12)
    ax2.set_ylim(-0.4, 1.2)
    ax2.tick_params(axis="y")

    # --- Title, legend ---
    plt.title(f"Rolling window estimates of AR(1) coefficients for Headline {index} Inflation", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=11)

    plt.tight_layout()
    plt.show()
    
# IR from local projections
def plotIresponses(R, label=r"Impulse response ($ISMI_t$)", color="darkgrey"):
    fig, ax = plt.subplots(figsize=(20/3, 6))

    # IRF
    ax.plot(
        R.index,
        R["IRF"],
        linewidth=2,
        color=color,
        label=label
    )

    # 90% confidence interval
    ax.fill_between(
        R.index,
        R["IC90"],
        R["IC10"],
        color=color,
        alpha=0.2,
        label="90% CI"
    )

    # 95% confidence interval
    ax.fill_between(
        R.index,
        R["SD+"],
        R["SD-"],
        color=color,
        alpha=0.1,
        label="±1 écart-type"
    )

    # Zero line
    ax.axhline(
        y=0,
        linewidth=1,
        color="black",
        linestyle="--"
    )

    # Labels
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Responses (% points)")
    ax.set_title("Impulse responses")

    # Horizon
    ax.set_xlim(0, 60)

    ax.legend(frameon=False)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    
def plotFourFrames(results, xAxis, lag=12):
    # Inflation fit and diagnostic on residuals
    fitted = results["FittedY"]
    pi = results["Y"]
    MSE = results["Err"] # Attention, erreurs sur la spécification centrée réduite
    rmse = utils.rmse(pi, fitted)

    # Figure...
    fig, axes = plt.subplots(
        2, 2,
        figsize=(20, 10)
    )

    # Figure 1 : Observed versus fitted inflation
    ax = axes[0, 0]

    ax.plot(
        xAxis,
        pi * 100,
        label=f"Observed {lag}-month inflation",
        color="grey",
        linewidth=1
    )

    ax.plot(
        xAxis,
        fitted * 100,
        label=f"Fitted {lag}-month inflation (RMSE = {rmse*100:.2f} bps)",
        color="darkgreen",
        linewidth=1
    )

    ax.set_title(f"Observed and fitted Headline {lag}-month Inflation")
    ax.set_ylabel("Inflation (%)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)


    # Figure 2 : Residuals
    ax = axes[0, 1]

    ax.plot(
        xAxis,
        MSE * 100,
        color="darkgrey",
        linewidth=1,
        label=f"Measurement errors"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_title("Measurement errors")
    ax.set_ylabel("Error (bps)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)


    # Figure 3 : ACF
    nlags = 36
    plot_acf(
        MSE[1:],
        lags=36,
        ax=axes[1, 0],
        zero=False
    )

    axes[1, 0].set_title("ACF — Measurement errors")
    axes[1, 0].set_xlabel("Lags")
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)


    # Figure 4 : PACF
    plot_pacf(
        MSE[1:],
        lags=nlags,
        ax=axes[1, 1],
        zero=False,
        method="ywm"
    )

    axes[1, 1].set_title("PACF — Measurement errors")
    axes[1, 1].set_xlabel("Lags")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)


    # Formatting...
    fig.suptitle(
        "Kalman Filter Results: Inflation Fit and Residual Diagnostics",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()
    

def plotHeadlineRho(results, xAxis, lag):
    # Rolling persistance coefficient since 1959 against headline PCE
    Rho = results["Rho"]
    pi = results["Y"]
    _, ax1 = plt.subplots(figsize=(20, 6))

    # --- Left axis (HICP inflation - 12-month) ---
    ax1.plot(
        xAxis,
        pi*100,
        label=f"Headline {lag}-month inflation",
        color="black",
        linewidth=1
    )

    ax1.set_xlabel("Dates", fontsize=12)
    ax1.set_ylabel(f"Headline {lag}-month inflation (%)")
    ax1.set_ylim(-4, 12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- Right axis (AR coefficients) ---
    ax2 = ax1.twinx()

    ax2.plot(
        xAxis,
        Rho,
        label=r'$\hat \rho_t$ coefficient',
        color="darkgrey",
        linewidth=1
    )

    ax2.set_ylabel("AR(1) coefficients value", fontsize=12)
    # ax2.set_ylim(0.8, 1.2)
    ax2.tick_params(axis="y")

    # --- Title, legend ---
    plt.title(f"KF estimates of AR(1) coefficients for Headline {lag}-month Inflation", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=11)

    plt.tight_layout()
    plt.show()
    return ax1, ax2