import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import pandas as pd
from pathlib import Path

# OLS to Latex
# AR(1) à coefficient constant
def olsToLatex(model,caption="Constant-coefficient AR(1) estimation",save=True):
    # Coefficient table
    results = pd.DataFrame({
        "Coefficient": model.params,
        "Std. Error": model.bse,
        "$t$-Stat.": model.tvalues,
        "$P > |t|$": model.pvalues,
        "$0.025$": model.conf_int()[0,:],
        "$0.975$": model.conf_int()[1,:]
    }, index=[r"Constant ($\hat \mu$)", r"Persistence ($\hat \rho$)"])

    # Formatting
    latex = results.to_latex(
        escape=False,
        float_format="%.3f",
        column_format="lrrrrrr",
        caption=caption,
        label="tab:ar1_ols"
    )

    # Add model statistics before bottomrule
    stats = (
        r"\midrule" + "\n"
        + rf"Observations & \multicolumn{{1}}{{r}}{{{int(model.nobs)}}} \\" + "\n"
        + rf"$R^2$ & \multicolumn{{1}}{{r}}{{{model.rsquared:.3f}}} \\" + "\n"
        + rf"Adjusted $R^2$ & \multicolumn{{1}}{{r}}{{{model.rsquared_adj:.3f}}} \\" + "\n"
        + rf"Log-likelihood & \multicolumn{{1}}{{r}}{{{model.llf:.2f}}} \\" + "\n"
        + rf"AIC & \multicolumn{{1}}{{r}}{{{model.aic:.2f}}} \\" + "\n"
        + rf"BIC & \multicolumn{{1}}{{r}}{{{model.bic:.2f}}} \\" + "\n"
        # + rf"DW & \multicolumn{{1}}{{r}}{{{model.dw:.2f}}} \\" + "\n"
        # + rf"BIC & \multicolumn{{1}}{{r}}{{{model.bic:.2f}}} \\" + "\n"
    )
    latexBis = latex.replace(r"\bottomrule", stats + "\n" + r"\bottomrule")

    if save:        
        # Save
        with open("Output/AR1_OLS.tex", "w", encoding="utf-8") as f:
            f.write(latexBis)

    return latexBis



# Plot consumption shares
# plot bars in stack manner
def plotConsumptionShares(
    dfW,
    yyyy=2026,
    title="Composition de l'IPCH par division ECOICOP 2"
):

    COLORS = {
            "positive": "#E39F00D3",
            "negative": "cornflowerblue",
            "ismi": "darkgrey",
            "headline": "black",
            "grid": "#CBD5E1",
        }

    # Annual weights
    dfW_annuel = dfW.groupby(dfW.index.year).first()
    dfW_annuel.index.name = "Year"

    # Select year and sort by weight
    weights = (
        dfW_annuel.loc[yyyy]
        .dropna()
        .sort_values(ascending=False)
    )

    # Normalize weights
    weights = weights / weights.sum()

    # Color palette: orange, blue, grey tones
    cmap = LinearSegmentedColormap.from_list(
        "custom_palette",
        [
            COLORS["positive"],
            "#F4D38A",
            "#DCE6F5",
            COLORS["negative"],
            "#486A9B",
            COLORS["ismi"],
            "#B8BDC5"
        ],
        N=len(weights)
    )

    colors = cmap(np.linspace(0, 1, len(weights)))

    # Figure
    fig, ax = plt.subplots(figsize=(12, 8))

    wedges, texts, autotexts = ax.pie(
        weights,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={
            "width": 0.38,
            "edgecolor": "white",
            "linewidth": 1.5
        },
        autopct=lambda p: f"{p:.0f}%" if p >= 2 else "",
        pctdistance=0.81,
        textprops={
            "fontsize": 11,
            "color": "black",
            "fontweight": "bold"
        }
    )

    # Improve percentage contrast
    for wedge, autotext in zip(wedges, autotexts):
        r, g, b, a = wedge.get_facecolor()
        luminance = 0.2126*r + 0.7152*g + 0.0722*b

        autotext.set_color(
            "white" if luminance < 0.50 else "black"
        )

    # Center text
    ax.text(
        0, 0.08,
        str(yyyy),
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
        color=COLORS["headline"]
    )

    ax.text(
        0, -0.12,
        "Consumption\nexpenditure shares",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["ismi"]
    )

    # Legend
    legend_labels = []

    for col, weight in weights.items():
        try:
            name = col.split("-", 1)[1].strip()
        except (AttributeError, IndexError):
            name = str(col)

        legend_labels.append(
            f"{name} ({weight*100:.1f}%)"
        )

    ax.legend(
    wedges,
    legend_labels,
    title="Categories",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
    prop={"size": 11},
    title_fontproperties={"size": 12, "weight": "bold"},
    frameon=False,
    labelspacing=1.0
)

    # ax.set_title(
    #     title,
    #     fontsize=16,
    #     pad=25
    # )

    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()