import numpy as np
import matplotlib.pyplot as plt

# RMSE between actual and fitted
def rmse(act,fit):
    meanSquaredError = ((fit - act) ** 2).mean()
    rmse = np.sqrt(meanSquaredError)
    return rmse


# Plot consumption shares
# plot bars in stack manner
def plotConsumptionShares(dfW, colors="Purples", yyyy=2026, title="Composition de l'IPCH par division ECOICOP 2"):
    dfW_annuel = dfW.groupby(dfW.index.year).first()
    dfW_annuel.index.name = "Year"

    fig, ax = plt.subplots(figsize=(12, 8))
    rowYear = dfW_annuel.loc[yyyy].sort_values(ascending=True)
    rowYear.to_frame().T.plot(kind="bar", stacked=True, ax=ax, width=0.5, colormap=colors)

    ax.set_ylabel(f"Poids au 01-01-{yyyy}")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.set_xticklabels([str(yyyy)], rotation=0)
    ax.set_ylim(0, 1.02)

    # --- Couleurs inversées pour les étiquettes... ---
    cmap_inverse = plt.get_cmap(colors+"_r")
    n_bars = len(rowYear)
    text_colors = cmap_inverse(np.linspace(0.2, 0.9, n_bars))

    # On ajoute 'idx' pour associer la bonne couleur inversée à chaque segment
    for idx, (container, col) in enumerate(zip(ax.containers, rowYear.index)):
        val = rowYear[col]
        label = f"{val*100:.0f}%"
        
        # On applique la couleur inversée correspondante via le paramètre 'color'
        ax.bar_label(
            container, 
            labels=[label], 
            label_type="center", 
            fontsize=8, 
            color=text_colors[idx], 
            fontweight="bold"
        )

    try:
        legend_labels = [f"{col.split('-')[1]} ({rowYear[col]*100:.0f}%)" for col in rowYear.index]
    except:
        legend_labels = [f"{col} ({rowYear[col]*100:.0f}%)" for col in rowYear.index]
    ax.legend(legend_labels, title="Division", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.show()