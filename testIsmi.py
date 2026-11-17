
from fismi.io import load_input, save_ismi, load_config
from fismi.models import estimate_ismi
from fismi.maths import rmse
import matplotlib.pyplot as plt
from archive.figures import ismiVersusIndex, posVersusNeg, FedStyleChart
import numpy as np
import pandas as pd


for country in ["US"]:

    print(f"\n--- {country} ---")

    config = load_config(country)
    prices, weights = load_input(country)

    allResults = estimate_ismi(
        prices=prices,
        weights=weights,
        config=config
    )
    
    # US... Benchmark with FED data
    if country == "US":
        USIsmi = allResults["results"]
        
        # Benchmark with Fed data
        ShapiroResults = pd.read_excel("archive/ismi_data.xlsx", "k3_AR1")
        # Convert the dates
        ShapiroResults["date"] = pd.to_datetime(ShapiroResults["date"])
        ShapiroResults = ShapiroResults.set_index("date")

        # Compare headline series 
        AllSeries = USIsmi.merge(
            ShapiroResults["ismi_headline_ar1_k3"],
            how="left",
            right_index=True,
            left_index=True
        )

        # Compute RMSE
        RMSE = rmse(AllSeries["ismi_headline_ar1_k3"].values, AllSeries["ISMI"].values)

        fig, ax = plt.subplots(figsize=(20, 6))
        x_axis = AllSeries.index
        ax.plot(
            x_axis,
            AllSeries["ismi_headline_ar1_k3"].values,
            color="darkgrey",
            label="ISMI Officiel",
            linewidth=1,
        )

        ax.plot(
            x_axis,
            AllSeries["ISMI"].values,
            label=f"ISMI Estimé (RMSE = {RMSE*100:.0f} bps)",
            color="royalblue",
            linewidth=1,
        )

        ax.legend()
        ax.set_xlabel("Dates", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.title("Qualité de la réplication - ISM Index", fontsize=14, fontweight="bold")
        plt.show()


        # Components (Positive)
        # Compare headline positive series 
        AllSeries = AllSeries.merge(
            ShapiroResults["ismi_headline_positive_ar1_k3"],
            how="left",
            right_index=True,
            left_index=True
        )

        # Compute RMSE
        RMSE = rmse(AllSeries["ismi_headline_positive_ar1_k3"].values, AllSeries["ISMI_Pos"].values)

        fig, ax = plt.subplots(figsize=(20, 6))
        x_axis = AllSeries.index
        ax.plot(
            x_axis,
            AllSeries["ismi_headline_positive_ar1_k3"].values,
            color="darkgrey",
            label="Part PCE soumis à 3 chocs positifs successifs (Shapiro)",
            linewidth=1,
        )

        ax.plot(
            x_axis,
            AllSeries["ISMI_Pos"].values,
            label=f"Part PCE soumis à 3 chocs positifs successifs (RMSE = {RMSE*100:.0f} pp)",
            color="orange",
            linewidth=1,
        )

        ax.legend()
        ax.set_xlabel("Dates", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.title("Qualité de la réplication - Part PCE soumis à 3 chocs positifs successifs", fontsize=14, fontweight="bold")
        plt.show()

        # Components (Negative)
        # Compare headline positive series 
        AllSeries = AllSeries.merge(
            ShapiroResults["ismi_headline_negative_ar1_k3"],
            how="left",
            right_index=True,
            left_index=True
        )

        # Compute RMSE
        RMSE = rmse(AllSeries["ismi_headline_negative_ar1_k3"].values, AllSeries["ISMI_Neg"].values)

        fig, ax = plt.subplots(figsize=(20, 6))
        x_axis = AllSeries.index
        ax.plot(
            x_axis,
            AllSeries["ismi_headline_negative_ar1_k3"].values,
            color="darkgrey",
            label="Part PCE soumis à 3 chocs négatifs successifs (Shapiro)",
            linewidth=1,
        )

        ax.plot(
            x_axis,
            AllSeries["ISMI_Neg"].values,
            label=f"Part PCE soumis à 3 chocs négatifs successifs (RMSE = {RMSE*100:.0f} pp)",
            color="royalblue",
            linewidth=1,
        )

        ax.legend()
        ax.set_xlabel("Dates", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.title("Qualité de la réplication - Part PCE soumis à 3 chocs négatifs successifs", fontsize=14, fontweight="bold")
        plt.show()

    path = save_ismi(country, allResults["results"])

    print(allResults["results"].tail())
    print("Saved:", path)

    headline = np.log(prices["HEADLINE"]).diff(12)
    headline = headline.reindex(allResults["results"].index)


    # Figure 1
    ismiVersusIndex(allResults["results"], headline*100, index=country)
    posVersusNeg(allResults["results"]["ISMI_Pos"],allResults["results"]["ISMI_Neg"],index=country)
    FedStyleChart(allResults["results"])
    

