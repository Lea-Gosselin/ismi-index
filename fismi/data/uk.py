import pandas as pd
import requests
from io import BytesIO
import numpy as np
from pathlib import Path
import os
from fismi.data.process import prepare_data
from datetime import date


# Path
ROOT = Path(__file__).resolve().parents[2]

def collect_uk(config):
    """
    Collect and prepare united kingdom CPI indices and weights.

    Returns
    -------
    prices : pd.DataFrame, monthly price indices including aggregate price index (HEADLINE)
    weights : pd.DataFrame, monthly normalized component weights
    """
    params = config["data"] # fismi.io, load_config
    urlEcoicop = params["ecoicop"]["url"]
    sheetEcoicop = params["ecoicop"]["sheet"]
    urlPrices = params["price_url"]
    sheetWeights = params["weights"]["sheet"]
    startDate = params["startDate"]
    headline = params["headline_ticker"]

    # Same for all
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }

    # 1. Collect ECOICOP classification for UK CPI...
    r = requests.get(urlEcoicop, headers=headers, timeout=30)
    r.raise_for_status()

    df = pd.read_excel(BytesIO(r.content), sheetEcoicop)
    iRow = np.where(df.iloc[:, 4].notna())[0][0]-1
    iCol = [0,1,4]
    dfTemp = df.copy().iloc[iRow:, iCol]
    dfTemp.iloc[0,2] = "ECOICOP"
    dfTemp.columns = dfTemp.iloc[0]
    dfTemp = dfTemp.iloc[1:].copy()
    dfTemp = dfTemp.dropna().reset_index(drop=True)

    # 1. A) Create groups, classes and sub-classes (the latter is the most granular level of disaggregation)
    patterns = {
        "groups":     r"^\d{2}\s",
        "classes":    r"^\d{2}\.\d(?:\s|/)",
        "subclasses": r"^\d{2}\.\d\.\d"
    }

    for level, pattern in patterns.items():
        mask = dfTemp["ECOICOP"].astype(str).str.match(pattern)
        dfTemp[level] = dfTemp["ECOICOP"].str.split().str[0].where(mask)

    for c in dfTemp["classes"].dropna().unique():
        subset = dfTemp.iloc[np.where(dfTemp["ECOICOP"].str.match(c))[0],:].dropna(subset="subclasses")
        if subset.shape[0] == 0:
            dfTemp.loc[dfTemp["classes"]==c, "subclasses"] = c
    dfEcoicop = dfTemp.dropna(subset="subclasses")
    dfEcoicop["Ticker"] = dfEcoicop["Indices"] + "-" + dfEcoicop["Weights"]


    # 2. Collect UK CPI indices as of Start Date (.yaml)...
    r = requests.get(urlPrices, headers=headers, timeout=30)
    r.raise_for_status()
    df = pd.read_excel(BytesIO(r.content))

    # 2. A) Sorting : only CPI INDEX series
    priceCols = ["Title", *df.columns[np.where(df.columns.str.match("CPI INDEX "))[0]]]
    weightCols = ["Title", *df.columns[np.where(df.columns.str.match("CPI WEIGHTS "))[0]]]
    
    dfPrice = df[priceCols]

    # 2. B) Sorting : only filtered sub-classes indices
    dfPrice.columns=dfPrice.iloc[0,:]
    dfPrice = dfPrice.iloc[1:,:]
    index =  np.where(dfPrice.columns.isin([*dfEcoicop["Indices"], "CDID", headline]))[0]
    dfPrice = dfPrice.iloc[:,index].dropna(subset=[dfPrice.iloc[:, index].columns[3]])
    dfPrice = dfPrice.rename(columns = {headline : "HEADLINE"})
    
    # 2. C) Monthly observations...
    idStart = np.where(dfPrice["CDID"].str.fullmatch(startDate))[0][0]
    dfPrice = dfPrice.iloc[idStart:,:]

    dfPrice = dfPrice.rename(columns={"CDID": "Dates"})
    dfPrice["Dates"] = pd.to_datetime(dfPrice["Dates"], format="%Y %b")
    
    # Final Prices dataframe... Set columns as the match between IPC and weights tickers
    dfPrices = dfPrice.set_index("Dates",drop=True)
    dfPrices = dfPrices.rename(columns=dfEcoicop.set_index("Indices")["Ticker"].to_dict())


    # 3. Collect UK CPI weights (annual)...    
    # 3. A) Sorting : only CPI WEIGHTS series
    dfWeights = df[weightCols]

    # 3. B) Sorting : only filtered sub-classes weights
    dfWeights.columns=dfWeights.iloc[0,:]
    dfWeights = dfWeights.iloc[1:,:]
    index =  np.where(dfWeights.columns.isin([*dfEcoicop["Weights"], "CDID"]))[0]
    dfWeights = dfWeights.iloc[:,index].dropna(subset=[dfWeights.iloc[:, index].columns[3]])
    
    # 3. C) Monthly observations...
    idStart = np.where(dfWeights["CDID"].str.fullmatch(startDate[:4]))[0][0]
    dfWeights = dfWeights.iloc[idStart:,:].ffill().dropna()

    dfWeights = dfWeights.rename(columns={"CDID": "Dates"})
    dfWeights["Dates"] = pd.to_datetime(dfWeights["Dates"].astype(str), format="%Y")
    
    # Final Prices dataframe... Set columns as the match between IPC and weights tickers
    dfWeights = dfWeights.set_index("Dates",drop=True)
    dfWeights = dfWeights.rename(columns=dfEcoicop.set_index("Weights")["Ticker"].to_dict())
    dfWFinal = dfWeights.reindex(dfPrices.index).ffill()

    # 4. Standardize and normalize
    prices, weights = prepare_data(dfPrices, dfWFinal)
    
    return prices, weights