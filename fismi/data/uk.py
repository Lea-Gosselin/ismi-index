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
    goodCols = ["Title", *df.columns[np.where(df.columns.str.match("CPI INDEX "))[0]]]
    dfTemp = df[goodCols]

    # 2. B) Sorting : only filtered sub-classes indices
    dfTemp.columns=dfTemp.iloc[0,:]
    dfTemp = dfTemp.iloc[1:,:]
    index =  np.where(dfTemp.columns.isin([*dfEcoicop["Indices"], "CDID", headline]))[0]
    dfTemp = dfTemp.iloc[:,index].dropna(subset=[dfTemp.iloc[:, index].columns[3]])
    dfTemp = dfTemp.rename(columns = {headline : "HEADLINE"})
    
    # 2. C) Monthly observations...
    idStart = np.where(dfTemp["CDID"].str.fullmatch(startDate))[0][0]
    dfTemp = dfTemp.iloc[idStart:,:]

    dfTemp = dfTemp.rename(columns={"CDID": "Dates"})
    dfTemp["Dates"] = pd.to_datetime(dfTemp["Dates"], format="%Y %b")
    
    # Final Prices dataframe... Set columns as the match between IPC and weights tickers
    dfPrices = dfTemp.set_index("Dates",drop=True)
    dfPrices = dfPrices.rename(columns=dfEcoicop.set_index("Indices")["Ticker"].to_dict())


    # 3. Collect UK CPI weights (annual)...
    yyyy = pd.Timestamp.today().year

    for year in [yyyy, yyyy - 1]:
        urlWeights = params["weights"]["url"].format(yyyy=year)
        r = requests.get(urlWeights, headers=headers, timeout=30)

        if r.ok:
            print(f"ONS weights successfully downloaded for {year}.")
            break

        print(f"ONS weights unavailable for {year} (HTTP {r.status_code}).")

        if year == yyyy - 1:
            r.raise_for_status()
            raise RuntimeError("Unable to download ONS weights.")

    df = pd.read_excel(BytesIO(r.content),sheetWeights)

    # 3. A) Resize the dataframe
    iRow = np.where(df.iloc[:, 3].notna())[0][0]
    iCol = np.where(df.iloc[7, :].notna())[0][0]
    df = df.iloc[iRow:, iCol:].dropna(subset=[df.columns[3]])

    # 3. B) Cleaning : header = tickers, index = Dates (year)
    dfTemp = df.T.copy()
    dfTemp.iloc[0, 0] = "Dates"
    dfTemp = dfTemp.dropna()
    dfTemp.columns = dfTemp.iloc[0]
    dfTemp = dfTemp.iloc[1:].copy()

    # 3. C) Manage dates : weights revisions in between, we keep the most up-to-date (latest ones)
    dfTemp["Dates"] = [i[:4] for i in dfTemp["Dates"].astype(str).values]
    dfTemp = dfTemp.drop_duplicates(subset="Dates",keep="last")
    dfTemp = dfTemp.loc[:, ~dfTemp.columns.duplicated()]

    dfTemp = dfTemp.set_index(dfTemp.columns[0])
    dfTemp.index = pd.to_datetime(dfTemp.index.astype(str), format="%Y")

    # 3. D) Annual to monthly frequency
    dfW = dfTemp.apply(pd.to_numeric, errors="coerce")
    dfW = dfW.resample("MS").ffill()
    dfW = dfW.drop("CHZQ",axis=1) # Drop 1,000 (headline)
    dfWFinal = dfW.loc[:, dfW.columns.isin(dfEcoicop["Weights"])]
    dfWFinal = dfWFinal.reindex(dfPrices.index).ffill()
    dfWFinal = dfWFinal.rename(columns=dfEcoicop.set_index("Weights")["Ticker"].to_dict())

    # 4. Standardize and normalize
    prices, weights = prepare_data(dfPrices, dfWFinal)
    
    return prices, weights