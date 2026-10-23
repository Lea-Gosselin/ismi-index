### Query, collect and process EA data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# From fIsmi toolbox
from fismi.data import query as sdx

# Settings
LvlPicked = "3"
headlineTicker = "HICP.M.U2.N.000000.4D0.INX"
incompleteTickers = ["022000", "133090", "011370"]

### ECB Website : Collect HICP indexes and Weights
## HICP = Database
## M    = Frequency (monthly)
## U2   = EA changing composition
## Y    = Working days and seasonally adjusted series
## 4D0  = European Commission (Eurostat)
## 4F0  = ECB (SDW)
## INX  = Index

urlPrices = "https://data-api.ecb.europa.eu/service/data/HICP/M.U2.N..4D0.INX"
dfIdx, tempIdx = sdx.getEcbData(urlPrices, colName="KEY", params=None)

urlWeights = "https://data-api.ecb.europa.eu/service/data/HICP/A.U2.N...INW"
dfW, tempW = sdx.getEcbData(urlWeights, colName="KEY", params=None)

# Get ECOICOP sub-keys, manage dates, resize dataframe
uniKeys = tempIdx.KEY.unique()
unEcoicop = [[i, i.split(".")[4]] for i in uniKeys]
temp = pd.DataFrame(unEcoicop, columns=["KEY", "ecoicop"])
tempIdx = pd.merge(tempIdx, temp, how="left", on="KEY")
tempIdx["ICP_ITEM"] = tempIdx["ecoicop"].values
tempIdx['TIME_PERIOD'] = pd.to_datetime(tempIdx['TIME_PERIOD'].astype(str), format='%Y-%m')
tempIdx = tempIdx.rename(columns={"OBS_VALUE":"INX"})
dfIdx = tempIdx[["KEY", "TITLE", "ICP_ITEM", "TIME_PERIOD", "INX"]]

# Merge Price indexes with their Weights
dFinal = pd.merge(
    dfIdx, dfW,
    how="left",
    left_on=["TIME_PERIOD", "ICP_ITEM"],
    right_on=["TIME_PERIOD", "ICP_ITEM"]
)

# Isolate atypical ECOICOP codes and create divisions, groups, classes, sub-classes and position consistently with ECOICOP classification
ecoicop = pd.Series(dFinal['ICP_ITEM'].unique())
atypicalCodes = ecoicop[~ecoicop.str.match(r'^\d{6}$')]
h = pd.DataFrame({'ICP_ITEM': ecoicop})
h['div']        = h['ICP_ITEM'].str[:2] + '0000'
h['group']      = h['ICP_ITEM'].str[:3] + '000'
h['class']      = h['ICP_ITEM'].str[:4] + '00'
h['ssclass']    = h['ICP_ITEM'].str[:5] + '0'
h['pos']        = h['ICP_ITEM']
h['Level']      = np.select(
    [h['ICP_ITEM'] == h['div'],
     h['ICP_ITEM'] == h['group'],
     h['ICP_ITEM'] == h['class'],
     h['ICP_ITEM'] == h['ssclass']],
    ['1', '2', '3', '4'],
    default='5' # Position
)

# Headline Inflation: Set to the lowest level (0)
dFinal = dFinal.merge(h, how="left", on="ICP_ITEM")
dFinal.loc[dFinal.KEY == headlineTicker, "Level"] = '0'   # Headline

# Atypical codes: Set to "Other" level (aggregates like food inflation, services, xTobacco...)
dFinal.loc[dFinal.ICP_ITEM.isin(atypicalCodes), "Level"] = "Other"

# Architecture ECOICOIP as made available by the ECB 
dFinal[["KEY", "TITLE", "ICP_ITEM", "div", "group", "class", "ssclass", "pos", "Level"]].drop_duplicates()

# Remove series were mainly NaN (and low weight)
cleanDf = dFinal[~dFinal["ICP_ITEM"].isin(incompleteTickers)]


### Pick the desired level of disaggregation...
# Weights (normed)
wNiv = cleanDf[cleanDf.Level == LvlPicked].pivot(columns="ICP_ITEM", values="WEIGHTS", index="TIME_PERIOD").ffill(inplace=True)
wNiv = wNiv.set_index("TIME_PERIOD",drop=True)
wNorm = wNiv.div(wNiv.sum(axis=1, skipna=True), axis=0)

# Index
Niv0  = cleanDf.loc[cleanDf["ICP_ITEM"]=="000000", ["TIME_PERIOD", "INX"]]
ixNiv = cleanDf[cleanDf.Level == LvlPicked].pivot(columns="ICP_ITEM", values="INX", index="TIME_PERIOD").ffill(inplace=True)
ixNiv = ixNiv.set_index("TIME_PERIOD",drop=True)
ixNiv = ixNiv.merge(
    Niv0,
    how="left",
    right_index=True,
    left_index=True
)

### Save the datasets
ixNiv.to_parquet(f"input/EArawItemsLvl{LvlPicked}.parquet")
wNorm.to_parquet(f"input/EArawWeightsLvl{LvlPicked}.parquet")