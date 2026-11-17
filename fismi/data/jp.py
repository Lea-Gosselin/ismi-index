
from pathlib import Path
import numpy as np
import pandas as pd
import os
from fismi.data.process import prepare_data
from fismi.data import query
from dotenv import load_dotenv
import matlab.engine
import requests
from io import BytesIO

# Path
ROOT = Path(__file__).resolve().parents[2]

# Load private environment variables (estat)
load_dotenv(ROOT / ".env")

def collect_jp(config):
    """
    Collect and prepare Japan CPI indices and weights.

    Returns
    -------
    prices : pd.DataFrame, monthly price indices including aggregate price index (HEADLINE)
    weights : pd.DataFrame, monthly normalized component weights
    """

    params = config["data"] # fismi.io, load_config
    csvUrl = params["price_url"]
    fileName = params["historical_weights"]
    mapping = params["mapping_coicop"]
    headline_series = params["headline_series"]
    
    # 1. Upload historical weights since 2000 (saved earlier)
    dfW = pd.read_excel(fileName)
    weights = dfW.set_index("Date")

    # 2. Get CPI price indexes per items from each reference databases
    r = requests.get(csvUrl, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(BytesIO(r.content),encoding='latin-1')
    colNames = df.iloc[1,1:].values # Item codes
    df = df.iloc[5:,:] 

    # 2.1 Construct dates
    dates = df.iloc[:,0]
    df = df.rename(columns={
        dates.name:"Date"
    })

    df["Date"] = pd.to_datetime(
        dates,
        format="%Y%m",
        errors="coerce"
    )

    # 2.2 Set date as index
    df = df.set_index("Date", drop=True)

    # 2.3 Dates as numeric and coerce errors
    df = df.apply(pd.to_numeric, errors = 'coerce')

    # 2.4 Get only COICOP columns
    df.columns = colNames

    # 3. Resize dataframes on weights dates (2000-01)
    df = df[df.index >= weights.index[0]]

    # 4. Map dataframes on common columns
    # Collect at the coicop level, correspondance between coicop and item_code
    # Historical mapping
    mappingHistorical = pd.read_excel(mapping, dtype={
        "item_code":str,
        "coicop_historical":str}).set_index("item_code")["coicop_historical"]

    # Appliquer à toutes les bases historiques COICOP 2025 et 2020 (historique, a fait partie de COICOP)
    common_cols = df.columns.intersection(mappingHistorical.index)

    # Select relevant items in prices, weights
    prices = df[common_cols]
    prices = prices.rename(columns=mappingHistorical) # Rename by ecoicop groups
    
    weights = weights[common_cols]
    weights = weights.rename(columns=mappingHistorical)
    weights = weights.reindex(prices.index).ffill() # Rename by ecoicop groups (prior to grouping)

    # 5. Weighted average of items by COICOP groups
    pricesWeighted = prices*weights
    
    # 6. Weighted average of items per COICOP 
    weights = weights.T.groupby(level=0).sum().T
    prices = (pricesWeighted.T.groupby(level=0).sum().T) / weights # Get index (not weighted) 53 columns
    
    # 5. Add headline CPI to prices
    headline = df[headline_series]
    prices.insert(0, "HEADLINE",headline)

    # 6. Apply the seasonal adjustment filter
    eng = matlab.engine.start_matlab()

    # prices : pandas DataFrame avec TIME_PERIOD et les indices de prix
    dates = prices.index.strftime("%Y-%m-%d").tolist()
    cols = prices.columns.tolist()
    values = matlab.double(
        prices.to_numpy(dtype=float).tolist()
    )

    # Call MATLAB
    eng.addpath(str(ROOT / "fismi" / "data"), nargout=0)
    eng.addpath(str(ROOT / "x13Toolbox" / "seas"), nargout=0)
    result = eng.seasonalAdjustment(dates, values, nargout=1)

    # Convert back to pandas
    prices_sa = pd.DataFrame(
        np.asarray(result),
        columns=cols
    )
    eng.quit()
    
    # Get seasonally-adjusted HICP from SDW directly 
    prices_sa.index = prices.index

    # 7. Standardize and normalize
    prices, weights = prepare_data(prices_sa, weights)
    
    return prices, weights


# ####################################################
# # Les poids - Une seule fois pour l'historique
# # Poids fixes du CPI national — fichiers officiels
# URLS = {
#     2000: "https://www.stat.go.jp/data/cpi/2005/zuhyou/hinmoku2.xls",
#     2005: "https://www.stat.go.jp/data/cpi/2005/zuhyou/hinmoku.xls",
#     2010: "https://www.stat.go.jp/data/cpi/2010/kaisetsu/zuhyou/hinmoku.xls",
#     2015: "https://www.stat.go.jp/english/data/cpi/zuhyou/2015base-list-2.xls",
#     2020: "https://www.stat.go.jp/english/data/cpi/zuhyou/2020base-list.xlsx",
#     2025: "https://www.stat.go.jp/english/data/cpi/zuhyou/2025base-list.xlsx",
# }

# def collect_weights(base_year, url):
#     response = requests.get(url, timeout=60)
#     response.raise_for_status()

#     # Les noms de feuilles diffèrent selon les bases.
#     sheets = pd.read_excel(
#         BytesIO(response.content),
#         sheet_name=None,
#         header=None,
#         dtype="string",
#     )

#     for sheet_name, raw in sheets.items():
#         raw = raw.apply(lambda col: col.str.strip())

#         # Repère le code de l'indice global et sa colonne.
#         positions = raw.eq("0001").fillna(False).stack()
#         positions = positions[positions]

#         if positions.empty:
#             continue

#         first_row, group_col = positions.index[0]

#         # Structure commune à partir de la colonne du code de groupe :
#         # groupe, produit, numéro de série, poids brut, poids / 10 000.
#         data = raw.iloc[first_row:].copy()

#         group_code = data.iloc[:, group_col]
#         item_code = data.iloc[:, group_col + 1]

#         weight_raw = pd.to_numeric(
#             data.iloc[:, group_col + 3], errors="coerce"
#         )

#         valid = (
#             group_code.str.fullmatch(r"\d{4}", na=False)
#             | item_code.str.fullmatch(r"\d{4}", na=False)
#         ) & weight_raw.notna()

#         data = data.loc[valid].copy()

#         # Les libellés sont répartis sur plusieurs colonnes hiérarchiques.
#         names = (
#             data.iloc[:, :group_col]
#             .replace(r"^\s*$", pd.NA, regex=True)
#         )

#         depth = names.notna().apply(
#             lambda row: max(
#                 (i for i, present in enumerate(row) if present),
#                 default=-1,
#             ),
#             axis=1,
#         )

#         # Une ligne suivie d'un niveau plus profond est un agrégat.
#         is_leaf = depth.shift(-1).fillna(-1).le(depth)

#         result = pd.DataFrame({
#             "base_year": base_year,
#             "group_code": data.iloc[:, group_col],
#             "item_code": data.iloc[:, group_col + 1],
#             "serial_number": data.iloc[:, group_col + 2],
#             "item_name": names.ffill(axis=1).iloc[:, -1],
#             "weight_raw": pd.to_numeric(
#                 data.iloc[:, group_col + 3], errors="coerce"
#             ),
#             "weight_per_10000": pd.to_numeric(
#                 data.iloc[:, group_col + 4], errors="coerce"
#             ),
#         })

#         result["is_product"] = (
#             result["item_code"].str.fullmatch(r"\d{4}", na=False)
#             & is_leaf
#         )
        
#         if base_year in (2020, 2025):
#             coicop = (
#                 data.iloc[:, -2]
#                 .str.strip()
#                 .replace(["-", "－", ""], pd.NA)
#             )

#             mapping = pd.DataFrame({
#                 "serial_number": data.iloc[:, group_col + 2],
#                 "coicop": coicop
#             }).dropna(subset=["coicop"])

#             result["coicop"] = result["serial_number"].map(
#                 mapping.set_index("serial_number")["coicop"]
#             )
#         else:
#             result["coicop"] = pd.NA

#         # Normalisation avec les poids bruts pour éviter les arrondis.
#         total = result.loc[
#             result["group_code"].eq("0001"), "weight_raw"
#         ].iloc[0]

#         result["weight"] = result["weight_raw"] / total
#         result["source_url"] = url

#         # Vérification : les produits élémentaires reconstituent le total.
#         product_total = result.loc[
#             result["is_product"], "weight_raw"
#         ].sum()

#         if abs(product_total - total) > 1:
#             raise ValueError(
#                 f"Base {base_year} : somme des poids des produits "
#                 f"({product_total}) différente du total ({total})."
#             )

#         return result.reset_index(drop=True)

#     raise ValueError(f"Base {base_year} : tableau de poids introuvable.")


# # Collecte des six bases
# tables = []

# for base_year, url in URLS.items():
#     table = collect_weights(base_year, url)
    
#     # Get only item codes 
#     tables.append(table.dropna(subset="item_code"))
    
#     # Save Excel
#     table.to_excel(f"{base_year}-Base Items CPI Japan.xlsx")

# # Tous les niveaux : produits et agrégats
# dfWeightsAll = pd.concat(tables, ignore_index=True)

# # Uniquement les produits sans doublons
# dfWeights = (
#     dfWeightsAll.loc[dfWeightsAll["is_product"]]
#     .reset_index(drop=True)
# )

# # Vérification : les poids des produits somment à 1 par base
# print(dfWeights.groupby("base_year")["weight"].agg(["count", "sum"]))
# dfWeights = dfWeights.rename(columns={
#     "base_year":"Date"
# })

# # Base de poids... Mensuel, forward-filled depuis 2000-01-01, serial numbers, / 10 000
# weightsAlternative = dfWeights.pivot(columns="item_code", values="weight", index="Date")
# weightsAlternative.index = pd.to_datetime(weightsAlternative.index, format="%Y")
# monthlyWeights = weightsAlternative.resample("MS").ffill()

# # Save into excel file... All items
# monthlyWeights.to_excel("2000-2030 All Base Items CPI Japan.xlsx") # Store filename in YAML

# # Collect at the coicop level, correspondance between coicop and item_code
# # Mapping de référence : COICOP 2025 (589)
# map2025 = (
#     dfWeights.loc[
#         dfWeights["Date"].eq(2025),
#         ["item_code", "coicop"]
#     ]
#     .dropna(subset=["item_code", "coicop"])
#     .drop_duplicates(subset="item_code")
#     .set_index("item_code")["coicop"]
# )

# map2020 = (
#     dfWeights.loc[
#         dfWeights["Date"].eq(2020),
#         ["item_code", "coicop"]
#     ]
#     .dropna(subset=["item_code", "coicop"])
#     .drop_duplicates(subset="item_code")
#     .set_index("item_code")["coicop"]
# )

# # Appliquer à toutes les bases historiques COICOP 2025 et 2020 (historique, a fait partie de COICOP)
# dfWeights["coicop_2025"] = dfWeights["item_code"].map(map2025)

# dfWeights["coicop_historical"] = (
#     dfWeights["coicop_2025"]
#     .fillna(dfWeights["item_code"].map(map2020))
# )

# # Sum over products, save mapping for prices...
# # dfWeights[["item_code","coicop_historical"]].dropna().to_excel("Historical COICOP mapping to items.xlsx")
# mapping = dfWeights.loc[dfWeights["Date"].ge(2025),["item_code","coicop_historical"]].dropna().set_index("item_code")["coicop_historical"]
# mapping.to_excel("Historical COICOP mapping to items.xlsx")

# # Base de 54 colonnes COICOP (poids du groupe COICOP = somme des poids des produits composants le groupe)
# coicopWeights = (
#     dfWeights
#     .dropna(subset=["coicop_historical"])
#     .groupby(["Date", "coicop_historical"])["weight"]
#     .sum()
#     .unstack(fill_value=0)
# )

# # Couverture
# print("COICOP 2020, 2025 coverage since 2000")
# print(coicopWeights.sum(axis=1))

# # Pour mapping avec les prix (601 produits, en attente d'être mappé)
# coicopWeightsItems = (
#     dfWeights
#     .dropna(subset=["coicop_historical"])
#     .groupby(["Date", "item_code"])["weight"]
#     .sum()
#     .unstack(fill_value=0)
# )

# # Dates
# coicopWeightsItems.index = pd.to_datetime(
#     coicopWeightsItems.index.astype(str),
#     format="%Y"
# )

# # Mensualisation
# monthlyCoicopWeights = coicopWeightsItems.resample("MS").ffill() # Months, ItemCodes, Weights (%)
# monthlyCoicopWeights.to_excel("2000-2030 COICOP Items CPI Japan.xlsx") # Store filename in YAML