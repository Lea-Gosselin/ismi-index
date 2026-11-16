import requests
import pandas as pd
import numpy as np
import os
from io import BytesIO

def collect_weights(base_year, url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Les noms de feuilles diffèrent selon les bases.
    sheets = pd.read_excel(
        BytesIO(response.content),
        sheet_name=None,
        header=None,
        dtype="string",
    )

    for sheet_name, raw in sheets.items():
        raw = raw.apply(lambda col: col.str.strip())

        # Repère le code de l'indice global et sa colonne.
        positions = raw.eq("0001").fillna(False).stack()
        positions = positions[positions]

        if positions.empty:
            continue

        first_row, group_col = positions.index[0]

        # Structure commune à partir de la colonne du code de groupe :
        # groupe, produit, numéro de série, poids brut, poids / 10 000.
        data = raw.iloc[first_row:].copy()

        group_code = data.iloc[:, group_col]
        item_code = data.iloc[:, group_col + 1]

        weight_raw = pd.to_numeric(
            data.iloc[:, group_col + 3], errors="coerce"
        )

        valid = (
            group_code.str.fullmatch(r"\d{4}", na=False)
            | item_code.str.fullmatch(r"\d{4}", na=False)
        ) & weight_raw.notna()

        data = data.loc[valid].copy()

        # Les libellés sont répartis sur plusieurs colonnes hiérarchiques.
        names = (
            data.iloc[:, :group_col]
            .replace(r"^\s*$", pd.NA, regex=True)
        )

        depth = names.notna().apply(
            lambda row: max(
                (i for i, present in enumerate(row) if present),
                default=-1,
            ),
            axis=1,
        )

        # Une ligne suivie d'un niveau plus profond est un agrégat.
        is_leaf = depth.shift(-1).fillna(-1).le(depth)

        result = pd.DataFrame({
            "base_year": base_year,
            "group_code": data.iloc[:, group_col],
            "item_code": data.iloc[:, group_col + 1],
            "serial_number": data.iloc[:, group_col + 2],
            "item_name": names.ffill(axis=1).iloc[:, -1],
            "weight_raw": pd.to_numeric(
                data.iloc[:, group_col + 3], errors="coerce"
            ),
            "weight_per_10000": pd.to_numeric(
                data.iloc[:, group_col + 4], errors="coerce"
            ),
        })

        result["is_product"] = (
            result["item_code"].str.fullmatch(r"\d{4}", na=False)
            & is_leaf
        )

        # Normalisation avec les poids bruts pour éviter les arrondis.
        total = result.loc[
            result["group_code"].eq("0001"), "weight_raw"
        ].iloc[0]

        result["weight"] = result["weight_raw"] / total
        result["source_url"] = url

        # Vérification : les produits élémentaires reconstituent le total.
        product_total = result.loc[
            result["is_product"], "weight_raw"
        ].sum()

        if abs(product_total - total) > 1:
            raise ValueError(
                f"Base {base_year} : somme des poids des produits "
                f"({product_total}) différente du total ({total})."
            )

        return result.reset_index(drop=True)

    raise ValueError(f"Base {base_year} : tableau de poids introuvable.")


# Collecte des six bases
tables = []

for base_year, url in URLS.items():
    table = collect_weights(base_year, url)
    tables.append(table)
    print(
        f"Base {base_year} : "
        f"{table['is_product'].sum()} produits récupérés"
    )

# Tous les niveaux : produits et agrégats
dfWeightsAll = pd.concat(tables, ignore_index=True)

# Uniquement les produits élémentaires, sans double comptage
dfWeights = (
    dfWeightsAll.loc[dfWeightsAll["is_product"]]
    .reset_index(drop=True)
)

# Vérification : les poids des produits somment à 1 par base
print(dfWeights.groupby("base_year")["weight"].agg(["count", "sum"]))