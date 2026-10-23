
from pathlib import Path

import numpy as np
import pandas as pd

from fismi.data.process import prepare_data
from fismi.data import query


def collect_ea(config):
    """
    Collect and prepare euro area HICP indices and weights.

    Returns
    -------
    prices : pd.DataFrame, monthly price indices including aggregate price index (HEADLINE)
    weights : pd.DataFrame, monthly normalized component weights
    """

    params = config["data"] # fismi.io, load_config
    level = params["level"]
    incomplete_tickers = params["incomplete_tickers"]
    headline_ticker = params["headline_ticker"]
    price_url = params["price_url"]
    weight_url = params["weight_url"]

    # 1. Download HICP indices and annual weights
    _, raw_prices = query.getEcbData(
        price_url, colName="KEY", params=None
    )
    _, raw_weights = query.getEcbData(
        weight_url, colName="KEY", params=None
    )

    # 2. Prepare monthly price indices
    p = raw_prices.copy()

    p["ICP_ITEM"] = p["KEY"].str.split(".").str[4]
    p["TIME_PERIOD"] = pd.to_datetime(
        p["TIME_PERIOD"].astype(str),
        format="%Y-%m",
    )
    p["INX"] = pd.to_numeric(p["OBS_VALUE"], errors="coerce")

    # 3. Prepare annual weights
    w = raw_weights.copy()

    if "ICP_ITEM" not in w.columns:
        w["ICP_ITEM"] = w["KEY"].str.split(".").str[4]

    w["YEAR"] = pd.to_numeric(
        w["TIME_PERIOD"], errors="raise"
    ).astype(int)

    w["WEIGHTS"] = pd.to_numeric(
        w["OBS_VALUE"], errors="coerce"
    )

    # 4. Determine ECOICOP aggregation levels
    codes = p["ICP_ITEM"].astype(str)
    valid = codes.str.fullmatch(r"\d{6}")
    levels = np.select(
        [
            codes.eq("000000"),
            valid & codes.str[2:].eq("0000"),
            valid & codes.str[3:].eq("000"),
            valid & codes.str[4:].eq("00"),
            valid & codes.str[5:].eq("0"),
        ],
        [0, 1, 2, 3, 4],
        default=5,
    )
    p["LEVEL"] = levels

    # 5. Select requested disaggregation level
    selected = p.loc[
        (p["LEVEL"] == level)
        & (~p["ICP_ITEM"].isin(incomplete_tickers))
    ].copy()

    headline = p.loc[
        p["ICP_ITEM"].eq("000000")
        & p["KEY"].eq(headline_ticker)
    ].copy()

    # 6. Pivot monthly indices
    prices = selected.pivot(
        index="TIME_PERIOD",
        columns="ICP_ITEM",
        values="INX",
    )

    headline = headline.set_index("TIME_PERIOD")["INX"]
    prices["HEADLINE"] = headline

    # 7. Pivot annual weights
    w = w.loc[
        w["ICP_ITEM"].isin(prices.columns.drop("HEADLINE"))
    ]

    annual_weights = w.pivot(
        index="YEAR",
        columns="ICP_ITEM",
        values="WEIGHTS",
    )

    # Map annual weights to the months of the same year
    weights = annual_weights.reindex(prices.index.year)
    weights.index = prices.index

    # 8. Standardize and normalize
    prices, weights = prepare_data(prices, weights)

    return prices, weights
