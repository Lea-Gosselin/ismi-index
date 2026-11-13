
from pathlib import Path

import numpy as np
import pandas as pd
import matlab.engine

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
    ROOT = Path(__file__).resolve().parents[2]
    params = config["data"] # fismi.io, load_config
    level = params["level"]
    incomplete_tickers = params["incomplete_tickers"]
    headline_ticker = params["headline_ticker"]
    price_url = params["price_url"]
    weight_url = params["weight_url"]
    

    # 0. Download CVS monthly headline index
    cvsUrl = price_url.rsplit("/", 1)[0] + "/" + headline_ticker.split(".",1)[1]
    _, headline_price = query.getEcbData(
    cvsUrl
    , colName="KEY", params=None
    )

    # 1. Download HICP indices and annual weights
    _, raw_prices = query.getEcbData(
        price_url, colName="KEY", params=None
    )
    
    _, raw_weights = query.getEcbData(
        weight_url, colName="KEY", params=None
    )

    # 1. Bis - Replace monthly Index by CVS monthly one
    raw_prices = raw_prices[~(raw_prices.KEY == "HICP.M.U2.N.000000.4D0.INX")] 
    raw_prices = pd.concat([headline_price, raw_prices])
    
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
    prices.insert(0,"HEADLINE",headline)
    
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
    
    # 8. Seasonal adjustment
    eng = matlab.engine.start_matlab()

    
    # prices : pandas DataFrame avec TIME_PERIOD et les indices de prix
    dates = prices.index.strftime("%Y-%m-%d").tolist()
    cols = prices.columns.tolist()
    values = matlab.double(
        prices[cols].to_numpy(dtype=float).tolist()
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

    # 9. Standardize and normalize
    prices, weights = prepare_data(prices_sa, weights)
    

    return prices, weights
