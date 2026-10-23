
from pathlib import Path
import numpy as np
import pandas as pd
import os
from fismi.data.process import prepare_data
from fismi.data import query
from dotenv import load_dotenv

# Path
ROOT = Path(__file__).resolve().parents[2]

# Load private environment variables
load_dotenv(ROOT / ".env")

def collect_us(config):
    """
    Collect and prepare united states PCE indices and weights.

    Returns
    -------
    prices : pd.DataFrame, monthly price indices including aggregate price index (HEADLINE)
    weights : pd.DataFrame, monthly normalized component weights
    """

    params = config["data"] # fismi.io, load_config
    level = params["level"]
    incomplete_tickers = params["incomplete_tickers"]
    api_key_bea = os.getenv(params["api_key_bea"])
    
    if not api_key_bea:
        raise ValueError(f"Missing BEA API key: {params['api_key_env']}")
    
    # 1. Download BEA data
    price_raw = query.getBeaData(
        params["price_table"], api_key_bea
    )
    expenditure_raw = query.getBeaData(
        params["weights_table"], api_key_bea
    )

    # 2. Load PCE classification
    classification = pd.read_excel(
        ROOT / params["classification_file"],
        dtype={"LineNumber": str, "Level": float},
    )

    selected_codes = classification.loc[
        classification["Level"] == level, "LineNumber"
    ].astype(str)

    # 3. Select component indices and expenditures
    def pivot_components(df):
        selected = df.loc[
            df["LineNumber"].isin(selected_codes)
        ].copy()

        if selected.duplicated(
            ["TIME_PERIOD", "LineNumber"]
        ).any():
            raise ValueError(
                "Duplicate BEA observations for component/date"
            )

        return selected.pivot(
            index="TIME_PERIOD",
            columns="LineNumber",
            values="DataValue",
        )

    prices = pivot_components(price_raw)
    weights = pivot_components(expenditure_raw)

    # 4. Add headline PCE price index
    headline = price_raw.loc[
        price_raw["SeriesCode"] == params["headline_series"]
    ]

    if headline.duplicated("TIME_PERIOD").any():
        raise ValueError("Duplicate headline observations")

    prices["HEADLINE"] = headline.set_index(
        "TIME_PERIOD"
    )["DataValue"]
    
    # 6. Remove incomplete tickers from the dataframes
    if level == 4:
        prices = prices.loc[:, ~prices.columns.isin(incomplete_tickers)]
        weights = weights.loc[:, ~weights.columns.isin(incomplete_tickers)]

    # 7. Standardize and normalize
    prices, weights = prepare_data(prices, weights)
    
    return prices, weights