
import pandas as pd
import numpy as np


def prepare_data(prices, weights, headline="HEADLINE"):
    """
    Standardize monthly price indices and expenditure weights dataframes

    Parameters
    ----------
    prices : pd.DataFrame, monthly price indices including aggregate price index (HEADLINE)
    weights : pd.DataFrame, monthly component weights
    headline : str, column identifying aggregate inflation

    Returns
    -------
    prices, weights : pd.DataFrame, validated and aligned datasets, normed weights
    """

    prices = prices.copy()
    weights = weights.copy()

    # 1. Monthly datetime indices
    for df in (prices, weights):
        df.index = pd.to_datetime(df.index)
        df.index = df.index.to_period("M").to_timestamp()
        df.sort_index(inplace=True)

        if df.index.has_duplicates:
            raise ValueError("Duplicate monthly dates at import")

        df.columns = df.columns.astype(str)

    # 2. Common component codes
    if headline not in prices.columns:
        raise ValueError(f"Missing headline price index: {headline}")

    components = prices.columns.drop(headline)

    # 3. Align monthly observations
    dates = prices.index.intersection(weights.index)
    prices = prices.loc[dates, [headline, *components]]
    weights = weights.loc[dates, components]

    if prices.empty:
        raise ValueError("No common monthly observations")

    # 4. Validate values
    if (prices <= 0).any().any():
        raise ValueError("Price indices must be positive")

    if (weights < 0).any().any():
        raise ValueError("Weights must be positive")

    # 5. Normalize weights accounting for available price indices
    # A component is active only when both its price and weight exist (some indexes are available later in the sample)
    # Sum weights of available components only
    total = weights.sum(axis=1, skipna=True)

    if (total <= 0).any():
        raise ValueError("No available components for some dates")

    # Normalize weights to sum to 1 each month
    weights.div(total, axis=0)

    return prices, weights
