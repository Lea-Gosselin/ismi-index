
import numpy as np
import pandas as pd

from fismi.models import kalAlgoSimple
from fismi.models import kalAlgoAugmented


def estimate_persistence(prices, config):
    """
    Estimate time-varying inflation persistence using a state-space model
    """

    params = config["persistence"]
    model_name = params["model"].lower()
    lag = params["inflation_lag"]

    # 1. Compute inflation
    headline = prices["HEADLINE"].dropna()
    inflation = np.log(headline).diff(lag).dropna()

    # 2. Select model
    models = {
        "simple": kalAlgoSimple,
        "augmented": kalAlgoAugmented,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = models[model_name]

    # 3. Estimate parameters
    x0 = np.asarray(params["initial_params"], dtype=float)

    estimation = model.optNegLogLike(
        x0,
        inflation.to_numpy(),
        cMu=params["estimate_mu"],
        cRho=params["estimate_rho_bar"],
    )

    if not estimation.success:
        raise RuntimeError(
            f"Kalman optimization failed: {estimation.message}"
        )

    # 4. Run Kalman filter with estimated parameters
    filtered = model.kalman_filter_inflation(
        estimation.x,
        inflation.to_numpy(),
        cMu=params["estimate_mu"],
        cRho=params["estimate_rho_bar"],
    )

    # 5. Convert filtered series to DataFrame
    dates = inflation.index[1:]

    results = pd.DataFrame(
        {
            "Inflation": filtered["Y"],
            "FittedInflation": filtered["FittedY"],
            "Rho": filtered["Rho"],
            "Residuals": filtered["Err"],
        },
        index=dates,
    )

    return {
        "results": results,
        "parameters": estimation.x,
        "loglikelihood": -estimation.fun,
        "optimization": estimation,
        "filtered": filtered,
        "model": model_name,
    }
