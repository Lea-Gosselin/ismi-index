
import numpy as np
import pandas as pd

from fismi.models import kalAlgoSimple
from fismi.models import kalAlgoAugmented
from fismi.models import kalAlgoRw
from fismi.models import kalAlgoArma

def estimate_persistence(prices, config):
    """
    Estimate time-varying inflation persistence using a state-space model
    """

    params = config["persistence"]
    sDate = params["startDate"]
    model_name = params["model"].lower()
    lag = params["inflation_lag"]
    
    # 1. Compute inflation on selected sample
    headline = prices["HEADLINE"].dropna()
    inflation = np.log(headline).diff(lag).dropna()
    inflation = inflation[inflation.index >= sDate]
    
    # 2. Select model
    models = {
        "simple": kalAlgoSimple,
        "augmented": kalAlgoAugmented,
        "rw": kalAlgoRw,
        "arma": kalAlgoArma
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = models[model_name]

    x0 = np.asarray(params["initial_params"], dtype=float)

    estimation = model.optNegLogLike(
        x0,
        inflation.to_numpy(),
        cMu=params["estimate_mu"],
        cRho=params["estimate_rho_bar"],
    )

    # 4. Run Kalman filter with estimated parameters
    filtered = model.kalman_filter_inflation(
        estimation.x,
        inflation.to_numpy(),
        cMu=params["estimate_mu"],
        cRho=params["estimate_rho_bar"]
    )

    # 5. Convert filtered series to DataFrame
    dates = inflation.index[1:]

    results = filtered
    
    # 6. Smooth estimates
    smoothed = model.kalman_smoother(
        filtered,
        estimation.x,
        cMu=params["estimate_mu"],
        cRho=params["estimate_rho_bar"]
        )
    
    # 7. Merge with results
    for i, key in enumerate(smoothed.keys()):
        results[key] = smoothed[key]

    # Store dates (index)
    results["dates"] = dates
    
    return {
        "results": results,
        "parameters": estimation.x,
        "loglikelihood": -estimation.fun,
        "optimization": estimation,
        "filtered": filtered,
        "smoothed": smoothed,
        "model": model_name,
    }
