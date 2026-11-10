
import numpy as np
import pandas as pd

from fismi.models import kalAlgoSimple
from fismi.models import kalAlgoAugmented

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

    if not estimation.success:
        raise RuntimeError(
            f"Kalman optimization failed: {estimation.message}"
        )
    
    print("Maximum constraint violation (< tol):",
      -np.min(model.persistence_constraint(estimation.x, inflation.to_numpy())))

    # 4. Run Kalman filter with estimated parameters
    filtered = model.kalman_filter_inflation(
        estimation.x,
        inflation.to_numpy(),
        cMu=params["estimate_mu"],
        cRho=params["estimate_rho_bar"]
    )

    # 5. Convert filtered series to DataFrame
    dates = inflation.index[1:]

    results = pd.DataFrame(
        {
            "Inflation": filtered["Y"],
            "FittedInflation": filtered["FittedY"],
            "ForecastInflation": filtered["ForecastY"],
            "RhoFiltered": filtered["Rho"],
            "Residuals": filtered["Err"],
            "forecastVarState": filtered["forecastVarState"],
            "filteredVarState": filtered["filteredVarState"],
            "forecastVarY": filtered["forecastVarY"],
            "forecastState":filtered["forecastState"],
            "filteredState": filtered["filteredState"],
            "KalmanGain": filtered["KalmanGain"],
            "LogLikelihood": filtered["LogLikelihood"]
        },
        index=dates,
    )
    
    # 6. Smooth estimates
    smoothed = model.kalman_smoother(
        filtered,
        estimation.x,
        cMu=params["estimate_mu"],
        cRho=params["estimate_rho_bar"]
        )
    
    rhoSmoothed = pd.DataFrame(
        {
            "RhoSmoothed": smoothed["smoothedRho"],
            "VarStateSmoothed": smoothed["smoothedVarState"]
        },
        index=dates,
    )
    
    # 7. Merge with results
    results = pd.merge(
        results,
        rhoSmoothed,
        right_index = True,
        left_index = True
    )
    
    return {
        "results": results,
        "parameters": estimation.x,
        "loglikelihood": -estimation.fun,
        "optimization": estimation,
        "filtered": filtered,
        "smoothed": smoothed,
        "model": model_name,
    }
