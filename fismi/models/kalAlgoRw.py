import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

def get_parameters(X0, cMu=True, cRho=True):
    """
    Parameters of the state-space model

    Measurement equation...
    --------------------
    Y_t = xi_t * Y_{t-1} + w_t
    
    Y_t = (pi_t - mu) 
        
    w_t ~ N(0, H) with H=D**2

    State equation...
    --------------
    xi_t = xi_{t-1} + v_t
    xi_0 = rho_bar
    v_t ~ N(0, Q) with Q = BB'

    Parameters
    ----------
    X0 : [mu, rho_bar, F, D, B]
    cMu : False means mu = 0
    cRho : False means rho_bar = 0

    Returns
    -------
    mu      : float
    rho_bar : float
    H       : float
    Q       : float
    """
    
    i=0
    
    # Long-run inflation
    if cMu:
        mu = X0[0]
        i += 1
    else: 
        mu = 0

    # Long-run persistence
    if cRho:
        rho_bar = X0[i]
        i += 1
    else:
        rho_bar = 0

    # Measurement innovation variance
    D = X0[i]
    H = D**2
    i += 1

    # State innovation variance
    B = X0[i]
    Q = B**2

    return mu, rho_bar, H, Q

# Filter...
def kalman_filter_inflation(X0, pi, cRho=True, cMu=True, return_loglik=False):
    """
    Kalman filter for the following state-space model:
    Y_t = xi_t * Y_{t-1} + w_t
    xi_t = xi_{t-1} + v_t

    with:       
        Y_t = (pi_t - mu) 
        w_t ~ N(0, H) with H=D**2
        
    and:        
        xi_0 = rho_bar
        v_t ~ N(0, Q) with Q = BB'
    
    Parameters
    ----------
    X0 :        Starting parameters
    pi :        Inflation rate
    cMu :       False means mu = 0
    cRho :      False means rho_bar = 0
    return_loglik : If True, returns the Gaussian log-likelihood

    Returns
    -------
    results :   Filtered and predicted states, persistence, inflation forecasts and innovations
    loglik :    Gaussian log-likelihood
    """

    # ---------------------------------------------------------
    # Consistency checks and data management
    # ---------------------------------------------------------
    mu, rho_bar, H, Q = get_parameters(X0, cMu, cRho)
        
    # Inflation (centered or not around its long-run level)
    Xt = pi[:-1] - mu

    # Reduced-form dependent variable:
    Yt = pi[1:] - mu

    T = Yt.shape[0]

    # ---------------------------------------------------------
    # Storage variables
    # ---------------------------------------------------------
    Stf = np.full(T, np.nan)
    Ytf = np.full(T, np.nan)
    PtS = np.full(T, np.nan)
    PtY = np.full(T, np.nan)
    Pt = np.full(T, np.nan)
    St = np.full(T, np.nan)
    Rhot = np.full(T, np.nan)
    Pit = np.full(T, np.nan)
    Err = np.full(T, np.nan)
    Kt = np.full(T, np.nan)
    Lt = np.full(T, np.nan)
    
    # RW or not ?
    F = 1

    # =========================================================
    # Kalman filter
    # =========================================================
    # 1. Initialisation
    # E[xi_0] = 0
    SForecast = rho_bar

    # Unconditional variance of stationary AR(1)
    # Var(xi_t) = 0 (RW)
    if F == 1:
        PForecast = 0
    else:
        PForecast = Q/(1-F**2)
    
    for t in range(T):
        if t == 0:
            Stf[t] = SForecast
            PtS[t] = PForecast
        else:
            # 2. Forecast the states (St)
            # xi_{t|t-1}    = xi_{t-1|t-1}
            # P^xi_{t|t-1}  = P^xi_{t-1|t-1} + Q
            Stf[t] = F * St[t-1]
            PtS[t] = F**2 * Pt[t-1] + Q
        
        # 3. Predict Yt
        # cY_{t|t-1} = pi_{t-1} xi_{t|t-1} 
        Ytf[t] = Xt[t] * Stf[t]
        
        # 4. Inflation forecast error and variance
        # e_t^Y = cY_t - cY_{t|t-1}
        # P^Y_{t|t-1} = Y_{t-1}^2 P^xi_{t|t-1} + H
        Err[t] = Yt[t] - Ytf[t]
        PtY[t] = Xt[t]**2 * PtS[t] + H
        
        # 5. Updating the states with the Kalman gain
        # K_t = P^xi_{t|t-1} pi_{t-1} / P^Y_{t|t-1}
        # xi_{t|t} = xi_{t|t-1} + K_t e_t^cY
        # P^xi_{t|t} = P^xi_{t|t-1} - K_t * pi_{t-1} * P^xi_{t|t-1}
        Kt[t] = (PtS[t] * Xt[t]) / PtY[t]
        St[t] = Stf[t] + Kt[t] * Err[t]
        Pt[t] = PtS[t] - Kt[t] * Xt[t] * PtS[t]

        # 6. Updating filtered variables
        Pit[t] = St[t] * Xt[t] + mu

        # 7. Log-likelihood
        # =====================================================
        Lt[t] = -0.5 * (
            np.log(2 * np.pi)
            + np.log(PtY[t])
            + Err[t]**2 / PtY[t]
        )

    # =========================================================
    # Results...
    # =========================================================
    results = {
        "Y": pi[1:],
        "FittedY": Pit,
        "ForecastY": (Stf * Xt + mu),
        "Err": Err,
        "forecastVarState": PtS,
        "filteredVarState": Pt,
        "forecastVarY": PtY,
        "forecastState":Stf,
        "filteredState": St,
        "Rho": St,
        "KalmanGain": Kt,
        "LogLikelihood": Lt
        }
    
    neg_loglik = -np.nansum(Lt)

    if return_loglik:
        return neg_loglik

    return results

def kalman_smoother(results, parameters, cMu=True, cRho=True):
    # Get parameters
    mu, rho_bar, H, Q = get_parameters(
        parameters,
        cMu=cMu,
        cRho=cRho
    )
    
    # RW or not ? 
    F = 1
    
    # Forecast and filtered estimates
    Stf = results["forecastState"]
    St = results["filteredState"]
    PtS = results["forecastVarState"]
    Pt = results["filteredVarState"]
    
    T = Stf.shape[0]
    
    # Store smoothed estimates
    StSmoothed = np.full(T, np.nan)
    PtSmoothed = np.full(T, np.nan)
    
    # Initialise at T
    StSmoothed[-1] = St[-1]
    PtSmoothed[-1] = Pt[-1]
    
    # Backward recursion
    for t in range(T-2, -1, -1):
        J = Pt[t] * F/ PtS[t+1]
        StSmoothed[t] = St[t] + J * (StSmoothed[t+1] - Stf[t+1])
        PtSmoothed[t] = Pt[t] + J**2 * (PtSmoothed[t+1] - PtS[t+1])
    return {
        "smoothedState": StSmoothed,
        "smoothedVarState": PtSmoothed,
        "smoothedRho": StSmoothed
    }

def objective(X0, pi, cRho=True, cMu=True):

    # Original negative log-likelihood
    nll = kalman_filter_inflation(
        X0, pi, cRho, cMu, return_loglik=True
    )

    return nll if np.isfinite(nll) else 1e12


# Optimise -LogLikelihood
def optNegLogLike(
    X0,
    pi,
    cMu=True,
    cRho=True
):

    bounds = []
    if cMu:
        bounds.append((None,None))                # mu
        
    if cRho:
        bounds.append((None,None))     

    bounds += [
        # (1e-3, None),      # D: std of inflation shocks
        # (1e-6, 10e-2),      # B: std of rho_t innovations (8e-2)
        (1e-6, None),      # D: std of inflation shocks
        (1e-6, None),      # B: std of rho_t innovations (8e-2)
    ]
    
    res = minimize(
        objective,
        X0,
        args=(pi, cRho, cMu),
        # method="SLSQP",
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-9,
            # "gtol": 1e-6,
            "maxiter": 10000
        }
    )

    return res