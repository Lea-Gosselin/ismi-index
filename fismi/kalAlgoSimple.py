import numpy as np
import pandas as pd
from scipy.optimize import minimize

def get_parameters(X0, cMu=True, cRho=True):
    """
    Parameters of the state-space model

    Measurement equation...
    --------------------
    cY_t = xi_t * Y_{t-1} + w_t
    
    Y_t = (pi_t - mu) 
    cY_t = Y_t - rho_bar * Y_{t-1}
        
    w_t ~ N(0, H) with H=D**2

    State equation...
    --------------
    xi_t = F * xi_{t-1} + v_t
    
    xi_t = (rho_t - rho_bar)

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
    F       : float
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
    
    # Persistence of latent state
    F = X0[i]
    i += 1

    # Measurement innovation variance
    D = X0[i]
    H = D**2
    i += 1

    # State innovation variance
    B = X0[i]
    Q = B**2

    return mu, rho_bar, F, H, Q

# Filter...
def kalman_filter_inflation(X0, pi, cRho=True, cMu=True, return_loglik=False):
    """
    Kalman filter for the following state-space model:
    cY_t = xi_t * Y_{t-1} + w_t
    xi_t = F * xi_{t-1} + v_t

    with:       
        Y_t = (pi_t - mu) 
        cY_t = Y_t - rho_bar * Y_{t-1}
        w_t ~ N(0, H) with H=D**2
        
    and:        
        xi_t = (rho_t - rho_bar)
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
    mu, rho_bar, F, H, Q = get_parameters(X0, cMu, cRho)
    
    # Inflation centered around its long-run level (mu)
    Xt = pi[:-1] - mu

    # Reduced-form dependent variable:
    # \tilde Y_t = (pi_t - mu) - rho_bar * (pi_{t-1} - mu)
    Yt = (pi[1:] - mu) - rho_bar * Xt
    
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

    # =========================================================
    # Kalman filter
    # =========================================================
    # 1. Initialisation
    # E[xi_0] = 0
    St[0] = 0.0

    # Unconditional variance of stationary AR(1)
    # Var(xi_t) = Q / (1 - F^2)
    Pt[0] = Q / (1.0 - F**2)
    # Pt[0] = 0.3


    
    for t in range(1, T):
        # 2. Forecast the states (St)
        # xi_{t|t-1}    = F xi_{t-1|t-1}
        # P^xi_{t|t-1}  = F^2 P^xi_{t-1|t-1} + Q
        Stf[t] = F * St[t-1]
        PtS[t] = F**2 * Pt[t-1] + Q
        
        # 3. Predict Yt
        # Y_{t|t-1} = Y_{t-1} xi_{t|t-1} 
        Ytf[t] = Xt[t] * Stf[t]
        
        # 4. Inflation forecast error and variance
        # e_t^Y = Y_t - Y_{t|t-1}
        # P^Y_{t|t-1} = Y_{t-1}^2 P^xi_{t|t-1} + H
        Err[t] = Yt[t] - Ytf[t]
        PtY[t] = Xt[t]**2 * PtS[t] + H
        
        # 5. Updating the states with the Kalman gain
        # K_t = P^xi_{t|t-1} Y_{t-1} / P^Y_{t|t-1}
        # xi_{t|t} = xi_{t|t-1} + K_t e_t^pi
        # P^xi_{t|t} = P^xi_{t|t-1} - K_t * Y_{t-1} * P^xi_{t|t-1}
        Kt[t] = (PtS[t] * Xt[t]) / PtY[t]
        St[t] = Stf[t] + Kt[t] * Err[t]
        Pt[t] = PtS[t] - Kt[t] * Xt[t] * PtS[t]

        # 6. Updating filtered variables
        # rho_t = rho_bar + P^xi_{t|t}
        # pi_{t|t} = rho_t * (mu + Y_{t-1})
        Rhot[t] = rho_bar + St[t]
        Pit[t] = (rho_bar + St[t]) * Xt[t] + mu
        
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
        "Err": Err,
        "StateMse": Pt,
        "centeredState": St,
        "Rho": Rhot,
        "KalmanGain": Kt,
        "LogLikelihood": Lt
        }
    
    neg_loglik = -np.nansum(Lt)

    if return_loglik:
        return neg_loglik

    return results


# Optimise -LogLikelihood
def optNegLogLike(
    X0,
    pi,
    cRho=True,
    cMu=True
):

    bounds = []
    if cMu:
        # bounds.append((0.01999, 0.02001))     # mu, set to be around target
        bounds.append((None, None))             # mu
        
    if cRho:
        bounds.append((None, None))             # rho_bar

    bounds += [
        (-0.999, 0.999),                        # F
        (1e-6, None),                           # D
        (1e-6, None),                           # B
    ]

    res = minimize(
        kalman_filter_inflation,
        X0,
        args=(pi, cRho, cMu, True),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxiter": 10000
        }
    )

    return res
