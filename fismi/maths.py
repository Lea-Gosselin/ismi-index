import numpy as np
import pandas as pd
from scipy.optimize import minimize

def get_parameters(X0, M):
    """
    Parameters of the state-space model

    Measurement equation...
    --------------------
    pi_t = mu + (rho_t + xi_t) * pi_{t-1} + w_t
    xi_t = (rho_t - rho_bar)
    w_t ~ N(0, H) with H=DD'

    State equation...
    --------------
    rho_t - rho_bar = F * (rho_{t-1} - rho_bar) + v_t
    v_t ~ N(0, Q) with Q = BB'

    Parameters
    ----------
    X0 : [
         mu
         rho_bar,
         F,
         D,
         B,
         ]

    M : Number of inflation components (1 so far)

    Returns
    -------
    mu      : (M,)
    rho_bar : float
    F       : float
    H       : (M, M)
    Q       : float
    """
    
    # Long-run monthly inflation
    mu = X0[0]

    # Long-run persistence
    rho_bar = X0[M]

    # State persistence
    F = X0[M+1]

    # Variance of state innovation
    D = X0[M+2]
    H = D**2

    # Variance of measurement errors
    B = X0[M+3]
    Q = B**2

    return mu, rho_bar, F, H, Q

# Filter...
def kalman_filter_inflation(X0, pi, return_loglik=False):
    """
    Kalman filter for the following state-space model:
        pi_t = mu + (rho_bar + xi_t) * pi_{t-1} + w_t
        xi_t = F * xi_{t-1} + v_t

    with:        w_t ~ N(0, H),         v_t ~ N(0, Q)
    and:        rho_t = rho_bar + xi_t

    Parameters
    ----------
    inflation : Monthly inflation rate pi_t
    mu :        Constant in the measurement equation
    rho_bar :   Long-run level of inflation persistence
    F :         Persistence of the latent deviation xi_t ; Stationarity requires |F| < 1
    H :         Variance of the measurement innovation w_t
    Q :         Variance of the state innovation v_t
    return_loglik : If True, returns the Gaussian log-likelihood

    Returns
    -------
    results :   Filtered and predicted states, persistence, inflation forecasts and innovations
    loglik :    Gaussian log-likelihood
    """

    # ---------------------------------------------------------
    # Consistency checks and data management
    # ---------------------------------------------------------
    T = pi.shape[0]
    try:
        M = pi.shape[1]
    except:
        M = 1
        
    mu, rho_bar, F, H, Q = get_parameters(X0, M)
    mu = 0
    rho_bar = 0
    
    # On centre les mesures...
    Y = pi - mu

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
    Xt = np.full(T, np.nan)
    St = np.full(T, np.nan)
    Err = np.full(T, np.nan)
    Kt = np.full(T, np.nan)
    Lt = np.full(T, np.nan)

    # =========================================================
    # Kalman filter
    # =========================================================
    
    # =========================================================
    # 1. Initialisation
    # E[xi_0] = 0
    St[0] = 0.0

    # Unconditional variance of stationary AR(1)
    # Var(xi_t) = Q / (1 - F^2)
    Pt[0] = Q / (1.0 - F**2)

    # Corresponding long-run persistence
    # rho_filt[0] = rho_bar
    
    for t in range(1, T):
        Xt = Y[t - 1]
        Yt = Y[t]
        
        # 2. Forecast the states (St)
        # xi_{t|t-1}    = F xi_{t-1|t-1}
        # P^xi_{t|t-1}  = F^2 P^xi_{t-1|t-1} + Q
        Stf[t] = F * St[t-1]
        PtS[t] = F**2 * Pt[t-1] + Q
        
        # 3. Predict Yt
        # Y_{t|t-1} = Y_{t-1} xi_{t|t-1} 
        Ytf[t] = Xt*Stf[t]
        
        # 4. Inflation forecast error and variance
        # e_t^Y = Y_t - Y_{t|t-1}
        # P^Y_{t|t-1} = Y_{t-1}^2 P^xi_{t|t-1} + H
        Err[t] = Yt-Ytf[t]
        PtY[t] = Xt**2 * PtS[t] + H
        
        # 5. Updating the states with the Kalman gain
        # K_t = P^xi_{t|t-1} Y_{t-1} / P^Y_{t|t-1}
        # xi_{t|t} = xi_{t|t-1} + K_t e_t^pi
        # P^xi_{t|t} = P^xi_{t|t-1} - K_t * Y_{t-1} * P^xi_{t|t-1}
        Kt[t] = (PtS[t] * Xt) / PtY[t]
        St[t] = Stf[t] + Kt[t] * Err[t]
        Pt[t] = PtS[t] - Kt[t] * Xt * PtS[t]

        # 6. Updating filtered variables
        # rho_t = rho_bar + P^xi_{t|t}
        # pi_{t|t} = rho_t * (mu + Y_{t-1})
        Rhot[t] = rho_bar + St[t]
        Pit[t] = Rhot[t] * (mu + Xt)
        
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
        "Y": pi,
        "FittedY": Pit,
        "Err": Err,
        "StateMse": Pt,
        "centeredState": St,
        "Rho": Rhot,
        "KalmanGain": Kt,
        "LogLikelihood": Lt
        }
    
    loglik = -np.nansum(Lt)

    if return_loglik:
        return loglik

    return results


# Optimise -LogLikelihood
def optNegLogLike(X0, pi):

    bounds = [
        (None, None),      # mu
        (None, None),      # rho_bar
        (0, 0.999),        # F
        (1e-6, None),      # D
        (1e-6, None),      # B
    ]

    res = minimize(
        kalman_filter_inflation,
        X0,
        args=(pi, True),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxiter": 10000
        }
    )

    return res