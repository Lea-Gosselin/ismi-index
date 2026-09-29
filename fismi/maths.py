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
    
    # if abs(F) >= 1:
    #     raise ValueError("Transition equation non-stationary.")

    # if H <= 0:
    #     raise ValueError("H must be strictly positive.")

    # if Q < 0:
    #     raise ValueError("Q must be non-negative.")

    # if T < 2:
    #     raise ValueError("At least two inflation observations are required.")

    # ---------------------------------------------------------
    # Storage variables
    # ---------------------------------------------------------

    xi_pred = np.full(T, np.nan)
    xi_filt = np.full(T, np.nan)
    P_xi_pred = np.full(T, np.nan)
    P_xi_filt = np.full(T, np.nan)
    rho_pred = np.full(T, np.nan)
    rho_filt = np.full(T, np.nan)
    pi_pred = np.full(T, np.nan)
    pi_filt = np.full(T, np.nan)
    innovation = np.full(T, np.nan)
    P_pi = np.full(T, np.nan)
    kalman_gain = np.full(T, np.nan)
    loglik_t = np.full(T, np.nan)

    # =========================================================
    # 1. Initialisation
    # =========================================================

    # E[xi_0] = 0
    xi_filt[0] = 0.0

    # Unconditional variance of stationary AR(1)
    # Var(xi_t) = Q / (1 - F^2)
    P_xi_filt[0] = Q / (1.0 - F**2)

    # Corresponding long-run persistence
    rho_filt[0] = rho_bar

    # =========================================================
    # Kalman filter
    # =========================================================

    for t in range(1, T):
        pi_lag = pi[t - 1]
        pi_t = pi[t]

        # =====================================================
        # 2. Predict the state
        # xi_{t|t-1}    = F xi_{t-1|t-1}
        # P^xi_{t|t-1}  = F^2 P^xi_{t-1|t-1} + Q
        # =====================================================

        xi_pred[t] = F * xi_filt[t - 1]
        P_xi_pred[t] = (F**2 * P_xi_filt[t - 1] + Q)

        # Predicted persistence...
        # rho_{t|t-1} = rho_bar + xi_{t|t-1}
        rho_pred[t] = rho_bar + xi_pred[t]

        # =====================================================
        # 3. Predict inflation
        # pi_{t|t-1} = mu + (rho_bar + xi_{t|t-1}) pi_{t-1}
        # =====================================================
        pi_pred[t] = (mu + rho_pred[t] * pi_lag)

        # -----------------------------------------------------
        # Inflation forecast error
        # e_t^pi = pi_t - pi_{t|t-1}
        # -----------------------------------------------------
        innovation[t] = (pi_t - pi_pred[t])

        # -----------------------------------------------------
        # Forecast-error variance
        # P^pi_{t|t-1} = pi_{t-1}^2 P^xi_{t|t-1} + H
        # -----------------------------------------------------
        P_pi[t] = (pi_lag**2 * P_xi_pred[t] + H)

        # =====================================================
        # 4. Kalman Gain
        # K_t = P^xi_{t|t-1} pi_{t-1}
        # =====================================================
        kalman_gain[t] = (P_xi_pred[t] * pi_lag / P_pi[t])

        # =====================================================
        # 5. Update the state
        # xi_{t|t} = xi_{t|t-1} + K_t e_t^pi
        # =====================================================
        xi_filt[t] = (xi_pred[t] + kalman_gain[t] * innovation[t])

        # -----------------------------------------------------
        # Updated state uncertainty
        # P^xi_{t|t} = P^xi_{t|t-1} - K_t pi_{t-1} P^xi_{t|t-1}
        # -----------------------------------------------------
        P_xi_filt[t] = (P_xi_pred[t] - kalman_gain[t] * pi_lag * P_xi_pred[t])

        # -----------------------------------------------------
        # Recover filtered persistence
        # rho_{t|t} = rho_bar + xi_{t|t}
        # -----------------------------------------------------
        rho_filt[t] = (rho_bar + xi_filt[t])
        
        # -----------------------------------------------------
        # Recover filtered inflation
        # pi_{t|t} = mu + rho_{t|t} * pi_{t-1}
        # -----------------------------------------------------
        pi_filt[t] = (mu + rho_filt[t] * pi_lag)


        # =====================================================
        # 6. Log-likelihood
        # =====================================================

        loglik_t[t] = -0.5 * (
            np.log(2 * np.pi)
            + np.log(P_pi[t])
            + innovation[t]**2 / P_pi[t]
        )

    # =========================================================
    # Results...
    # =========================================================
    results = {
        "Inflation": pi,
        "InflationPredicted": pi_pred,
        "InflationFiltered": pi_filt,
        "Innovation": innovation,
        "XiPredicted": xi_pred,
        "XiFiltered": xi_filt,
        "PxiPredicted": P_xi_pred,
        "PxiFiltered": P_xi_filt,
        "RhoPredicted": rho_pred,
        "RhoFiltered": rho_filt,
        "PInflation": P_pi,
        "KalmanGain": kalman_gain,
        "LogLikelihood": loglik_t
        }
    
    loglik = -np.nansum(loglik_t)

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