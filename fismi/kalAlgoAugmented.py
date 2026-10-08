import numpy as np
import pandas as pd
from scipy.optimize import minimize


# PARAMETERS
def get_parameters(X0, cMu=True, cRho=True):
    """
    State-space model
    -----------------
    Y_t = pi_t - mu
    xi_t = Rhot_t - rho_bar

    Measurement equation:
        cY_t = K_t Z_t

    with:
        cY_t = Y_t - rho_bar * Y_{t-1}
        K_t = [Y_{t-1}, 1]
        Z_t = [xi_t, u_t]'

    Transition equation:
        Z_t = A Z_{t-1} + E_t

    with:
        xi_t = F xi_{t-1} + v_t
        u_t  = phi u_{t-1} + w_t

        A = [[F,   0],
             [0, phi]]

        E_t ~ N(0, Sigma)

        Sigma = [[Q, 0],
                 [0, H]]

        Q = B**2
        H = D**2

    Parameters
    ----------
    X0: [mu, rho_bar, F, phi, D, B]

    cMu: False imposes mu = 0

    cRhot: False imposes rho_bar = 0

    Returns
    -------
    mu, rho_bar, A, Sigma, H, Q 
    """

    i = 0

    # Long-run inflation
    if cMu:
        mu = X0[i]
        i += 1
    else:
        mu = 0.0

    # Long-run persistence
    if cRho:
        rho_bar = X0[i]
        i += 1
    else:
        rho_bar = 0.0

    # AR coefficient for the state
    F = X0[i]
    i += 1

    # AR coefficient for the residuals (shocks)
    phi = X0[i]
    i += 1

    # Err of the residuals 
    D = X0[i]
    i += 1
    H = D**2

    # Err of the states
    B = X0[i]
    Q = B**2

    # Transition matrix of the augmented state
    A = np.array([
        [F,   0.0],
        [0.0, phi]
    ])

    # Augmented state Err covariance matrix
    Sigma = np.array([
        [Q,   0.0],
        [0.0, H]
    ])

    return mu, rho_bar, A, Sigma, H, Q


# Kalman Filter...
def kalman_filter_inflation(
    X0,
    pi,
    cRho=True,
    cMu=True,
    return_loglik=False
):
    """
    Kalman filter for the augmented state-space model.

    Structural representation
    -------------------------
        pi_t - mu = Rhot_t (pi_{t-1} - mu) + u_t
        Rhot_t - rho_bar = F (Rhot_{t-1} - rho_bar) + v_t
        u_t = phi u_{t-1} + w_t

    State-space representation
    --------------------------
        cY_t = K_t Z_t
        Z_t = A Z_{t-1} + E_t

    where :
        Y_t = pi_t - mu
        cY_t = Y_t - rho_bar Y_{t-1}

        Z_t = [xi_t, u_t]'
        xi_t = Rhot_t - rho_bar
        K_t = [Y_{t-1}, 1]
        E_t = [v_t, w_t]'
        Var(E_t) = Sigma
    """

    # Parameters
    mu, rho_bar, A, Sigma, H, Q = get_parameters(X0, cMu=cMu, cRho=cRho)
    pi = np.asarray(pi, dtype=float)
    T = len(pi) - 1

    # Transform observed inflation
    # Y_{t-1} = pi_{t-1} - mu
    Xt = pi[:-1] - mu

    # cY_t = Y_t - rho_bar Y_{t-1}
    Yt = (pi[1:] - mu) - rho_bar * Xt

    # Measurement matrices:
    # K_t = [Y_{t-1}, 1]
    Kt = np.column_stack([
        Xt,
        np.ones(T)
    ])

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------
    # Predicted states Z_{t|t-1}
    Ztf = np.full((T, 2), np.nan)

    # Filtered states Z_{t|t}
    Zt = np.full((T, 2), np.nan)

    # Predicted covariance P_{t|t-1}
    PtZ = np.full((T, 2, 2), np.nan)

    # Filtered covariance P_{t|t}
    Pt = np.full((T, 2, 2), np.nan)

    # Predicted transformed inflation
    Ytf = np.full(T, np.nan)

    # Error
    Err = np.full(T, np.nan)

    # Error variance
    PtY = np.full(T, np.nan)

    # Kalman gain (2 x 1)
    KalGain = np.full((T, 2), np.nan)

    # Filtered Rhot_t
    Rhot = np.full(T, np.nan)

    # Filtered u_t
    Ut = np.full(T, np.nan)

    # Fitted inflation
    Pit = np.full(T, np.nan)

    # Log-likelihood contributions
    Lt = np.full(T, np.nan)

    # =========================================================
    # Kalman filter
    # =========================================================
    # 1. Initialisation
    # Unconditional mean
    # E[Z_0] = [0, 0]'
    Zt[0] = np.zeros(2)

    # Stationary unconditional covariance
    # Var(xi_t) = Q / (1-F^2)
    # Var(u_t)  = H / (1-phi^2)
    F = A[0, 0]
    phi = A[1, 1]

    if abs(F) >= 1 or abs(phi) >= 1:
        if return_loglik:
            return 1e12
        raise ValueError("F and phi must satisfy |F| < 1 and |phi| < 1.")

    Pt[0] = np.array([
        [Q / (1.0 - F**2), 0.0],
        [0.0, H / (1.0 - phi**2)]
    ])

    I = np.eye(2)

    for t in range(1, T):
        # 2. Forecast the states (Zt)
        # Z_{t|t-1} = A Z_{t-1|t-1}
        # P_{t|t-1} = A P_{t-1|t-1} A' + Sigma
        Ztf[t] = A @ Zt[t-1]
        PtZ[t] = (A @ Pt[t-1] @ A.T + Sigma)

        # 3. Predict Yt
        # E[tilde Y_t | t-1] = K_t Z_{t|t-1}
        K_t = Kt[t].reshape(1, 2)
        Ytf[t] = (K_t @ Ztf[t]).item()

        # 4. Inflation forecast error and variance
        # e_t^Y = Y_t - Y_{t|t-1}
        # S_t = K_t P_{t|t-1} K_t'
        # No additional measurement variance: u_t is already part of the state
        Err[t] = (Yt[t] - Ytf[t])
        PtY[t] = (K_t @ PtZ[t] @ K_t.T).item()

        if PtY[t] <= 0:
            if return_loglik:
                return 1e12
            raise ValueError(
                f"Non-positive Err variance at t={t}"
            )

        # 5. Compute the Kalman Gain
        # K_t = P^Z_{t|t-1} K_{t-1} / P^Y_{t|t-1}
        G_t = (PtZ[t] @ K_t.T / PtY[t])
        KalGain[t] = G_t[:, 0]

        # 6. Update the states
        # Z_{t|t} = Z_{t|t-1} + K_t * e_t^Y
        # P^Z_{t|t} = P^Z_{t|t-1} - K_t * Y_{t-1} * P^Z_{t|t-1}
        Zt[t] = (Ztf[t] + G_t[:, 0] * Err[t])

        # (Joseph form for numerical stability)
        M = I - G_t @ K_t
        Pt[t] = (M @ PtZ[t] @ M.T)

        # 7. Updating filtered variables
        xi_t = Zt[t, 0]
        u_t = Zt[t, 1]

        Rhot[t] = rho_bar + xi_t
        Ut[t] = u_t

        # Structural equation:
        # pi_t - mu = Rhot_t (pi_{t-1} - mu) + u_t
        Pit[t] = (mu + Rhot[t] * Xt[t] + Ut[t])

        # 7. Log-likelihood
        # ====================================================
        Lt[t] = -0.5 * (
            np.log(2.0 * np.pi)
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
        "ErrVariance": PtY,
        "StateMse": Pt,
        "PredictedState": Ztf,
        "FilteredState": Zt,
        "Xi": Zt[:, 0],
        "Ut": Ut,
        "Rho": Rhot,
        "KalmanGain": KalGain,
        "LogLikelihood": Lt,
    }

    # Return negative log-likelihood
    neg_loglik = -np.nansum(Lt)

    if return_loglik:
        return neg_loglik

    return results


# ============================================================
# OPTIMISATION
# ============================================================
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
        (-0.999, 0.999),                        # phi
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