import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import solve
from statsmodels.tsa.arima_process import arma_acf, arma_pacf

def get_parameters(X0, cMu=True, cRho=True):
    """
    Parameters of the state-space model
    Parameters
    ----------
    X0 : [mu, rho_bar, a1, a2, ..., aq, v, w]
    cMu : False means mu = 0
    cRho : False means rho_bar = 0

    Returns
    -------
    mu      : float
    rho_bar : float
    a1      : float
    ...
    aq      : float
    v       : float
    w       : float
    """
    
    i=0
    q=len(X0)
    
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
    
    # MA component
    F = X0[i:-2]
    q = len(F)

    # Measurement innovation variance
    sigma_e = X0[-2]
    e = sigma_e**2
    
    # State innovation variance
    sigma_u = X0[-1]
    u = sigma_u**2

    # State innovation matrix
    Q = np.diag([u, e])

    return mu, rho_bar, F, Q, e, u, q

# Filter...
def kalman_filter_inflation(X0, pi, cRho=True, cMu=True, return_loglik=False):
    # ---------------------------------------------------------
    # Consistency checks and data management
    # ---------------------------------------------------------
    mu, rho_bar, F, Q, e, u, q = get_parameters(X0, cMu, cRho)
    pi = np.asarray(pi, dtype=float).ravel()

    # Inflation (centered or not around its long-run level)
    Xt = pi[:-1] - mu

    # Reduced-form dependent variable:
    Yt = (pi[1:] - mu) 
    T = len(Yt)

    # ---------------------------------------------------------
    # State-space matrices
    # ---------------------------------------------------------
    N = q + 2    
    
    # Diffuse initialisation for persistence parameter    
    N = q + 2

    # Transition matrix (N, N)
    Tmat = np.zeros((N, N))
    Tmat[0, 0] = 1.0
    Tmat[2:, 1:-1] = np.eye(q)

    # Innovation loading matrix (N, 2)
    R = np.zeros((N, 2))
    R[0, 0] = 1.0
    R[1, 1] = 1.0

    RQR = R @ Q @ R.T
    
    # ---------------------------------------------------------
    # Storage variables
    # ---------------------------------------------------------
    Stf = np.zeros((T, N))
    St = np.zeros((T, N))
    Rhot = np.zeros(T)

    Ytf = np.zeros(T)
    Pit = np.zeros(T)
    Pitt = np.zeros(T)
    Err = np.zeros(T)

    PtS = np.zeros((T, N, N))
    PtY = np.zeros(T)
    Pt = np.zeros((T, N, N))
    
    Kt = np.zeros((T, N))
    Lt = np.zeros(T)

    # =========================================================
    # Kalman filter
    # =========================================================
    # 1. Initialisation
    
    # Esperance of the state
    SForecast = np.zeros(N)
    SForecast[0] = rho_bar
    
    # Variance of state
    PForecast = np.diag(
        [u] + [e] * (N - 1)
    )

    I = np.eye(N)
    
    for t in range(T):
        if t == 0:
            Stf[t] = SForecast
            PtS[t] = PForecast
        else:
            # 2. Forecast the states (St)
            Stf[t] = Tmat @ St[t-1]
            PtS[t] = (
                Tmat @ Pt[t-1] @ Tmat.T + RQR
            )
        
        # 2. Observation vector
        Zt = np.concatenate((
            [Xt[t], 1.0], F
        ))
        
        # 3. Observation prediction
        Ytf[t] = Zt @ Stf[t]
        
        # 4. Forecast error and variance
        Err[t] = Yt[t] - Ytf[t]
        PtY[t] = Zt @ PtS[t] @ Zt
        
        if PtY[t] <= 0 or not np.isfinite(PtY[t]):
            if return_loglik:
                return np.inf
            raise ValueError("Invalid forecast variance")
        
        # 5. Kalman gain
        Kt[t] = (PtS[t] @ Zt) / PtY[t]

        # 6. State update
        St[t] = Stf[t] + Kt[t] * Err[t]

        # Joseph covariance update
        A = I - np.outer(Kt[t], Zt)
        Pt[t] = (
            A @ PtS[t] @ A.T
        )

        # Numerical symmetry
        Pt[t] = 0.5 * (Pt[t] + Pt[t].T)

        # 7. Inflation estimates
        Pit[t] = Ytf[t] + mu
        Pitt[t] = Zt @ St[t] + mu

        # 8. Filtered persistence
        Rhot[t] = St[t, 0]

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
        "FittedY": Pitt,
        "ForecastY": Pit,
        "Err": Err,
        "forecastVarState": PtS,
        "filteredVarState": Pt,
        "forecastVarY": PtY,
        "forecastState":Stf,
        "filteredState": St,
        "Rho": Rhot,
        "KalmanGain": Kt,
        "LogLikelihood": Lt
        }
    
    neg_loglik = -np.nansum(Lt)

    if return_loglik:
        return neg_loglik

    return results

def kalman_smoother(results, parameters, cMu=True, cRho=True):
    # Get parameters
    mu, rho_bar, F, Q, e, u, q = get_parameters(
        parameters,
        cMu=cMu,
        cRho=cRho
    )
    
    # Filter outputs
    Stf = results["forecastState"]
    St = results["filteredState"]
    PtS = results["forecastVarState"]
    Pt = results["filteredVarState"]

    n_obs, N = St.shape

    # Transition matrix
    Tmat = np.zeros((N, N))
    Tmat[0, 0] = 1.0
    Tmat[2:, 1:-1] = np.eye(q)

    # Storage
    StSmoothed = np.zeros_like(St)
    PtSmoothed = np.zeros_like(Pt)

    # Terminal conditions
    StSmoothed[-1] = St[-1]
    PtSmoothed[-1] = Pt[-1]

    # Backward recursion
    for t in range(n_obs - 2, -1, -1):

        # RTS smoothing gain:
        # J = Pt[t] @ Tmat.T @ inv(PtS[t+1])
        A = Pt[t] @ Tmat.T

        J = solve(
            PtS[t+1],
            A.T,
            assume_a="sym"
        ).T

        # Smoothed state
        StSmoothed[t] = (
            St[t]
            + J @ (StSmoothed[t+1] - Stf[t+1])
        )

        # Smoothed covariance
        PtSmoothed[t] = (
            Pt[t]
            + J @ (PtSmoothed[t+1] - PtS[t+1]) @ J.T
        )

        # Numerical symmetry
        PtSmoothed[t] = 0.5 * (
            PtSmoothed[t] + PtSmoothed[t].T
        )

    return {
        "smoothedState": StSmoothed,
        "smoothedVarState": PtSmoothed,
        "smoothedRho": StSmoothed[:, 0],
        "smoothedVarRho": PtSmoothed[:, 0, 0]
    }

def objective(X0, pi, cRho=True, cMu=True):
    # Original negative log-likelihood
    nll = kalman_filter_inflation(
        X0, pi, cRho, cMu, return_loglik=True
    )

    return nll if np.isfinite(nll) else 1e12

def persistence_constraint(
    X0, pi, cRho=True, cMu=True, rho_max=0.999
):
    results = kalman_filter_inflation(
        X0, pi, cRho, cMu,
        return_loglik=False
    )

    rho_filtered = np.asarray(results["Rho"])
    rho_forecast = np.asarray(results["forecastState"][:, 0])

    rho_all = np.concatenate([
        rho_filtered,
        rho_forecast
    ])

    # Constraints >= 0
    return np.concatenate([
        rho_max - rho_all,
        rho_max + rho_all
    ])


def objective_penalized(
    X0, pi, cRho=True, cMu=True,
    rho_max=0.9999, penalty=1e6
):

    try:

        # 1. Kalman filter
        results = kalman_filter_inflation(
            X0, pi, cRho, cMu,
            return_loglik=False
        )

        # 2. Negative log-likelihood
        nll = -np.sum(results["LogLikelihood"])

        # 3. Kalman smoother
        smoother = kalman_smoother(
            results, X0,
            cMu=cMu,
            cRho=cRho
        )

        # 4. Extract persistence trajectories
        rho_all = np.concatenate([
            results["Rho"],
            results["forecastState"][:, 0],
            smoother["smoothedRho"]
        ])

        if not np.isfinite(nll):
            return 1e12

        if not np.all(np.isfinite(rho_all)):
            return 1e12

        # 5. Stationarity violations
        violations = np.maximum(
            np.abs(rho_all) - rho_max,
            0.0
        )

        # 6. Penalized negative log-likelihood
        return nll + penalty * np.sum(violations**2)

    except (
        ValueError,
        np.linalg.LinAlgError,
        FloatingPointError
    ):
        return 1e12
    
    
# Optimise -LogLikelihood
def optNegLogLike(
    X0,
    pi,
    cMu=True,
    cRho=True,
    rho_max=0.9999
):

    q = len(X0) - int(cMu) - int(cRho) - 2

    bounds = []

    if cMu:
        bounds.append((None, None))

    if cRho:
        bounds.append((-rho_max, rho_max))

    # Fix MA coefficients at their initial values
    i = int(cMu) + int(cRho)

    bounds += [
        (theta, theta)
        for theta in X0[i:i+q]
    ]

    bounds += [
        (1e-8, None),
        (1e-8, None)
    ]

    # Starting parameters
    x = np.asarray(X0, dtype=float).copy()

    # Progressive penalty
    penalties = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

    for penalty in penalties:

        res = minimize(
            objective_penalized,
            x,
            args=(pi, cRho, cMu, rho_max, penalty),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-12,
                "gtol": 1e-6,
                "maxiter": 10000,
                "maxls": 50
            }
        )

        # Evaluate estimated trajectories
        results = kalman_filter_inflation(
            res.x, pi, cRho, cMu
        )

        smoother = kalman_smoother(
            results, res.x,
            cMu=cMu,
            cRho=cRho
        )

        rho_all = np.concatenate([
            results["Rho"],
            results["forecastState"][:, 0],
            smoother["smoothedRho"]
        ])

        max_rho = np.max(np.abs(rho_all))

        nll = -np.sum(results["LogLikelihood"])

        feasible = (
            np.all(np.isfinite(rho_all))
            and max_rho < 1.0
        )

        print(
            f"Penalty: {penalty:.0e} | "
            f"NLL: {nll:.6f} | "
            f"Max |rho|: {max_rho:.8f} | "
            f"Success: {res.success} | "
            f"Feasible: {feasible}"
        )

        # Stop if convergence AND feasibility
        if res.success and feasible:
            break

        # Warm start
        x = res.x.copy()

    res["max_abs_rho"] = max_rho
    res["stationary_paths"] = feasible
    res["unpenalized_nll"] = nll

    if not feasible:
        print("WARNING: Stationarity restriction not satisfied.")

    return res