import numpy as np
import pandas as pd
import statsmodels.api as sm


# Largest autoregressive root of inflation
def rolling_largest_ar_root(inflation, window=120, p=4):
    """
    Rolling AR(p) estimation of monthly inflation persistence.

    Parameters
    ----------
    inflation : Monthly inflation (e.g. 100 * log(P_t/P_{t-1}))
    window : Number of monthly observations per window
    p : Autoregressive order

    Returns
    -------
    Largest companion eigenvalue modulus,
    AR coefficients, and stationarity indicator.
    """

    y = inflation.dropna().astype(float).sort_index()

    if window <= p + 1:
        raise ValueError("window must exceed p + 1")

    rows = []

    for end in range(window, len(y) + 1):

        sample = y.iloc[end-window:end]

        # Construct lagged regressors within the window
        X = pd.concat(
            [sample.shift(j).rename(f"AR{j}")
             for j in range(1, p + 1)],
            axis=1
        )

        data = pd.concat(
            [sample.rename("y"), X], axis=1
        ).dropna()

        model = sm.OLS(
            data["y"],
            sm.add_constant(data[X.columns])
        ).fit()

        phi = model.params[X.columns].to_numpy()

        # Companion matrix
        companion = np.zeros((p, p))
        companion[0, :] = phi

        if p > 1:
            companion[1:, :-1] = np.eye(p - 1)

        eigenvalues = np.linalg.eigvals(companion)

        # Largest eigenvalue in modulus
        rho_max = np.max(np.abs(eigenvalues))

        rows.append({
            "Date": sample.index[-1],
            "LargestRoot": rho_max,
            "SumAR": phi.sum(),
            "Stationary": rho_max < 1,
            **{f"AR{j+1}": phi[j] for j in range(p)}
        })

    return pd.DataFrame(rows).set_index("Date")


# RMSE between actual and fitted
def rmse(act,fit):
    meanSquaredError = np.nanmean((fit - act) ** 2)
    rmse = np.sqrt(meanSquaredError)
    return rmse

# Projections
def localProjections(Y, X, h=np.arange(61)):
    coefs = []
    Err = []
    IC90 = []
    IC10 = []
    SDPlus = []
    SDMinus = []

    for horz in h:
        y = Y.iloc[:, horz].values

        # OLS 
        model = sm.OLS(y, X.values).fit(
        # cov_type="HAC",
        # cov_kwds={"maxlags": max(1, horz)}
        ).get_robustcov_results(cov_type='HAC', maxlags=max(1, horz))
        coef = model.params

        # Contemporaneous coefficient
        B0 = coef[0]
        seB0 = model.bse[0]

        coefs.append(coef)
        Err.append(model.resid)

        IC10.append(B0 - 1.645 * seB0)
        IC90.append(B0 + 1.645 * seB0)
        SDMinus.append(B0 - 1 * seB0)
        SDPlus.append(B0 + 1 * seB0)
        
    finalCoefs = pd.DataFrame(coefs, columns=X.columns)
    return finalCoefs, IC10, IC90, SDPlus, SDMinus