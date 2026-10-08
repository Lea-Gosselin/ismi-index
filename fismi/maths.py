import numpy as np
import pandas as pd
import statsmodels.api as sm

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