import numpy as np
import pandas as pd
import statsmodels.api as sm

def twoStepsAr(dfInflation, rolling_window=110):
    """2-steps AR(1) regressions (Lansing & Shapiro, 2026).

    dfInflation : DataFrame (Dates x Catégories) - Monthly 12-month inflation rate from unchained items
    """

    # Input
    categories = dfInflation.columns
    
    # Stockage temporaire des chocs (résidus de l'AR(1))
    shocks_dict = {cat: pd.Series(index=dfInflation.index, dtype=float) for cat in categories}
    a_dict = {cat: pd.Series(index=dfInflation.index, dtype=float) for cat in categories}
    rho_dict = {cat: pd.Series(index=dfInflation.index, dtype=float) for cat in categories}
    
    for t_idx in range(rolling_window, len(dfInflation)):
        current_date = dfInflation.index[t_idx]

        for cat in categories:
            # 120 month rolling window (Lansing & Shapiro, 2026)
            window_data = dfInflation[cat].iloc[t_idx - rolling_window : t_idx + 1]

            y = window_data.iloc[1:].values        # Inflation rate T
            x = window_data.iloc[:-1].values       # Inflation rate lagged (T-1)
            x = sm.add_constant(x)

            try:
                ### 2-step regressions (to get A and Rho coefficients)             
                #   1) Current inflation shock
                model1 = sm.OLS(y, x).fit()
                residuals = model1.resid
                
                # Inflation AR(1) coefficient (constant)
                a_dict[cat].at[current_date] = model1.params[1]
                
                #   2) AR(1) on OLS' residuals
                yRes = residuals[1:]        
                xRes = residuals[:-1]
                model2 = sm.OLS(yRes, xRes).fit()
                rho_dict[cat].at[current_date] = model2.params[0]
                
                #   3) Store
                # shocks_dict[cat].at[current_date] = model2.fittedvalues[-1]
                shocks_dict[cat].at[current_date] = residuals[-1]

            except:
                a_dict[cat].at[current_date] = None
                rho_dict[cat].at[current_date] = None
                shocks_dict[cat].at[current_date] = None

    df_shocks = pd.DataFrame(shocks_dict)
    aCoef = pd.DataFrame(a_dict)
    rhoCoef = pd.DataFrame(rho_dict)
    
    return df_shocks, aCoef, rhoCoef


def replicateIsmi(dfInflation, df_shocks, dfW, rolling_window=120, k=3):
    """Replication of ISMI Index (Lansing & Shapiro, 2026).
    -------
    posItems : Positive ISMI contribution by component, weighted.
    negItems : Negative ISMI contribution by component, weighted.
    posIsmi  : Aggregate positive ISMI.
    negIsmi  : Aggregate negative ISMI.
    ismIndex : Total ISMI = positive - negative.
    """

    print(
        f"Step 1 : Estimate {len(dfInflation.columns)} AR(1) "
        f"on a {rolling_window} month rolling window..."
    )

    # Estimate shocks
    df_shocks, _, _ = twoStepsAr(
        dfInflation,
        rolling_window=rolling_window
    )

    # Dates for which we can calculate the index
    dates = df_shocks.index.intersection(dfW.index)

    ismIndex = []
    posIsmi = []
    negIsmi = []

    # Per components
    posItems = []
    negItems = []

    print("Step 2 : Compute inflation momentum and aggregate...")

    for current_date in dates:

        t_pos = df_shocks.index.get_loc(current_date)

        # K consecutive shocks
        recent_shocks = df_shocks.iloc[
            t_pos - k + 1 : t_pos + 1
        ]

        # Positive / negative momentum signal by component
        pos_signal = (recent_shocks > 0).all(axis=0).astype(int)

        neg_signal = (recent_shocks < 0).all(axis=0).astype(int)

        # Weights
        w = dfW.loc[current_date]

        # Weighted contribution by component
        share_pos = pos_signal * w
        share_neg = neg_signal * w

        # Store component-level contributions
        posItems.append(share_pos)
        negItems.append(share_neg)

        # Aggregate positive / negative ISMI
        pos = share_pos.sum()
        neg = share_neg.sum()

        posIsmi.append(pos)
        negIsmi.append(neg)

        # Total ISMI
        ismIndex.append(pos - neg)

    ### Convert to DataFrames
    dates = dates[rolling_window+k-1:]
    
    posItems = pd.DataFrame(
        posItems[rolling_window+k-1:],
        index=dates,
        columns=dfInflation.columns
    )

    negItems = pd.DataFrame(
        negItems[rolling_window+k-1:],
        index=dates,
        columns=dfInflation.columns
    )

    posIsmi = pd.Series(
        posIsmi[rolling_window+k-1:],
        index=dates,
        name="ISMI+"
    )

    negIsmi = pd.Series(
        negIsmi[rolling_window+k-1:],
        index=dates,
        name="ISMI-"
    )

    ismIndex = pd.Series(
        ismIndex[rolling_window+k-1:],
        index=dates,
        name="ISMIndex"
    )

    print("Done")

    return posItems, negItems, posIsmi, negIsmi, ismIndex