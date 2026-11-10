from fismi.io import load_input, load_config
import numpy as np
import pandas as pd
from fismi.models import kalAlgoSimple
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm 
import statistics as stat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Select country
country = "US"

# Select seed
seed = 42
rng = np.random.default_rng(seed=seed)

# Start estimates
print(f"\n--- {country} ---")
config = load_config(country)
startDate = config["persistence"]["startDate"]
prices, weights = load_input(country)
    
rng = np.random.default_rng(42)

mom = np.log(
    prices["HEADLINE"].dropna()
).diff(
    1
)
yoy = np.log(
    prices["HEADLINE"].dropna()
).diff(
    12
)

# Merge Mom and Yoy inflation rate
df = pd.DataFrame({
    "MoM": mom,
    "YoY": yoy
})
T = df.shape[0]
df = df.reset_index()

# # Flag level shifts (determinist)
# mask = (
#     df["TIME_PERIOD"].between("1968-08-01", "1991-08-01") |
#     df["TIME_PERIOD"].between("2007-10-01", "2008-09-01") |
#     df["TIME_PERIOD"].between("2021-04-01", "2023-10-01") |
#     df["TIME_PERIOD"].between("2026-03-01", "2026-09-01")
# )

# df["LevelShift"] = 0
# df.loc[mask, "LevelShift"] = 0.03 / 12
# dateShift = df.index[mask]

# Flag level shifts (random, Bernoulli)
rng = np.random.default_rng(42)
p = 0.05  # Probability of a level shock each month
delta = 0.03 / 12

# Random level shifts
df["LevelShift"] = delta * rng.binomial(
    n=1,
    p=p,
    size=len(df)
)

# Dates with shocks
mask = df["LevelShift"] != 0
dateShift = df.index[mask]

# Start MC 
K = 1000

# Parameters
mu = 9.36e-3
rho = 0.481
sigE = 1.73e-3
sigU = 0.043540 
F = 0.950284
x0 = [mu, rho, F, sigE, sigU] # True parameters values
N = len(x0)

# State innovations
epsT = rng.normal(0, sigE, size=(T,K)) # T.K tirages

# Model A: time-varying constant (pi_t = mu_t + rho * pi_{t-1} + e_t)
D = np.full((T, K), np.nan)
piT = np.full(T, np.nan)
muT = np.full(T, np.nan)

muT[0] = mu
piT[0] = mu

for k in range(K):
    for t in range(1, T):

        if t in dateShift:
            muT[t] = mu + df.iloc[t]["LevelShift"]
        else:
            muT[t] = mu

        piT[t] = muT[t] + rho * piT[t-1] + epsT[t, k]

    # Store simulation k
    D[:, k] = piT


# Model B: time-varying slope (pi_t = mu + rho_t * pi_{t-1} + e_t)
# Initial parameters taken from OLS AR(1) : pi_t = mu + rho * pi_{t-1} + e_t

F = 0 # O is RW, [-0.99 ; 0.99] is autocorrelated
sigU = 1

x0bis = np.full((N,K), np.nan)

for k in range(K):
    inflation = D[:,k]
    Y = inflation[1:]
    X = sm.add_constant(inflation[:-1])
    model = sm.OLS(Y, X).fit()
    resid = model.resid
    x0bis[:,k] = [
        model.params[0],
        model.params[1], 
        F,
        resid.std(),
        sigU
    ]
    

# Dataframe des paramètres DGP A estimés sur A simulé par OLS (K estimations)
x0bisDf = pd.DataFrame(x0bis, ["hatMu", "hatRhoBar","hatF","hatSigEps","hatSigU"]).transpose()

# Summary statistics

olsStatA = pd.DataFrame({
    "Mean": x0bisDf.mean(),
    "Median": x0bisDf.median(),
    "Mode": x0bisDf.apply(
            lambda x: (
                lambda counts, bins: (
                    bins[np.argmax(counts)] + bins[np.argmax(counts) + 1]
                ) / 2
            )(*np.histogram(x.dropna(), bins=30))
        ),
    "Variance": x0bisDf.var(),
    "Q05": x0bisDf.quantile(0.05),
    "Q95": x0bisDf.quantile(0.95)
})

print(f"\n--- OLS Statistics : constant AR(1) with drift ---")
print(olsStatA)

# On garde la moyenne des paramètres via OLS pour initialiser le MLE
x0Init = x0bis.mean(axis=1)
thetaEst = np.full((N,K), np.nan)
logLikEst = np.full(K, np.nan)
rhoEst = np.full((T-1,K), np.nan)
piEst = np.full((T-1,K), np.nan)
filtVarState = np.full((T-1,K), np.nan)

for k in range(K):
    # Optimisation (MLE)
    inflation = D[:,k]
    estimation = kalAlgoSimple.optNegLogLike(
            x0Init,
            inflation,
            cMu=True,
            cRho=True
        )
    
    # Stockage
    thetaEst[:,k] = estimation.x
    logLikEst[k] = -estimation.fun
    
    # Filtrage
    filtered = kalAlgoSimple.kalman_filter_inflation(
        estimation.x,
        inflation,
        cMu=True,
        cRho=True
    )
    rhoEst[:,k] = filtered["Rho"]
    piEst[:,k] = filtered["FittedY"]
    filtVarState[:,k] = filtered["filteredVarState"]
    
    # Smoothing
    smoothed = kalAlgoSimple.kalman_smoother(filtered, estimation.x)
    

# Estimations results (N*K) for each simulation
thetaDfEst = pd.DataFrame(thetaEst.transpose(), columns=["hatMu", "hatRhoBar","hatF","hatSigEps","hatSigU"])
thetaDfEst["hatRhoMean"] = np.mean(rhoEst, axis=0) # Average estimated persistance over T for each K 
thetaDfEst["hatConst"] = np.mean((1-rhoEst)*thetaDfEst["hatMu"].to_numpy(), axis=0) # Average estimated persistance over T for each K 

# Summary statistics
thetaStats = pd.DataFrame({
    "Mean": thetaDfEst.mean(),
    "Median": thetaDfEst.median(),
    "Mode": thetaDfEst.apply(
            lambda x: (
                lambda counts, bins: (
                    bins[np.argmax(counts)] + bins[np.argmax(counts) + 1]
                ) / 2
            )(*np.histogram(x.dropna(), bins=30))
        ),
    "Variance": thetaDfEst.var(),
    "Q05": thetaDfEst.quantile(0.05),
    "Q95": thetaDfEst.quantile(0.95)
})
print(thetaStats)

# Plot distributions
for col in thetaDfEst.columns:

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        thetaDfEst[col].dropna(),
        bins=30,
        density=False,
        color="lightsteelblue",
        edgecolor="white",
        alpha=0.7
    )

    # Mean
    ax.axvline(
        thetaStats.loc[col, "Mean"],
        color="darkred",
        linewidth=2,
        label=f"Mean: {thetaStats.loc[col, 'Mean']:.4f}"
    )

    # Median
    ax.axvline(
        thetaStats.loc[col, "Median"],
        color="darkblue",
        linestyle="--",
        linewidth=2,
        label=f"Median: {thetaStats.loc[col, 'Median']:.4f}"
    )
    
    # Mode
    ax.axvline(
        thetaStats.loc[col, "Mode"],
        color="orange",
        linestyle="dotted",
        linewidth=2,
        label=f"Mode: {thetaStats.loc[col, 'Mode']:.4f}"
    )

    # Quantiles
    for q, label in [("Q05", "5%"), ("Q95", "95%")]:
        ax.axvline(
            thetaStats.loc[col, q],
            color="darkgrey",
            linestyle=":",
            linewidth=2,
            label=f"Q{label}: {thetaStats.loc[col, q]:.4f}"
        )

    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel("Estimated parameter")
    ax.set_ylabel("Number of simulations")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# Maintenant, plot Rho, Rho moyen et l'IC autour (on sait que le véritable DGP est un level shift avec Rho fixe, mais à quel point sommes-nous loin ?)
rhoMean = []
rhoMed = []
rhoQ05 = []
rhoQ95 = []

for t in range(T-1):
    rhoMean.append(np.mean(rhoEst[t,:]))
    rhoMed.append(np.median(rhoEst[t,:]))
    rhoQ05.append(np.quantile(rhoEst[t,:],0.05))
    rhoQ95.append(np.quantile(rhoEst[t,:],0.95))

# True DGP and Monte Carlo parameter estimates
rhoMC = thetaDfEst["hatRhoMean"].mean()

# Dates
dates = pd.to_datetime(df["TIME_PERIOD"][1:]).to_numpy()
strDate = pd.to_datetime(df.loc[mask, "TIME_PERIOD"])

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 6))

# Background: level-shift periods
for t in strDate:
    ax.axvspan(
        t,
        t + pd.DateOffset(months=1),
        color="lightcoral",
        alpha=0.15,
        linewidth=0
    )

# Empirical 90% interval
ax.fill_between(
    dates,
    rhoQ05,
    rhoQ95,
    color="steelblue",
    alpha=0.2,
    label="MC quantiles [5%, 95%]"
)

# Mean estimated persistence
ax.plot(
    dates,
    rhoMean,
    color="darkblue",
    linewidth=1.8,
    label=r"Mean estimated $\hat{\rho}_t$"
)

# Median estimated persistence
ax.plot(
    dates,
    rhoMed,
    color="darkred",
    linewidth=1.8,
    label=r"Median estimated $\hat{\rho}_t$"
)

# True DGP persistence
ax.axhline(
    rho,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=rf"True DGP $\rho_0$ = {rho:.3f}"
)

# Settings
ax.set_title("Monte Carlo Estimates of Inflation Persistence")
ax.set_ylabel(r"Persistence ($\rho$)")
ax.set_xlabel("")
ax.legend(frameon=False, loc="best")
ax.grid(alpha=0.15)

plt.tight_layout()
plt.show()


# Plot piT simulé, rhoT estimé
piTmean = D[1:,:].mean(axis=1)
df["muT"] = mu + df["LevelShift"].fillna(0)

fig, axes = plt.subplots(
    2, 1,
    figsize=(16, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# --- 1. Simulated inflation and intercept ---
axes[0].plot(
    dates, piTmean * 100,
    color="black",
    linewidth=1.3,
    label=r"Average simulated inflation $\pi_t$"
)

axes[0].plot(
    dates, df["muT"][1:] * 100,
    color="darkred",
    linewidth=1.5,
    label=r"Simulated intercept $\mu_t$"
)

axes[0].set_ylabel("Inflation / Intercept (%)")
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.2)

# --- 2. Estimated persistence ---
axes[1].plot(
    dates, rhoMean,
    color="darkblue",
    linewidth=1.5,
    label=r"Average filtered persistence $\hat{\rho}_t$"
)

axes[1].axhline(
    rho,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label=rf"True persistence $\rho_0$ = {rho:.3f}"
)

axes[1].set_ylabel("Persistence")
axes[1].legend(frameon=False)
axes[1].grid(alpha=0.2)

# --- Level shifts (background) ---
for ax in axes:
    for t in strDate:
        ax.axvspan(
            t,
            t + pd.DateOffset(months=1),
            color="lightcoral",
            alpha=0.15,
            linewidth=0
        )

axes[1].set_xlabel("Date")
fig.suptitle(
    f"Simulated Inflation and Estimated Persistence — MC Simulation {k+1}",
    fontsize=14
)

plt.tight_layout()
plt.show()


# Comparaison entre les IC estimés théoriques et empiriques
rhoQ05FiltEst = []
rhoQ95FiltEst = []

for t in range(T-1):
    std = np.sqrt(filtVarState[t,:].mean())
    rhoQ05FiltEst.append(rhoMean[t]-1.645*std)
    rhoQ95FiltEst.append(rhoMean[t]+1.645*std)

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 6))

# Background: level-shift periods
for t in strDate:
    ax.axvspan(
        t,
        t + pd.DateOffset(months=1),
        color="lightcoral",
        alpha=0.15,
        linewidth=0
    )

# Empirical 90% interval
ax.fill_between(
    dates,
    rhoQ05,
    rhoQ95,
    color="steelblue",
    alpha=0.2,
    label="MC quantiles [5%, 95%]"
)

# Theoretical estimated 90% interval
ax.fill_between(
    dates,
    rhoQ05FiltEst,
    rhoQ95FiltEst,
    color="grey",
    alpha=0.2,
    label=r"Estimated quantiles $IC_t^{90} = \pm \bar \hat \rho_{t|t} - 1.645 \sqrt{P_{t|t}^{\xi}}$"
)

# Mean estimated persistence
ax.plot(
    dates,
    rhoMean,
    color="darkblue",
    linewidth=1.8,
    label=r"Mean estimated $\hat{\rho}_t$"
)

# Median estimated persistence
ax.plot(
    dates,
    rhoMed,
    color="darkred",
    linewidth=1.8,
    label=r"Median estimated $\hat{\rho}_t$"
)

# True DGP persistence
ax.axhline(
    rho,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=rf"True DGP $\rho_0$ = {rho:.3f}"
)

# Settings
ax.set_title("Monte Carlo Estimates of Inflation Persistence")
ax.set_ylabel(r"Persistence ($\rho$)")
ax.set_xlabel("")
ax.legend(frameon=False, loc="best")
ax.grid(alpha=0.15)

plt.tight_layout()
plt.show()


# Les 2 modèles sont-ils égaux ? 
muMean = thetaStats["Mean"]["hatMu"] # Modèle B                  
rhoTlevel = rho + (muT[1:] - muMean)/X[:,1] # Rho_t = Rho + (mu_t - mu) / pi_{t-1}

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 6))

# Background: level-shift periods
for t in strDate:
    ax.axvspan(
        t,
        t + pd.DateOffset(months=1),
        color="lightcoral",
        alpha=0.15,
        linewidth=0
    )

# Mean estimated persistence
ax.plot(
    dates,
    rhoMean,
    color="darkblue",
    linewidth=1.8,
    label=r"Mean estimated $\hat{\rho}_t$"
)

# Median estimated persistence
ax.plot(
    dates,
    rhoTlevel,
    color="darkred",
    linewidth=1.8,
    label=r"$\bar \hat{\rho}_t \approx \rho + (\mu_t - \hat \mu) / \pi_{t-1}$"
)

# True DGP persistence
ax.axhline(
    rho,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=rf"True DGP $\rho_0$ = {rho:.3f}"
)

# Settings
ax.set_title("Monte Carlo Estimates of Inflation Persistence")
ax.set_ylabel(r"Persistence ($\rho$)")
ax.set_xlabel("")
ax.legend(frameon=False, loc="best")
ax.grid(alpha=0.15)

plt.tight_layout()
plt.show()


# Sur données simulées, on regarde le niveau moyen, la persistance moyenne, glissante
# --- Rolling OLS settings ---
window = 60

muRolling = np.full((T, K), np.nan)
rhoRolling = np.full((T, K), np.nan)

# --- Estimate rolling AR(1) for each simulation ---
for k in range(K):

    inflation = D[:, k]

    for t in range(window, T):

        Y = inflation[t-window+1:t+1]
        X = sm.add_constant(inflation[t-window:t])

        model = sm.OLS(Y, X).fit()

        muRolling[t, k] = model.params[0]
        rhoRolling[t, k] = model.params[1]

# --- Monte Carlo statistics ---
muMean = np.nanmean(muRolling, axis=1)
rhoMean = np.nanmean(rhoRolling, axis=1)

muQ05 = np.nanquantile(muRolling[window:], 0.05, axis=1)
muQ95 = np.nanquantile(muRolling[window:], 0.95, axis=1)

rhoQ05 = np.nanquantile(rhoRolling[window:], 0.05, axis=1)
rhoQ95 = np.nanquantile(rhoRolling[window:], 0.95, axis=1)

# --- Dates ---
dates = pd.to_datetime(df["TIME_PERIOD"])
plotDates = dates.iloc[window:]

# --- Plot ---
fig, axes = plt.subplots(
    2, 1, figsize=(16, 9), sharex=True
)

# Intercept
axes[0].plot(
    plotDates, muMean[window:],
    color="darkblue",
    label=r"Rolling $\hat{\mu}_t$"
)

axes[0].fill_between(
    plotDates, muQ05, muQ95,
    color="steelblue", alpha=0.2,
    label="MC 5–95% quantiles"
)

axes[0].plot(
    dates,
    mu + df["LevelShift"].fillna(0).to_numpy(),
    color="black",
    linestyle="--",
    label=r"True $\mu_t$"
)

axes[0].set_ylabel(r"Intercept $\hat{\mu}_t$")
axes[0].legend(frameon=False)

# Persistence
axes[1].plot(
    plotDates, rhoMean[window:],
    color="darkblue",
    label=r"Rolling $\hat{\rho}_t$"
)

axes[1].fill_between(
    plotDates, rhoQ05, rhoQ95,
    color="steelblue", alpha=0.2,
    label="MC 5–95% quantiles"
)

axes[1].axhline(
    rho,
    color="black",
    linestyle="--",
    label=rf"True $\rho={rho:.3f}$"
)

axes[1].set_ylabel(r"Persistence $\hat{\rho}_t$")
axes[1].legend(frameon=False)

# Level shift periods
for ax in axes:
    for t in strDate:
        ax.axvspan(
            t,
            t + pd.DateOffset(months=1),
            color="lightcoral",
            alpha=0.15,
            linewidth=0
        )

plt.tight_layout()
plt.show()