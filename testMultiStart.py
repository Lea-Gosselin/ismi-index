from fismi.io import load_input, load_config
import numpy as np
import pandas as pd
from fismi.models import kalAlgoRw
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from fismi.maths import rmse, rolling_largest_ar_root, localProjections
from scipy.optimize import minimize
import statsmodels.api as sm 
import statistics as stat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import warnings

########################################################
# Settings
# Select country
country = "US"
cMu = True
cRho = True

print(f"\n--- {country} ---")
config = load_config(country)
startDate = config["persistence"]["startDate"]
prices, weights = load_input(country)

inflation = np.log(
    prices["HEADLINE"].dropna()
).diff(
    config["persistence"]["inflation_lag"]
).dropna()

# Resize...
inflation = inflation[
    inflation.index >= startDate
]

# Dates
dates = inflation.index


########################################################
# Persistance et constante glissantes...

# Sur données simulées, on regarde le niveau moyen, la persistance moyenne, glissante
# --- Rolling OLS settings ---
window = 36
T = inflation.shape[0]
muRolling = np.full(T, np.nan)
rhoRolling = np.full(T, np.nan)

# --- Estimate rolling AR(1) for each simulation ---
for t in range(window, T):

    Y = inflation[t-window+1:t+1].values
    X = sm.add_constant(inflation[t-window:t].values)

    model = sm.OLS(Y, X).fit()

    muRolling[t] = model.params[0]
    rhoRolling[t] = model.params[1]

# Plot...
fig, ax1 = plt.subplots(figsize=(16, 5))
ax2 = ax1.twinx()

# Persistence — left axis
ax1.plot(
    dates,
    rhoRolling,
    color="darkblue",
    linewidth=1.5,
    label=r"Rolling persistence $\hat{\rho}_t$"
)

ax1.axhline(
    0, color="grey", linestyle=":", linewidth=1
)
ax1.axhline(
    1, color="grey", linestyle=":", linewidth=1
)

ax1.set_ylabel(r"Persistence ($\hat{\rho}_t$)", color="darkblue")
ax1.tick_params(axis="y", labelcolor="darkblue")

# Intercept — right axis
ax2.plot(
    dates,
    muRolling,
    color="steelblue",
    alpha=0.7,
    label=r"Rolling intercept $\hat{c}_t$"
)

ax2.set_ylabel("AR(1) intercept (pp)", color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2)
ax1.grid(alpha=0.25)

plt.tight_layout()
plt.show()


########################################################
# Les changements de persistance sont-ils robustes à un changement de fenêtre ? 
fig, ax = plt.subplots(figsize=(16, 5))

for window in [36, 60, 120]:

    rho = np.full(len(inflation), np.nan)

    for t in range(window, len(inflation)):
        Y = inflation[t-window+1:t+1].values
        X = sm.add_constant(inflation[t-window:t].values)

        model = sm.OLS(Y, X).fit()
        rho[t] = model.params[1]

    ax.plot(
        dates,
        rho,
        label=f"{window//12}-year rolling AR(1)",
        linewidth=1.5
    )

ax.axhline(0, color="grey", linestyle=":")
ax.axhline(1, color="grey", linestyle=":")

ax.set_ylabel(r"Rolling persistence ($\hat{\rho}_t$)")
ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


########################################################
# Quel processus ARMA suit l'inflation mensuelle ?

# Best ARMA(p,q) process
# Maximum orders
p_max = 12
q_max = 6
results = []

for p, q in product(range(p_max + 1), range(q_max + 1)):

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = ARIMA(
                inflation,
                order=(p, 0, q),
                trend="c",
                enforce_stationarity=True,
                enforce_invertibility=True
            ).fit()

        if not model.mle_retvals.get("converged", False):
            continue

        results.append({
            "p": p,
            "q": q,
            "AIC": model.aic,
            "BIC": model.bic,
            "LogLik": model.llf
        })

    except (ValueError, np.linalg.LinAlgError):
        continue

dfResults = pd.DataFrame(results)

if dfResults.empty:
    raise RuntimeError("No ARMA model converged.")

# Best specifications
bestAIC = dfResults.loc[dfResults["AIC"].idxmin()]
bestBIC = dfResults.loc[dfResults["BIC"].idxmin()]

print("Best ARMA according to AIC:")
print(bestAIC)

print("\nBest ARMA according to BIC:")
print(bestBIC)

# Top 10 specifications
display(dfResults.nsmallest(10, "BIC"))

# Visualiser les critères...
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

for ax, criterion in zip(axes, ["AIC", "BIC"]):

    pivot = dfResults.pivot(
        index="p",
        columns="q",
        values=criterion
    )

    im = ax.imshow(
        pivot,
        origin="lower",
        aspect="auto",
        cmap="viridis"
    )

    ax.set_xticks(range(q_max + 1))
    ax.set_yticks(range(p_max + 1))
    ax.set_xlabel("MA order (q)")
    ax.set_ylabel("AR order (p)")
    ax.set_title(criterion)

    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()


########################################################
# Estimer le meilleur model (parcimonieux) et tester les résidus
p_opt = int(bestBIC["p"])
q_opt = int(bestBIC["q"])

bestModel = ARIMA(
    inflation,
    order=(p_opt, 0, q_opt),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
).fit()

print(bestModel.summary())

# Residual diagnostics
bestModel.plot_diagnostics(figsize=(13, 8))
plt.tight_layout()
plt.show()

# Companion eigenvalues are inverse AR polynomial roots
ar_roots = bestModel.arroots

if len(ar_roots) > 0:
    rho_max = 1 / np.min(np.abs(ar_roots))
else:
    rho_max = 0.0

print(f"Selected ARMA({p_opt},{q_opt})")
print(f"Largest AR eigenvalue modulus: {rho_max:.4f}")

# A partir de l'ARMA identifié, LAR comme estimation de la persistance du processus, sur une fenêtre de 10 ans
results = rolling_largest_ar_root(
    inflation*100,
    window=120,
    p=p_opt
)

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    results.index,
    results["LargestRoot"],
    color="darkblue",
    linewidth=1.8,
    label="Largest AR root modulus"
)

ax.axhline(
    1,
    color="darkred",
    linestyle="--",
    label="Unit root"
)

ax.set(
    title=f"US Inflation Persistence — Rolling {window/12:.0f}-year AR({p_opt})",
    ylabel="Largest autoregressive root",
    xlabel=""
)

ax.legend(frameon=False)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()


########################################################
# Y a t'il preuve de changement de moyenne post-Covid ? 
# Covid dummy
covidDummy = pd.Series(
    (inflation.index >= pd.Timestamp("2020-03-01")).astype(int),
    index=inflation.index,
    name="PostCovid"
)

# Model with Covid dummy
model1 = ARIMA(
    inflation,
    exog=covidDummy,
    order=(p_opt, 0, q_opt),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
).fit()

print(model1.summary())


########################################################
# Le changement de niveau s'est-il résorbé ?
# Hypothesized break dates
T1 = "2021-04-01"
T2 = "2024-01-01"

exog = pd.DataFrame(index=inflation.index)

exog["D1"] = (
    inflation.index >= pd.Timestamp(T1)
).astype(int)

exog["D2"] = (
    inflation.index >= pd.Timestamp(T2)
).astype(int)

model = ARIMA(
    inflation,
    exog=exog,
    order=(p_opt, 0, q_opt),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
).fit()

print(model.summary())

# Mean shift after the second break
delta_post = model.params["D1"] + model.params["D2"]

# Variance of the linear combination
cov = model.cov_params()

var_post = (
    cov.loc["D1", "D1"]
    + cov.loc["D2", "D2"]
    + 2 * cov.loc["D1", "D2"]
)

se_post = np.sqrt(var_post)

# Wald test: H0 = delta1 + delta2 = 0
z_stat = delta_post / se_post
p_value = 2 * norm.sf(abs(z_stat))

ci_low = delta_post - 1.96 * se_post
ci_high = delta_post + 1.96 * se_post

print(f"Post-period mean shift: {delta_post:.3f} pp")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Wald test p-value: {p_value:.4f}")


########################################################
# Multi start
rng = np.random.default_rng(42)
inflation = inflation.to_numpy()

runs = []

for i in range(100):
    x0 = np.array([
        rng.uniform(-0.01, 0.01),       # mu
        rng.uniform(-0.99, 0.99),       # rho_bar
        rng.uniform(1e-6, 0.5),        # D
        rng.uniform(1e-6, 0.5)         # B
    ])

    try:
        res = kalAlgoRw.optNegLogLike(
            x0,
            inflation,
            cMu=cMu,
            cRho=cRho
        )

        nll = kalAlgoRw.kalman_filter_inflation(
            res.x,
            inflation,
            cMu=cMu,
            cRho=cRho,
            return_loglik=True
        )

        runs.append({
            "run": i,
            "success": res.success,
            "message": res.message,
            "objective": res.fun,
            "loglik": -res.fun,
            "mu": res.x[0],
            "rho_bar": res.x[1],
            "D": res.x[2],
            "B": res.x[3],
            "lambda": (res.x[2] / res.x[3]) ** 2
        })

    except (ValueError, FloatingPointError, OverflowError):
        continue

dfMultiStart = pd.DataFrame(runs)

dfMultiStart = dfMultiStart.sort_values(
    "objective"
).reset_index(drop=True)


########################################################
# Do several different initial parameters lead to the same NLL ?
successful = dfMultiStart.loc[(dfMultiStart["success"]==True)]

# Plot the distribution of each parameter
params = ["mu", "rho_bar", "D", "B", "lambda"]

# Best solution according to the objective function
best = successful.loc[successful["objective"].idxmin()]

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 8),
    constrained_layout=True
)

for ax, c in zip(axes.flat, params):

    values = successful[c].dropna()

    ax.hist(
        values,
        bins=30,
        color="skyblue",
        edgecolor="black",
        alpha=0.7
    )

    ax.axvline(
        best[c],
        color="darkred",
        linestyle="--",
        linewidth=1.5,
        label="Best optimum"
    )

    ax.set_title(c.upper(), fontsize=13)
    ax.set_xlabel("Estimated value")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.show()


# Plot some of the successful results...
params = ["mu", "rho_bar", "D", "B"]
x0 = successful[params].iloc[0].values

result = kalAlgoRw.kalman_filter_inflation(
    x0,
    inflation,
    cMu=cMu,
    cRho=cRho,
    return_loglik=False
)

df = pd.DataFrame(result)
df.index = dates[-df.shape[0]:]

# 1. Observed vs fitted inflation
rMse = rmse(df["Y"], df["FittedY"])
fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(df.index, df["Y"] * 100,
        color="darkgrey", label="Observed inflation")
ax.plot(df.index, df["FittedY"] * 100,
        color="darkblue", label=f"Fitted inflation ({rMse*100:.2f} pp)")

ax.set_ylabel("Inflation (%)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# 2. Time-varying persistence and confidence interval
fig, ax = plt.subplots(figsize=(16, 5))
alpha=0.90
z = norm.ppf(alpha)  # 90% bilateral confidence interval

RhoFiltered = df["Rho"]
std = np.sqrt(df["forecastVarState"])

df["RhoLower90"] = (RhoFiltered) - z * std
df["RhoUpper90"] = (RhoFiltered) + z * std

ax.fill_between(
    df.index,
    df["RhoLower90"],
    df["RhoUpper90"],
    color="grey",
    alpha=0.20,
    linewidth=0,
    label=f"{alpha*100:.0f}% confidence interval"
)

ax.plot(df.index, RhoFiltered,
        color="orange",
        label=r"Filtered $\rho_t$")

ax.axhline(x0[1], color="red", linestyle="--", linewidth=1)

ax.set_ylabel(r"Persistence ($\rho_t$)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Probability that persistence exceeds its long-run mean
rhoBar = x0[1]
z = (df["Rho"] - 1) / np.sqrt(df["filteredVarState"])
probHighPersistence = norm.cdf(z)
df["probHighPersistence"] = probHighPersistence

fig, axes = plt.subplots(
    2, 1, figsize=(14, 7), sharex=True
)

# Filtered persistence
axes[0].plot(
    df.index, df["Rho"],
    color="darkblue", label="Filtered persistence"
)

axes[0].axhline(
    rhoBar, color="black", linestyle="--",
    label="Long-run persistence"
)

axes[0].set_ylabel("Persistence")
axes[0].legend()
axes[0].grid(alpha=0.2)

# Probability
axes[1].plot(
    df.index, df["probHighPersistence"],
    color="darkorange",
    label=r"$P(\hat \rho_{t|t} > 1)$"
)

axes[1].axhline(0.90, color="red", linestyle="--")
axes[1].axhline(0.10, color="green", linestyle="--")
axes[1].axhline(0.50, color="grey", linestyle=":")

axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Probability")
axes[1].legend()
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.show()


# Plot with ISMI...
dfIsmi = pd.read_parquet(f"/Users/lea_gosselin/GitRepo/ismi-index/results/{country}Ismi.parquet")
df["ISMI"] = dfIsmi["ISMI"].reindex(df.index)

df["InflationRegime"] = np.select(
    [
        (df["ISMI"] > 0) & (df["probHighPersistence"] >= 0.90),
        (df["ISMI"] > 0) & (df["probHighPersistence"] <= 0.10),
        (df["ISMI"] < 0) & (df["probHighPersistence"] >= 0.90),
        (df["ISMI"] < 0) & (df["probHighPersistence"] <= 0.10)
    ],
    [
        "Persistent inflationary pressures",
        "Transitory inflationary pressures",
        "Persistent disinflationary pressures",
        "Transitory disinflationary pressures"
    ],
    default="Uncertain"
)



# Constante... Time-varying...
cFiltered = x0[0]*(1-RhoFiltered)

fig, ax = plt.subplots(
    figsize=(16, 8)
)
ax.plot(df.index, df["Y"] * 100,
            color="darkgrey", label="Monthly inflation")
ax.plot(df.index, cFiltered * 100,
            color="orange", label=r"Filtered time-varying constant $c_{t|t-1} = \mu (1-\rho_{t|t-1})$")

ax.legend()
ax.grid(alpha=0.3)


# Etude des erreurs...
errors = df["Err"].to_numpy()

print("Mean innovation:", np.mean(errors))
print("Innovation std:", np.std(errors))
print("RMSE forecast:", rmse(df["Y"], df["ForecastY"]))

fig, ax = plt.subplots(figsize=(16, 4))

ax.plot(df.index, errors * 100, color="darkblue")
ax.axhline(0, color="black", linestyle="--")
ax.axhline(np.mean(errors) * 100,
           color="red", linestyle=":",
           label="Mean innovation")

ax.set_ylabel("Forecast error (pp)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# Standardized one-step-ahead Kalman innovations
z = (
    np.asarray(df["Err"])
    / np.sqrt(np.asarray(df["forecastVarY"]))
)

# ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(z, lags=36, ax=axes[0])
axes[0].set_title("Standardized KF Innovations (ACF)")

plot_pacf(z, lags=36, ax=axes[1], method="ywm")
axes[1].set_title("Standardized KF Innovations (PACF)")

axes[0].grid(alpha=0.3)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()



### NLL et delta NLL
valid = successful.sort_values("objective")
valid = valid[valid["objective"] < 0]
valid["delta_nll"] = valid["objective"] - best["objective"]
print("Descriptive statistics of valid NLL")
print(valid["delta_nll"].describe())

for threshold in [1e-4, 1e-2, 0.1, 1, 5]: # How many of the K draws lead to almost identical (up to a threshold) NLL ?
    n = (valid["delta_nll"] <= threshold).sum()
    print(f"Delta NLL <= {threshold} (threshold) : {n}/{len(valid)} identical likelihood")

# Plot the likelihood profile
fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(
    np.arange(len(valid)),
    valid["delta_nll"],
    s=12,
    alpha=0.7
)

ax.set_yscale("symlog", linthresh=1e-4)
ax.set_ylabel("Difference from best NLL")
ax.set_xlabel("Ranked optimization runs")
ax.grid(alpha=0.3)

plt.show()

### Do best solutions converge to the same parameters ?
# For several different value, some of the NLL obtained are identical. Do they lead to the same optimal parameters ?
near = valid.loc[
    valid["delta_nll"] <= 0.01
].copy()

print("\nDispersion among best solutions:")
print(
    near[params].agg(["min", "max", "std"]).T
)




# Test profilé
names = ["mu", "rho_bar", "D", "B"]
N = len(names) # Number of parameters
K = 40

bounds = [
    (-0.01, 0.01),
    (-0.99, 0.99),
    (0.0001, 0.05),
    (1e-4, 1) #     (1e-6, 0.5)
]

# Without any constraint... Run KL and MLE 
def nll(theta):
    return kalAlgoRw.kalman_filter_inflation(
        theta, inflation,
        cMu=True, cRho=True,
        return_loglik=True
    )

# For each parameter j
def profile_parameter(j, grid, theta_start):

    other = [k for k in range(N) if k != j] # Index of all other parameters (that are kept fixed)
    bounds_other = [bounds[k] for k in other]

    rows = []

    # We compute the NLL keeping all parameters but the j-iest fixed. Parameter j 
    for value in grid:

        def objective_profile(x):
            theta = np.empty(5)
            theta[j] = value
            theta[other] = x
            return nll(theta)

        res = minimize(
            objective_profile,
            theta_start[other],
            method="L-BFGS-B",
            bounds=bounds_other
        )

        rows.append({
            "value": value,
            "nll": res.fun,
            "success": res.success
        })

    return pd.DataFrame(rows)

# Best admissible solution, in parameter order
theta_hat = valid.iloc[0][names].to_numpy(dtype=float)

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

profile = {}
xFinal = []
for j, name in enumerate(names):
    # For each parameter (Mu, RhoBar...) we define 
    grid = np.linspace(
        bounds[j][0],
        bounds[j][1],
        K
    )

    profile[name] = profile_parameter(j, grid, theta_hat)

    ax = axes.flat[j]

    ax.plot(profile[name]["value"], profile[name]["nll"])
    ax.axvline(theta_hat[j], color="red", linestyle="--")

    ax.set_title(name)
    ax.set_xlabel("Fixed parameter value")
    ax.set_ylabel("Profile negative log-likelihood")
    ax.grid(alpha=0.3)
    
    # Store values corresponding to lowest likelihood
    minVal = profile[name].sort_values(by="nll",ascending=True).iloc[0,0]
    xFinal.append(minVal)

# axes.flat[-1].axis("off")

plt.tight_layout()
plt.show()


