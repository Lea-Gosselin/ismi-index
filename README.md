# Identifying Inflation Regimes: Evidence from Inflation Shock Momentum and Time-Varying Persistence

**Independent research and implementation by Léa Gosselin**

This repository provides a Python workflow to monitor inflation dynamics through two complementary indicators: the **Inflation Shock Momentum Index (ISMI)** and **time-varying inflation persistence**. It currently supports the **United States (US)**, with a structure intended to accommodate additional economies.

The ISMI implementation replicates and extends the approach of **Lansing and Shapiro (2026)** to country-specific inflation components. The persistence estimates are obtained from state-space models with time-varying autoregressive coefficients. Together, the indicators help describe the breadth, direction, and persistence of inflationary pressures; they do **not**, by themselves, establish a causal classification of inflation regimes.

> **Research disclaimer:** The data transformations, estimates, charts, and interpretations presented in this repository and its dashboard are independently produced. Any errors are the author's responsibility. They should not be attributed to the original researchers or their institutions.

## Interactive dashboard

The dashboard displays two charts for each supported economy:

1. **ISMI and inflation persistence**, including the estimated long-run persistence parameter and inflation-threshold shading.
2. **ISMI decomposition and headline inflation**, showing positive momentum, negative momentum, and the aggregate index.

The interface includes **Full**, **10Y**, and **1Y** time ranges and downloadable data.

- **Local dashboard:** `http://localhost:8000/dashboard/`
- **Original ISMI dashboard:** [Lansing and Shapiro — ISMI](https://adamshap.github.io/ismi-webapp/ism_webapp.html)
- **Published dashboard:** [Inflation Monitor | Tracking Inflation Regimes](https://lea-gosselin.github.io/ismi-index/dashboard/)

## 0. Requirements and configuration

### Python environment

- **Python:** 3.11 recommended (the project has been developed with Python 3.11.6).
- **Dependencies:** see `pyproject.toml`.

From the repository root:
```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS and Linux
python3 -m pip install --upgrade pip
python3 -m pip install -e .
pip install --upgrade certifi
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead.

### API credentials

US PCE data are retrieved from the **U.S. Bureau of Economic Analysis (BEA)**. Obtain a BEA API key and create a `.env` file at the repository root:

```dotenv
BEA_API_KEY=your_bea_api_key_here
```

The application reads the key from the environment, for example using `python-dotenv`. Ensure `.gitignore` includes `.env` and `.venv/`.

### Country configurations

Country-specific source settings and model options are stored in `config/EA.yaml` and `config/US.yaml`. Verify the data sources, component classifications, estimation windows, and model options before running a new specification. UK and JP are planned extensions and should not be considered operational until their data collectors and models have been validated.

## 1. Run or update the estimates

All commands below are executed from the repository root.

```bash
# Run both models for both supported economies, using saved input data
python3 run.py --country all

# Update input data and rerun both models
python3 run.py --country all --refresh-data

# Re-estimate only the US ISMI
python3 run.py --country US --step ismi

# Re-estimate only euro area inflation persistence
python3 run.py --country EA --step persistence

# Update only the US input data
python3 run.py --country US --step data
```

| Option | Values | Description |
| --- | --- | --- |
| `--country` | `EA`, `US`, `all` | Economy or economies to process; default: `all`. |
| `--step` | `data`, `ismi`, `persistence`, `all` | Pipeline stage; default: `all`. |
| `--refresh-data` | Flag | Re-download input data before running the selected model stage. |

**Note:** `--refresh-data` is a flag, not a value for `--step`. Without this flag, the pipeline reads previously saved input files.

### Refresh the dashboard data

After running the estimations, regenerate the JSON files used by the website:

```bash
python3 -m fismi.dashboard
```

Preview the dashboard locally:

```bash
python3 -m http.server 8000
```

Then open **http://localhost:8000/dashboard/**. Press **Ctrl+C** in the terminal to stop the server.

If the site is published through GitHub Pages, the updated JSON files must also be committed and deployed (or published by an automated workflow) before visitors see new estimates.

## 2. Repository structure

The layout below illustrates the main project components. Some auxiliary modules may differ as the package evolves.

```text
ismi-index/
├── fismi/                           # Python research package
│   ├── __init__.py
│   ├── config.py                    # Read country YAML configurations
│   ├── data/                        # Data collection and processing
│   │   ├── __init__.py
│   │   ├── ea.py                    # Euro area data collector
│   │   ├── us.py                    # US data collector
│   │   └── processing.py            # Shared transformations and checks
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ismi.py                  # ISMI and positive/negative momentum
│   │   ├── persistence.py           # Common persistence interface
│   │   ├── kalman_simple.py         # Simple state-space model
│   │   └── kalman_augmented.py      # Augmented state-space model
│   ├── io.py                        # Read/write Parquet outputs
│   ├── pipeline.py                  # Pipeline orchestration
│   └── dashboard.py                 # Export dashboard JSON
├── config/
│   ├── EA.yaml
│   └── US.yaml
├── Input/
│   ├── EA/
│   │   ├── IPC.parquet
│   │   └── Weights.parquet
│   └── US/
│       ├── IPC.parquet
│       └── Weights.parquet
├── Results/
│   ├── EAIsmi.parquet
│   ├── USIsmi.parquet
│   ├── EAPersistence.parquet
│   ├── EAPersistence.json
│   ├── USPersistence.parquet
│   └── USPersistence.json
├── dashboard/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── data/
│       ├── EA_dashboard.json
│       └── US_dashboard.json
├── tests/                          # Automated checks (where available)
├── run.py                          # Command-line entry point
├── pyproject.toml
├── .gitignore
└── README.md
```

## 3. Indicators and interpretation

### Inflation Shock Momentum Index (ISMI)

Following [Lansing and Shapiro (2026)](https://www.frbsf.org/wp-content/uploads/wp2026-10.pdf), the Inflation Shock Momentum Index (ISMI) measures the share of PCE categories experiencing consecutive months of positive inflation shocks minus the share experiencing consecutive negative shocks. The index provides a real-time measure of inflation momentum across different product categories. The underlying inflation measure and component classification differ by economy: **PCE for the US** and **HICP for the euro area**.

### Time-varying inflation persistence

A state-space framework estimates the evolution of inflation persistence over time. The dashboard displays the filtered coefficient \(\rho_t\) alongside its estimated long-run reference value \(\bar\rho\). Interpretation depends on the model specification, sample, inflation transformation, and parameter restrictions selected in each country's configuration.

The dashboard is descriptive: differences in ISMI and persistence should not automatically be interpreted as structural changes or as evidence of a particular monetary-policy response.

## 4. Reproducibility and data notes

- **Sources:** BEA PCE data (US) and ECB HICP data (EA).
- **Frequency:** Monthly.
- **Data coverage:** Depends on source availability and the estimation windows configured for each economy.
- **Component coverage:** Some expenditure categories are introduced or revised over time. Availability and weighting should be checked when reproducing the ISMI.
- **Numerical comparability:** Log-likelihoods and persistence estimates depend on the inflation scale, sample, and model specification.
- **Public release:** Review the redistribution conditions of source data and the contents of exported JSON files before publishing.

## References

1. Lansing, K. J., & Shapiro, A. H. (2026). *Measuring inflation shock momentum*. Federal Reserve Bank of San Francisco Working Paper 2026-10.
2. Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become harder to forecast? *Journal of Money, Credit and Banking*.
3. Beechey, M., & Österholm, P. (2007). *The rise and fall of U.S. inflation persistence*. Finance and Economics Discussion Series, Board of Governors of the Federal Reserve System.
4. Sbordone, A. M. (2007). Inflation persistence: Alternative interpretations and policy implications. *Journal of Monetary Economics, 54*(5), 1311–1339.
5. Cogley, T., & Sargent, T. J. (2005). Drifts and volatilities: Monetary policies and outcomes in the post-WWII U.S. *Review of Economic Dynamics, 8*(2), 262–302.
6. Carvalho, C., Eusepi, S., Moench, E., & Preston, B. (2023). Anchored inflation expectations. *American Economic Journal: Macroeconomics, 15*(1), 1–47.
7. Cogley, T., & Sargent, T. J. (2001). Evolving post-World War II U.S. inflation dynamics. *NBER Macroeconomics Annual, 16*, 331–373.
8. Chan, J. C. C., Koop, G., & Potter, S. M. (2013). A new model of trend inflation. *Journal of Business & Economic Statistics, 31*(1), 94–106.
9. Chan, J. C. C., Clark, T. E., & Koop, G. (2018). A new model of inflation, trend inflation, and long-run inflation expectations. *Journal of Money, Credit and Banking, 50*(1), 5–53.
10. Pivetta, F., & Reis, R. (2007). The persistence of inflation in the United States. *Journal of Economic Dynamics and Control*.

## Author
Independent research and implementation  
Contact: [lea_gosselin@outlook.fr](mailto:lea_gosselin@outlook.fr)
