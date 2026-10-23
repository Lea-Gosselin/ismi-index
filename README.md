# Identifying Inflation Regimes: Evidence from Inflation Shock Momentum and Time-Varying Persistence

## 0. Settings
- Environment variables
- API Keys
- Requirements
- Python version

## 1. How-to update
```bash
python3 run.py --country EA
python3 run.py --country US
python3 run.py --country all
```

ismi-index/
│
├── fismi/                         # TOOLBOX
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── query.py               # ECB, BEA, ONS, Japan
│   │   ├── processing.py          # Common transformations
│   │   └── validation.py          # Data consistency controls
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ismi.py                # ISMI, ISMI+, ISMI-
│   │   ├── persistence.py         # Common Kalman Filter(s)
│   │   ├── kalman_simple.py
│   │   └── kalman_augmented.py
│   │
│   ├── figures.py                 # Common charts
│   ├── maths.py                   # Common functions for RMSE, local projections... 
│   ├── utils.py                   
│   ├── io.py                      # Saving, loading...
│   └── pipeline.py                # Orchestrator
│
├── Input/
│   ├── EAIPC.parquet
│   ├── EAWeights.parquet
│   ├── USIPC.parquet
│   ├── EAWeights.parquet
│   └── ...
│
├── Results/
│   ├── ISMI/
│   │   ├── EAIsmi.parquet
│   │   ├── USIsmi.parquet
│   │   └── ...
│   └── Persistence/
│       ├── EAPersistence.parquet
│       ├── USPersistence.parquet
│       └── ...
│
├── config/
│   ├── EA.yaml
│   ├── US.yaml
│   ├── UK.yaml
│   └── JP.yaml
│
├── Dashboard.ipynb
├── run.py                        
├── pyproject.toml
├── requirements.txt
└── README.md

# References



