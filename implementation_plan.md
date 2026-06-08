# Implementation Plan: RideShare ML Matching Engine

This plan covers the "Incentive-Aware Multi-Objective Matching" project, structured not just as an academic exercise, but as a scalable foundation for the future Qmutez startup. We are prioritizing strict separation of concerns, modularity, and environment reproducibility.

## User Review Required

> [!IMPORTANT]
> **Data Assumptions and Simulation Logic:** Since we do not have real P2P carpooling data, the core of this project relies on securely simulating "Drivers" and "Riders" from single-passenger taxi trips. Please review the *Proposed Architecture > src/simulation/* to ensure the assumptions for generating candidate pairs align with your academic expectations.

## Proposed Architecture & File Structure

We will adopt a standard professional data science project structure that allows seamless transition from development to production.

```text
RideShare/
│
├── config/                 # Environment and module configurations
│   ├── dev_config.yaml     # Dev settings (small data samples, quick training)
│   └── prod_config.yaml    # Prod settings (full dataset, extensive tuning)
│
├── data/                   # Data storage (gitignored)
│   ├── raw/                # Original, immutable NYC Taxi files (.parquet)
│   ├── processed/          # Cleaned, standardized data
│   └── simulated/          # Generated Driver-Rider candidate pairs
│
├── docs/                   # Documentation for future developers
│   ├── api_reference.md    # Documentation of core functions
│   └── simulation_logic.md # Detailed explanation of how pairs are generated
│
├── notebooks/              # Jupyter notebooks for EDA and presentation
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_simulation_analysis.ipynb
│   └── 03_model_evaluation.ipynb
│
├── src/                    # Source code for the application
│   ├── __init__.py
│   ├── data/               # Data fetching and cleaning logic
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── simulation/         # P2P Carpool simulation logic
│   │   ├── __init__.py
│   │   ├── pair_generator.py # Logic for matching O/D pairs
│   │   └── feature_eng.py    # Route overlap/detour calculations
│   └── models/             # ML modeling pipelines
│       ├── __init__.py
│       ├── baselines.py      # Rule-based and Logistic Regression
│       └── xgb_matching.py   # Advanced multi-objective model
│
├── tests/                  # Automated testing suite
│   ├── conftest.py
│   ├── test_loader.py
│   └── test_simulation.py
│
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Dev dependencies (notebooks, pytest)
├── setup.py                # Makes `src` pip installable
└── README.md               # Project overview and run instructions
```

## Environment Setup Strategy

1.  **Dependency Management:** We will use a standard virtual environment (like `venv` or `conda`).
2.  **`requirements.txt`:** Will strictly contain runtime dependencies (e.g., `pandas`, `xgboost`, `scikit-learn`, `pyyaml`).
3.  **`requirements-dev.txt`:** Will contain development tools needed only for building/testing (e.g., `jupyter`, `pytest`, `black`, `flake8`).
4.  **`config.yaml` driven:** The code will read parameters (like bounding box coordinates, time window definitions, max detour thresholds) from YAML files, so we can change simulation rules without editing core `.py` files.

## Development Workflow

### 1. Data Pipeline (`src/data/`)
A module to handle downloading and initial cleaning of the NYC Taxi dataset. We will fetch `.parquet` files (for speed/efficiency) from the NYC TLC website, drop invalid records, and save to `data/processed/`.

### 2. P2P Simulator (`src/simulation/`)
The engine of the project. It reads processed trips, identifies the AM and PM commuting windows, splits profiles into Drivers/Riders, and engineers the candidate pairing features (detour distance, coordinate overlap, time difference, fare ratio, pickup hour/day-of-week, and shareability count). Outputs are saved to `data/simulated/`.

### 3. Modeling (`src/models/`)
Implementation of the initial benchmarks and advanced model, reading only from `data/simulated/`.
- **Rule-Based Baseline:** Threshold-based acceptance.
- **LogReg Baseline:** Simple probabilistic matching.
- **XGBoost:** Multi-objective scoring.

## Verification Plan

### Automated Tests
- Run `pytest tests/` to verify core logic (e.g., geodesic distance calculations, edge cases in time window overlaps) before any model training.
- CI/CD ready: The test structure will be compatible with GitHub Actions for future production deployments.

### Manual / Analytical Verification
- Review the analysis exposed in `notebooks/03_model_evaluation.ipynb` to ensure simulated matches are statistically viable and models meet the academic rubric.
