# 01 - Project Architecture

## Overview
The Qmutez project is structured as a scalable, production-ready machine learning framework. The architecture is explicitly designed to separate data ingestion, simulation logic, and model training, allowing smooth transitions from local development to cloud deployment.

## Directory Structure
```text
RideShare/
├── config/                 # YAML configuration (Environment toggling)
├── data/
│   ├── raw/                # Immutable storage for downloaded datasets
│   ├── processed/          # Cleaned datasets ready for simulation
│   └── simulated/          # Generated Driver-Rider features (Model input)
├── docs/                   # Modular project documentation
├── notebooks/              # Jupyter notebooks for visual EDA and trade-off analysis
├── src/                    # Core source code modules
│   ├── data/               # Ingestion and cleaning scripts
│   ├── simulation/         # BallTree spatial pairing and feature logic
│   └── models/             # Benchmark and advanced ML frameworks
├── tests/                  # Pytest unit tests
└── environment.yml         # Conda dependency exact locking
```

## Architectural Philosophy
1.  **Immutability:** Raw data (`data/raw/`) is never modified directly. All cleaning scripts output to `data/processed/`.
2.  **Configuration Driven:** Hardcoded thresholds (e.g., maximum detour constraints, sampling ratios) are strictly forbidden within the Python scripts. All parameters are injected via `config/dev_config.yaml` or `config/prod_config.yaml`.
3.  **Simulation Abstraction:** Because native peer-to-peer carpooling data is unavailable, the `src/simulation/` module acts as an isolated sandbox to generate viable features algorithmically before models ever see the data.
