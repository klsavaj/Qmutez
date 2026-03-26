# Qmutez: Incentive-Aware Multi-Objective Matching

This repository contains the Machine Learning pipeline for the Qmutez peer-to-peer commuter carpooling application.

## Overview
Unlike standard rideshare applications (Uber/Lyft), Qmutez focuses on matching regular commuters, minimizing driver detour while calculating accurate match probabilities and fare distributions.

This codebase uses historical NYC Taxi data to simulate a complex Driver-Rider environment and trains an advanced XGBoost model to perform multi-objective matching.

## Architecture
- `config/`: Contains YAML files to switch between `dev` (small sample) and `prod` (full dataset) modes.
- `src/data/`: Modules to fetch and clean raw TLC Parquet files.
- `src/simulation/`: The core engine that generates training pairs using BallTree spatial indexing, Haversine distance, temporal overlap, and multi-factor stochastic labeling.
- `src/models/`: Contains Rule-Based baselines, Logistic Regression baselines, and the final Multi-Objective XGBoost model.
- `notebooks/`: Jupyter Notebooks demonstrating analytical findings and rendering trade-off curves.

## Getting Started

### 1. Environment Setup
You must use Conda to handle the heavy ML and geographic dependencies.
```bash
conda env create -f environment.yml
conda activate qmutez
```

### 2. Download and Process Data
*Note: The production dataset requires downloading the ~800MB post-COVID August 2025 NYC Taxi Data.*
```bash
conda run -n qmutez python src/data/loader.py
```

### 3. Run the P2P Simulation
Generates candidate Driver/Rider pairs from the cleaned taxi data.
```bash
conda run -n qmutez python src/simulation/pair_generator.py
```

### 4. Evaluate Models
Run the baselines:
```bash
conda run -n qmutez python src/models/baselines.py
```
Run the advanced XGBoost model:
```bash
conda run -n qmutez python src/models/xgb_matching.py
```

Or view the end-to-end analysis visually via the Jupyter Notebook in `notebooks/03_model_evaluation.ipynb`.


---
# HOW TO RUN?

Here is the exact order of the pipeline.

Because we engineered this with strict Separation of Concerns, the Machine Learning architecture requires exactly 3 sequential steps.

Step 1 (loader.py): Data Ingestion & Coordinate Synthesis. (You only run this ONCE to download the 3.5M rows, apply Zonal Bounding Box Distribution with point-in-polygon validation, and generate corridor-aware dropoff bearings).
Step 2 (pair_generator.py): Simulation. (This takes the processed rides and generates all feasible driver-rider training pairs using BallTree spatial indexing).
Step 3 (03_model_evaluation.ipynb or xgb_matching.py): Intelligence. (This feeds the simulated matches into the XGBoost AI).
If you want to run it in DEV mode (Fast Testing / Debugging)
Run these commands in order in your terminal. This will use 

dev_config.yaml
 (which downsamples the data to 5% and only takes seconds to run).

Command 1: (Only run if you haven't downloaded the parquet file yet)

```bash
PYTHONPATH=. conda run -n qmutez python src/data/loader.py
```
Command 2: (Run the Fast Simulation)

```bash
PYTHONPATH=. conda run -n qmutez python src/simulation/pair_generator.py
```
Command 3: (Run the XGBoost Evaluation visually in Jupyter)

```bash
export PYTHONPATH=/Users/krunal/Desktop/RideShare:$PYTHONPATH && conda run -n qmutez --cwd /Users/krunal/Desktop/RideShare jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_evaluation.ipynb
```
### If you want to run it in PROD mode (Server-Side / Massive Data)
When it's time to generate the true, massive dataset with Tens of Thousands of matches, you run these commands. The only difference is that we force Step 2 to use `--config config/prod_config.yaml` (3km NYC-dense thresholds) to unleash the full dataset.

Command 1: (Skip unless you deleted the raw data)

Command 2: (Run the Massive Production Simulation — This will take ~5 minutes!)

```bash
PYTHONPATH=. conda run -n qmutez python -c "from src.simulation.pair_generator import PairGenerator; sim = PairGenerator('data/processed/cleaned_yellow_tripdata_2025-08.parquet', 'config/prod_config.yaml'); sim.filter_commuting_window(); sim.split_drivers_and_riders(); sim.generate_candidate_pairs(limit=25000)"
```

Command 3: (Run the XGBoost AI on the massive dataset)

```bash
export PYTHONPATH=/Users/krunal/Desktop/RideShare:$PYTHONPATH && conda run -n qmutez --cwd /Users/krunal/Desktop/RideShare jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_evaluation.ipynb
```