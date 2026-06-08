#!/bin/bash
# Reproduce all results from the paper
# Usage: ./run_all.sh

set -e

echo "Step 1/4: Data ingestion and coordinate synthesis..."
PYTHONPATH=. python src/data/loader.py

echo "Step 2/4: Generating driver-rider pairs..."
PYTHONPATH=. python src/simulation/pair_generator.py

echo "Step 3/4: Training XGBoost + baselines, generating figures..."
PYTHONPATH=. python src/models/xgb_matching.py
PYTHONPATH=. python src/models/baselines.py

echo "Step 4/4: Generating feature distribution plot..."
PYTHONPATH=. python src/visualization/generate_feature_distributions.py

echo ""
echo "Copying figures to report..."
cp figures/*.png report/figures/

echo "All figures are in figures/ and report/figures/"