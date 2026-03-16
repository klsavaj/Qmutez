# 02 - Tech Stack Justification

## 1. Conda (`environment.yml`)
**Why not pip/venv?**
Machine learning libraries (like XGBoost) and geospatial libraries (like `geopy`) rely on highly specific underlying C++ and strict C-compiler dependencies. Standard `pip` implementations frequently fail or suffer performance losses on Mac/Linux environments due to library mismatches. Conda guarantees exact binary reproducibility for data science workflows.

## 2. Apache Parquet (`.parquet`)
**Why not CSV?**
The NYC TLC Taxi Dataset contains millions of rows per month. Storing these as plain-text `.csv` files would require massive disk space and extremely slow read/write I/O bounds. Parquet is a columnar storage format that compresses data heavily and allows pandas to read variables directly into memory partitions up to 10-50x faster than CSVs.

## 3. XGBoost (`XGBRegressor`, `XGBClassifier`)
**Why not Deep Learning or standard Random Forests?**
Deep Learning thrives on unstructured data (images, text) but is often outperformed by Gradient Boosted Trees (like XGBoost) on clean, structured, tabular datasets like ours. We selected an ensemble tree method over linear regression because the relationship between detour distances, overlapping coordinate geometry, and a human driver's "acceptance probability" is highly non-linear.
