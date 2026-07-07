#!/usr/bin/env python3
import os
import logging
import pandas as pd
from pathlib import Path

# Ensure project src is importable (if running from repo root)
import sys
sys.path.insert(0, os.path.abspath("."))

# Logging so we can see info messages from modules
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_test")

# Create minimal sample data folders
Path("data/walmart").mkdir(parents=True, exist_ok=True)
Path("data/tesco").mkdir(parents=True, exist_ok=True)

# --- Walmart sample data ---
train_w = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=10, freq="D"),
    "store": [1]*10,
    "dept": [10]*10,
    "weekly_sales": range(10, 20),
})
test_w = train_w.copy()
features_w = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=10, freq="D"),
    "store": [1]*10,
    "markdown1": [0]*10,
})
stores_w = pd.DataFrame({
    "store": [1],
    "store_type": ["A"],
})

train_w.to_csv("data/walmart/train.csv", index=False)
test_w.to_csv("data/walmart/test.csv", index=False)
features_w.to_csv("data/walmart/features.csv", index=False)
stores_w.to_csv("data/walmart/stores.csv", index=False)

# --- Tesco sample data ---
sales_t = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=8, freq="D"),
    "store_id": [101]*8,
    "product_id": [1001]*8,
    "quantity": [5, 6, 7, 8, 5, 6, 7, 8],
})
inventory_t = sales_t.copy()
products_t = pd.DataFrame({
    "product_id": [1001],
    "product_name": ["Sample Product"]
})

sales_t.to_csv("data/tesco/sales.csv", index=False)
inventory_t.to_csv("data/tesco/inventory.csv", index=False)
products_t.to_csv("data/tesco/products.csv", index=False)

# Import project modules
from src.data_loader import WalmartDataLoader, TescoDataLoader, DataValidator
from src.data_processor import FeatureEngineer, DataScaler, TimeSeriesSplitter, OutlierDetector

# Test Walmart loader & preprocessing
w_loader = WalmartDataLoader(data_dir="data/walmart")
w_dfs = w_loader.load_all_files()
print("Walmart files loaded:")
for k, v in w_dfs.items():
    print(f"  {k}: shape={v.shape}")
df_w_pre = w_loader.preprocess_train_data()
print("Walmart preprocessed head:\n", df_w_pre.head())

# Test Tesco loader & preprocessing
t_loader = TescoDataLoader(data_dir="data/tesco")
sales = t_loader.load_sales_data()
print("Tesco sales loaded:", sales.shape)
products = t_loader.load_product_data()
print("Tesco products loaded:", products.shape)
sales_proc = t_loader.preprocess_sales_data()
print("Tesco preprocessed head:\n", sales_proc.head())

# Test feature engineering
fe = FeatureEngineer.create_lag_features(df_w_pre.sort_values("date"), target_col="weekly_sales", lags=[1,2])
print("Lag feature columns in Walmart df:", [c for c in fe.columns if "lag" in c])

# Test scaler
scaler = DataScaler("standard")
# select numeric columns from sales_proc; if empty, fall back to df_w_pre numeric columns
numeric_sales = sales_proc.select_dtypes(include=[float, int])
if numeric_sales.shape[1] == 0:
    numeric_sales = df_w_pre.select_dtypes(include=[float, int])
scaled = scaler.fit_transform(numeric_sales)
print("Scaled shape:", scaled.shape)

# Test validator
missing = DataValidator.check_missing_values(df_w_pre)
print("Missing percent (Walmart):", missing)

logger.info("Smoke test complete.")
