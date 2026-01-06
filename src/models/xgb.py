import pandas as pd, joblib, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

DATA   = Path("data/interim/feature_store.parquet")
MODEL  = Path("experiments/baselines/xgb.pkl")
RESULT = Path("experiments/baselines/xgb_metrics.json")

def train_xgb():
    df = pd.read_parquet(DATA)
    X = df.drop(columns=["cpi", "date"])
    y = df["cpi"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    reg = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    joblib.dump(reg, MODEL)
    pd.Series(metrics).to_json(RESULT)
    print("XGB metrics:", metrics)

if __name__ == "__main__":
    train_xgb()
