import pandas as pd, joblib, numpy as np
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA   = Path("data/interim/feature_store.parquet")
MODEL  = Path("experiments/baselines/arima.pkl")
RESULT = Path("experiments/baselines/arima_metrics.json")

def train_arima():
    df = pd.read_parquet(DATA)
    ts = df["cpi"]
    train_size = int(len(ts) * 0.7)
    train, test = ts[:train_size], ts[train_size:]

    model = ARIMA(train, order=(1,1,1)).fit()          # no trend arg
    preds = model.forecast(steps=len(test))

    mae  = mean_absolute_error(test, preds)
    rmse = np.sqrt(mean_squared_error(test, preds))
    mape = np.mean(np.abs((test - preds) / test)) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    joblib.dump(model, MODEL)
    pd.Series(metrics).to_json(RESULT)
    print("ARIMA metrics:", metrics)

if __name__ == "__main__":
    train_arima()
