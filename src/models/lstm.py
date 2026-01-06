import pandas as pd, joblib, torch, torch.nn as nn, pytorch_lightning as pl
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np

DATA   = Path("data/interim/feature_store.parquet")
MODEL  = Path("experiments/baselines/lstm.pt")
RESULT = Path("experiments/baselines/lstm_metrics.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TimeSeriesDataset(Dataset):
    def __init__(self, arr, seq_len=12):
        self.arr = arr
        self.seq_len = seq_len
    def __len__(self): return len(self.arr) - self.seq_len
    def __getitem__(self, idx):
        x = torch.tensor(self.arr[idx:idx+self.seq_len], dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.arr[idx+self.seq_len], dtype=torch.float32)
        return x, y

class LSTMNet(pl.LightningModule):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
        self.loss = nn.MSELoss()
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))
    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def train_lstm():
    df = pd.read_parquet(DATA)
    scaler = MinMaxScaler()
    vals   = scaler.fit_transform(df[["cpi"]]).flatten()

    ds  = TimeSeriesDataset(vals)
    train_ds, val_ds = torch.utils.data.random_split(ds, [int(.8*len(ds)), len(ds)-int(.8*len(ds))])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    model = LSTMNet()
    trainer = pl.Trainer(max_epochs=30, accelerator=DEVICE, enable_progress_bar=True, logger=False)
    trainer.fit(model, train_loader, val_loader)

    # eval on val_loader
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x)
            preds.append(out.squeeze())
            trues.append(y)
    preds = torch.cat(preds).cpu().numpy()
    trues = torch.cat(trues).cpu().numpy()

    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    trues = scaler.inverse_transform(trues.reshape(-1, 1)).flatten()

    mae  = np.mean(np.abs(trues - preds))
    rmse = np.sqrt(np.mean((trues - preds) ** 2))
    mape = np.mean(np.abs((trues - preds) / trues)) * 100

    metrics = {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}
    torch.save({"state": model.state_dict(), "scaler": scaler}, MODEL)
    pd.Series(metrics).to_json(RESULT)
    print("LSTM metrics:", metrics)

if __name__ == "__main__":
    train_lstm()
