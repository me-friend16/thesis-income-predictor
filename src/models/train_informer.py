import pandas as pd, torch, pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.nn as nn, numpy as np

DATA      = Path("data/interim/feature_store.parquet")
CKPT_DIR  = Path("experiments/informer")
CKPT_DIR.mkdir(exist_ok=True)

SEQ_LEN, OUT_LEN = 12, 3

# ---- import our Informer model ----
from src.models.informer import Informer

class InformerLM(pl.LightningModule):
    def __init__(self, enc_in=1, dec_in=1, d_model=256, n_heads=8, e_layers=2, d_ff=512, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = Informer(enc_in, dec_in, d_model, n_heads, e_layers, d_ff, OUT_LEN)
        self.criterion = nn.MSELoss()
    def forward(self, x_enc, x_dec):
        return self.model(x_enc, x_dec)
    def training_step(self, batch, _):
        x_enc, x_dec, y = batch
        preds = self(x_enc, x_dec)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def make_loaders(df, seq_len=SEQ_LEN, out_len=OUT_LEN, bs=32):
    vals = df["cpi"].values
    X, Y = [], []
    for i in range(len(vals) - seq_len - out_len + 1):
        X.append(vals[i:i+seq_len])
        Y.append(vals[i+seq_len:i+seq_len+out_len])
    X = np.array(X, dtype=np.float32)[:, :, None]   # [N, seq_len, 1]
    Y = np.array(Y, dtype=np.float32)               # [N, out_len]
    dec = np.zeros((X.shape[0], out_len, 1), dtype=np.float32)

    train_x, val_x, train_dec, val_dec, train_y, val_y = train_test_split(
        X, dec, Y, test_size=0.2, shuffle=False)
    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_dec), torch.tensor(train_y))
    val_ds   = TensorDataset(torch.tensor(val_x),   torch.tensor(val_dec),   torch.tensor(val_y))
    return DataLoader(train_ds, batch_size=bs), DataLoader(val_ds, batch_size=bs)

def train():
    df = pd.read_parquet(DATA)
    train_loader, val_loader = make_loaders(df)
    model = InformerLM()
    trainer = pl.Trainer(max_epochs=50, accelerator="auto", default_root_dir=CKPT_DIR)
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.model.state_dict(), CKPT_DIR / "informer.pt")
    print("Informer checkpoint saved ->", CKPT_DIR / "informer.pt")

if __name__ == "__main__":
    train()
