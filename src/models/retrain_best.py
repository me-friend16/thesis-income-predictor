import torch, pandas as pd
from pathlib import Path
from src.models.train_informer import InformerLM, make_loaders, DATA as FAIR_DATA, CKPT_DIR
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# best params from console
params = {'lr': 0.000999040695717467, 'd_model': 512, 'n_heads': 4,
          'e_layers': 2, 'd_ff': 1024, 'dropout': 0.2397}

print("Retraining with", params)

train_loader, val_loader = make_loaders(pd.read_parquet(FAIR_DATA), bs=64)
model = InformerLM(
    enc_in=1, dec_in=1,
    **{k: v for k, v in params.items() if k not in {"lr", "dropout"}},
    lr=params["lr"]
)

trainer = pl.Trainer(max_epochs=50, accelerator="auto", default_root_dir=CKPT_DIR)
trainer.fit(model, train_loader, val_loader)

# eval on val_loader
model.eval()
preds, y_true = [], []
with torch.no_grad():
    for x_enc, x_dec, y in val_loader:
        out = model(x_enc, x_dec)
        preds.append(out)
        y_true.append(y)
preds = torch.cat(preds).cpu().numpy().flatten()
y_true = torch.cat(y_true).cpu().numpy().flatten()

rmse = np.sqrt(mean_squared_error(y_true, preds))
mae  = mean_absolute_error(y_true, preds)
mape = np.mean(np.abs((y_true - preds) / y_true)) * 100

metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
print("Best Informer metrics:", metrics)
pd.Series(metrics).to_json(CKPT_DIR / "informer_best_metrics.json")
torch.save(model.model.state_dict(), CKPT_DIR / "informer_best.pt")
