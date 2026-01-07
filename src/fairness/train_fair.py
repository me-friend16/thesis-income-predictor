import torch, pandas as pd
from pathlib import Path
from src.models.train_informer import InformerLM, make_loaders, CKPT_DIR
import pytorch_lightning as pl

DATA = Path("data/interim/feature_store_fair.parquet")

class FairInformerLM(InformerLM):
    def training_step(self, batch, _):
        x_enc, x_dec, y = batch[:3]
        preds = self(x_enc, x_dec)
        loss = torch.nn.functional.mse_loss(preds, y)
        self.log("train_loss", loss)
        return loss

def train_fair():
    train_loader, val_loader = make_loaders(pd.read_parquet(DATA), bs=64)
    model = FairInformerLM(enc_in=1, dec_in=1, d_model=512, n_heads=4, e_layers=2, d_ff=1024, lr=1e-3)
    trainer = pl.Trainer(max_epochs=50, accelerator="auto", default_root_dir=CKPT_DIR)
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.model.state_dict(), CKPT_DIR / "informer_fair.pt")
    print("Fair model saved ->", CKPT_DIR / "informer_fair.pt")

if __name__ == "__main__":
    train_fair()
