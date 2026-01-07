import pandas as pd, numpy as np, torch, joblib
from pathlib import Path
from src.models.train_informer import InformerLM, make_loaders
from sklearn.metrics import mean_absolute_error

DATA   = Path("data/interim/feature_store_fair.parquet")
MODEL  = Path("experiments/informer/informer_best.pt")
METRICS= Path("experiments/informer/fairness_audit.json")

def audit():
    df = pd.read_parquet(DATA)
    train_loader, val_loader = make_loaders(df, bs=64)

    # load best model (state-dict version)
    model = InformerLM(enc_in=1, dec_in=1, d_model=512, n_heads=4, e_layers=2, d_ff=1024, lr=1e-3)
    model.model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()

    results = []
    with torch.no_grad():
        for x_enc, x_dec, y in val_loader:
            out = model(x_enc, x_dec)
            results.append(pd.DataFrame({"y_true": y.cpu().numpy().flatten(),
                                         "y_pred": out.cpu().numpy().flatten()}))
    res = pd.concat(results).reset_index(drop=True)
    res = pd.concat([res, df.iloc[-len(res):][["gender", "age_group"]].reset_index(drop=True)], axis=1)

    def err_ratio(group):
        return mean_absolute_error(group.y_true, group.y_pred)

    audit = res.groupby(["gender", "age_group"]).apply(err_ratio).reset_index(name="mae")
    overall = err_ratio(res)
    audit["error_ratio"] = audit.mae / overall
    audit.to_json(METRICS, indent=2)
    print(audit)
    return audit

if __name__ == "__main__":
    audit()
