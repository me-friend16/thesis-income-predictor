import joblib, torch, shap, pandas as pd, numpy as np
from pathlib import Path
from src.models.train_informer import InformerLM, make_loaders, DATA as FAIR_DATA

MODEL = Path("experiments/informer/informer_best.pt")
EXPLAINER = Path("experiments/informer/shap_explainer.pkl")
BACKGROUND = 100   # sub-second explanation

class SHAPExplainer:
    def __init__(self):
        self.model = self._load_model()
        self.explainer = self._build_explainer()

    def _load_model(self):
        model = InformerLM(enc_in=1, dec_in=1, d_model=512, n_heads=4, e_layers=2, d_ff=1024, lr=1e-3)
        model.model.load_state_dict(torch.load(MODEL, map_location="cpu"))
        model.eval()
        return model

    def _build_explainer(self):
        if EXPLAINER.exists():
            return joblib.load(EXPLAINER)
        train_loader, _ = make_loaders(pd.read_parquet(FAIR_DATA), bs=BACKGROUND)
        x_enc, x_dec, _ = next(iter(train_loader))
        background = x_enc[:BACKGROUND].numpy().squeeze(-1)   # [100, 12]
        explainer = shap.KernelExplainer(self._predict_numpy, background)
        joblib.dump(explainer, EXPLAINER)
        return explainer

    def _predict_numpy(self, x):
        # x: [N, 12]  ->  [N, 12, 1] tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        dec = torch.zeros((x.shape[0], 3, 1))
        with torch.no_grad():
            out = self.model(x_tensor, dec).numpy()[:, 0]  # take 1st month
        return out

    def explain(self, x_raw):
        # x_raw: list of length 12
        x = np.array(x_raw).reshape(1, -1)
        shap_values = self.explainer.shap_values(x, nsamples=100)
        return {
            "base_value": float(self.explainer.expected_value),
            "shap_values": shap_values.tolist(),
            "feature_names": [f"lag_{i+1}" for i in range(12)]
        }
