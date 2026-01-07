from fastapi import FastAPI
from pydantic import BaseModel
from src.explain.shap_engine import SHAPExplainer

app = FastAPI(title="Income-Informer SHAP")
explainer = SHAPExplainer()

class SeriesIn(BaseModel):
    lag_values: list[float]   # 12 historical CPI values

@app.post("/explain")
def explain(body: SeriesIn):
    return explainer.explain(body.lag_values)
