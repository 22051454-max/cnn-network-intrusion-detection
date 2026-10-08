import numpy as np
import pandas as pd

from app import create_app
from ids.features import FEATURES, normalize_label
from ids.infer import NumpyIDS
from ids.synthetic import generate


def test_feature_count():
    assert len(FEATURES) == 69


def test_label_normalisation():
    assert normalize_label("Web Attack � Brute Force") == "Web Attack - Brute Force"
    assert normalize_label(" DoS Hulk ") == "DoS Hulk"


def test_numpy_model_predicts_all_classes():
    ids = NumpyIDS("artifacts")
    df = generate(30, seed=123)
    _, preds, conf = ids.predict_frame(df)
    acc = np.mean(np.array(preds) == df["Label"].to_numpy())
    assert acc > 0.8 and conf.min() >= 0 and conf.max() <= 1


def test_inf_rows_dropped():
    df = generate(5, seed=1)
    df.loc[0, "Flow Bytes/s"] = np.inf
    _, preds, _ = NumpyIDS("artifacts").predict_frame(df)
    assert len(preds) == len(df) - 1


def test_web_sample_and_api():
    c = create_app().test_client()
    assert b"Traffic breakdown" in c.post("/", data={"sample": "1"}).data
    flows = generate(2, seed=3).drop(columns=["Label"]).to_dict(orient="records")
    r = c.post("/api/predict", json={"flows": flows})
    assert r.status_code == 200 and r.get_json()["rows"] == len(flows)
    assert c.post("/api/predict", json={"flows": [{"x": 1}]}).status_code == 400
