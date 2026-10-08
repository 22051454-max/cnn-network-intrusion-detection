"""Pure-numpy forward pass matching ids.model.build_model, so inference needs no TensorFlow."""
import json
import os

import numpy as np

from .features import FEATURES, Scaler, clean_frame


def _conv1d_same(x, w, b):
    # x: (N, L, Cin), w: (K, Cin, Cout)
    k = w.shape[0]
    pad_l = (k - 1) // 2
    xp = np.pad(x, ((0, 0), (pad_l, k - 1 - pad_l), (0, 0)))
    L = x.shape[1]
    cols = np.stack([xp[:, i:i + L, :] for i in range(k)], axis=2)  # (N, L, K, Cin)
    return np.einsum("nlkc,kco->nlo", cols, w) + b


def _maxpool(x, p=2):
    L = (x.shape[1] // p) * p
    return x[:, :L, :].reshape(x.shape[0], L // p, p, x.shape[2]).max(axis=2)


def _relu(x):
    return np.maximum(x, 0)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NumpyIDS:
    def __init__(self, artifacts_dir):
        self.W = dict(np.load(os.path.join(artifacts_dir, "weights.npz")))
        self.scaler = Scaler.from_json(os.path.join(artifacts_dir, "scaler.json"))
        with open(os.path.join(artifacts_dir, "labels.json")) as fh:
            self.labels = json.load(fh)
        metrics_path = os.path.join(artifacts_dir, "metrics.json")
        self.metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}

    def predict_proba(self, X_raw):
        x = self.scaler.transform(X_raw)[..., None].astype(np.float32)
        W = self.W
        x = _maxpool(_relu(_conv1d_same(x, W["conv1_w"], W["conv1_b"])))
        x = _maxpool(_relu(_conv1d_same(x, W["conv2_w"], W["conv2_b"])))
        x = x.reshape(x.shape[0], -1)
        x = _relu(x @ W["dense1_w"] + W["dense1_b"])
        return _softmax(x @ W["out_w"] + W["out_b"])

    def predict_frame(self, df, batch=4096):
        df, X = clean_frame(df)
        probs = np.concatenate([self.predict_proba(X[i:i + batch]) for i in range(0, len(X), batch)]) if len(X) else np.zeros((0, len(self.labels)))
        idx = probs.argmax(1)
        return df, [self.labels[i] for i in idx], probs.max(1) if len(X) else np.array([])


__all__ = ["NumpyIDS", "FEATURES"]
