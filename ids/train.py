"""Train the CNN on CICIDS2017 CSVs (or synthetic data) and export artifacts.

Usage:
  python -m ids.train --data data/MachineLearningCVE           # real dataset (folder of CSVs)
  python -m ids.train --synthetic --rows-per-class 3000        # demo / CI
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .features import CLASSES, FEATURES, Scaler, clean_frame, normalize_label


def load_cicids(folder, max_rows_per_class=None, seed=42):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {folder}")
    frames = []
    for f in files:
        print("loading", os.path.basename(f))
        frames.append(pd.read_csv(f, low_memory=False, encoding="latin-1"))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns=lambda c: c.strip())
    df["Label"] = df["Label"].map(normalize_label)
    df = df.drop_duplicates()
    if max_rows_per_class:  # downsample the huge BENIGN / DoS Hulk classes
        df = df.groupby("Label", group_keys=False).apply(lambda g: g.sample(min(len(g), max_rows_per_class), random_state=seed))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Folder with CICIDS2017 MachineLearningCSV files")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rows-per-class", type=int, default=3000, help="synthetic rows, or cap per class for real data")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args(argv)

    import tensorflow as tf
    from .model import build_model, export_numpy_weights

    tf.keras.utils.set_random_seed(42)
    if a.synthetic:
        from .synthetic import generate
        df, source = generate(a.rows_per_class), "synthetic"
    elif a.data:
        df, source = load_cicids(a.data, a.rows_per_class), "CICIDS2017"
    else:
        ap.error("pass --data or --synthetic")

    df, X = clean_frame(df)
    labels = [c for c in CLASSES if c in set(df["Label"])]
    y = df["Label"].map({c: i for i, c in enumerate(labels)}).to_numpy()
    # stratify only when every class has at least 2 samples (Heartbleed has 11 rows in the real data)
    strat = y if np.bincount(y).min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=42,
                                                stratify=y_tr if strat is not None else None)
    scaler = Scaler().fit(X_tr)
    T = lambda m: scaler.transform(m)[..., None]
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    model = build_model(len(FEATURES), len(labels))
    model.summary()
    t0 = time.time()
    model.fit(T(X_tr), y_tr, validation_data=(T(X_val), y_val), epochs=a.epochs, batch_size=a.batch_size,
              class_weight={int(c): float(w) for c, w in zip(np.unique(y_tr), np.sqrt(cw))},
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)], verbose=2)
    pred = model.predict(T(X_te), batch_size=2048, verbose=0).argmax(1)
    acc = float((pred == y_te).mean())
    report = classification_report(y_te, pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=range(len(labels)))
    print(classification_report(y_te, pred, labels=range(len(labels)), target_names=labels, zero_division=0))
    print(f"Test accuracy: {acc:.4f}")

    os.makedirs(a.out, exist_ok=True)
    model.save(os.path.join(a.out, "ids_cnn.keras"))
    export_numpy_weights(model, os.path.join(a.out, "weights.npz"))
    scaler.to_json(os.path.join(a.out, "scaler.json"))
    json.dump(labels, open(os.path.join(a.out, "labels.json"), "w"), indent=1)
    json.dump({"source": source, "test_accuracy": round(acc, 4), "macro_f1": round(report["macro avg"]["f1-score"], 4),
               "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)), "train_seconds": round(time.time() - t0, 1),
               "per_class_f1": {k: round(v["f1-score"], 4) for k, v in report.items() if k in labels},
               "confusion_matrix": cm.tolist()}, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cmn, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=75, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(f"Confusion matrix ({source}, accuracy {acc:.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "confusion_matrix.png"), dpi=120)
    except ImportError:
        pass
    print("artifacts written to", a.out)


if __name__ == "__main__":
    main()
