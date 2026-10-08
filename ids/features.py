"""CICIDS2017 feature schema and preprocessing shared by training and inference."""
import json
import re

import numpy as np

# The 78 flow features in CICIDS2017 MachineLearningCSV files (column names stripped of whitespace).
ALL_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
# Duplicated or all-zero in CICIDS2017; dropping them gives a 69-feature input vector.
DROP = {"Fwd Header Length.1", "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
        "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"}
FEATURES = [c for c in ALL_COLUMNS if c not in DROP]

# BENIGN + 14 attack classes.
CLASSES = [
    "BENIGN", "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris", "FTP-Patator",
    "Heartbleed", "Infiltration", "PortScan", "SSH-Patator", "Web Attack - Brute Force", "Web Attack - Sql Injection",
    "Web Attack - XSS",
]


def normalize_label(label: str) -> str:
    """CICIDS2017 web-attack labels contain a mis-encoded dash ('Web Attack � Brute Force')."""
    label = re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7e]", "-", str(label))).strip()
    return re.sub(r"Web Attack\s*-+\s*", "Web Attack - ", label)


def clean_frame(df):
    """Strip column names, coerce numerics, replace inf with NaN and drop bad rows."""
    df = df.rename(columns=lambda c: c.strip())
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} expected columns, e.g. {missing[:3]}")
    X = df[FEATURES].apply(lambda s: s.astype("float64", errors="ignore")).replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    return df.loc[keep], X.loc[keep].to_numpy(dtype=np.float64)


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Scaler:
    """log1p + standardisation, serialisable to JSON so inference needs only numpy."""

    def fit(self, X):
        Z = signed_log(X)
        self.mean, self.std = Z.mean(0), Z.std(0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, X):
        return ((signed_log(X) - self.mean) / self.std).astype(np.float32)

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump({"features": FEATURES, "mean": self.mean.tolist(), "std": self.std.tolist()}, fh)

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        s = cls()
        s.mean, s.std = np.array(d["mean"]), np.array(d["std"])
        return s
