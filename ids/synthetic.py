"""Generate CICIDS2017-shaped synthetic flows so the pipeline runs without the 1 GB dataset.

Each class has a rough behavioural profile (ports, packet counts, sizes, timing, flags) inspired by the
attack descriptions in the CICIDS2017 paper. It is NOT a substitute for the real data: use it for demos,
tests and CI only.
"""
import numpy as np
import pandas as pd

from .features import ALL_COLUMNS, CLASSES

# port(s), duration_us (log-mean), fwd_pkts, bwd_pkts, fwd_size, bwd_size, syn, fin, rst, psh, ack, urg, init_win_fwd
PROFILES = {
    "BENIGN":                   ([80, 443, 53, 8080, 123], 12.0, 8, 7, 300, 600, 0.3, 0.6, 0.05, 0.5, 0.7, 0.0, 8192),
    "Bot":                      ([8080, 8888], 11.0, 4, 3, 180, 120, 0.1, 0.3, 0.1, 0.8, 0.9, 0.0, 8192),
    "DDoS":                     ([80], 13.5, 4, 0.5, 20, 5, 0.1, 0.1, 0.0, 0.1, 0.9, 0.5, 256),
    "DoS GoldenEye":            ([80], 15.0, 8, 6, 400, 1500, 0.2, 0.1, 0.4, 0.9, 1.0, 0.1, 29200),
    "DoS Hulk":                 ([80], 9.0, 6, 5, 350, 2000, 0.0, 0.9, 0.0, 0.2, 0.9, 0.0, 29200),
    "DoS Slowhttptest":         ([80], 16.5, 3, 1, 10, 0.2, 0.9, 0.1, 0.4, 0.1, 0.2, 0.0, 29200),
    "DoS slowloris":            ([80], 17.0, 5, 2, 30, 2, 0.2, 0.0, 0.2, 0.9, 0.9, 0.0, 29200),
    "FTP-Patator":              ([21], 14.0, 10, 14, 20, 50, 0.1, 0.2, 0.0, 1.0, 1.0, 0.0, 29200),
    "Heartbleed":               ([444], 18.5, 2800, 2000, 120, 4200, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 251),
    "Infiltration":             ([444, 135, 139, 445], 15.5, 15, 12, 90, 300, 0.4, 0.3, 0.2, 0.4, 0.7, 0.0, 8192),
    "PortScan":                 (None, 3.5, 1, 1, 0.5, 2, 1.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1024),
    "SSH-Patator":              ([22], 14.5, 22, 33, 60, 70, 0.1, 0.2, 0.0, 1.0, 1.0, 0.0, 29200),
    "Web Attack - Brute Force": ([80], 15.8, 4, 3, 600, 3000, 0.0, 0.2, 0.0, 1.0, 1.0, 0.0, 29200),
    "Web Attack - Sql Injection": ([80], 13.3, 5, 4, 900, 600, 0.1, 0.5, 0.0, 1.0, 1.0, 0.0, 29200),
    "Web Attack - XSS":         ([80], 16.2, 6, 5, 1200, 2500, 0.0, 0.3, 0.0, 1.0, 1.0, 0.0, 29200),
}


def _pos(rng, mean, n, spread=0.35):
    return np.maximum(rng.lognormal(np.log(max(mean, 1e-3)), spread, n), 0)


def generate(n_per_class=2000, seed=7, classes=None, overlap=0.08):
    rng = np.random.default_rng(seed)
    frames = []
    for label in classes or CLASSES:
        port, dur, fp, bp, fs, bs, syn, fin, rst, psh, ack, urg, win = PROFILES[label]
        n = n_per_class
        ports = rng.integers(1, 65535, n) if port is None else rng.choice(port, n)
        duration = np.exp(rng.normal(dur, 1.0, n))
        fwd = np.maximum(np.round(_pos(rng, fp, n, 0.5)), 1)
        bwd = np.round(_pos(rng, bp, n, 0.5))
        fsize, bsize = _pos(rng, fs, n, 0.5), _pos(rng, bs, n, 0.5)
        # a fraction of rows borrow benign-looking timing to make the task non-trivial
        mix = rng.random(n) < overlap
        duration[mix] = np.exp(rng.normal(12.0, 1.5, mix.sum()))
        tot_f, tot_b = fwd * fsize, bwd * bsize
        pkts = fwd + bwd
        dur_s = duration / 1e6
        iat = duration / np.maximum(pkts - 1, 1)
        flags = {k: (rng.random(n) < p).astype(float) for k, p in
                 dict(syn=syn, fin=fin, rst=rst, psh=psh, ack=ack, urg=urg).items()}
        d = {
            "Destination Port": ports, "Flow Duration": duration, "Total Fwd Packets": fwd,
            "Total Backward Packets": bwd, "Total Length of Fwd Packets": tot_f, "Total Length of Bwd Packets": tot_b,
            "Fwd Packet Length Max": fsize * 1.6, "Fwd Packet Length Min": fsize * 0.2, "Fwd Packet Length Mean": fsize,
            "Fwd Packet Length Std": fsize * 0.5, "Bwd Packet Length Max": bsize * 1.6, "Bwd Packet Length Min": bsize * 0.1,
            "Bwd Packet Length Mean": bsize, "Bwd Packet Length Std": bsize * 0.6,
            "Flow Bytes/s": (tot_f + tot_b) / np.maximum(dur_s, 1e-6), "Flow Packets/s": pkts / np.maximum(dur_s, 1e-6),
            "Flow IAT Mean": iat, "Flow IAT Std": iat * rng.uniform(0.3, 1.5, n), "Flow IAT Max": iat * rng.uniform(1.5, 4, n),
            "Flow IAT Min": iat * rng.uniform(0, 0.3, n), "Fwd IAT Total": duration * rng.uniform(0.5, 1, n),
            "Fwd IAT Mean": duration / np.maximum(fwd, 1), "Fwd IAT Std": iat * 0.8, "Fwd IAT Max": iat * 3,
            "Fwd IAT Min": iat * 0.1, "Bwd IAT Total": duration * rng.uniform(0, 0.9, n) * (bwd > 0),
            "Bwd IAT Mean": duration / np.maximum(bwd, 1) * (bwd > 0), "Bwd IAT Std": iat * 0.7 * (bwd > 0),
            "Bwd IAT Max": iat * 2.5 * (bwd > 0), "Bwd IAT Min": iat * 0.1 * (bwd > 0),
            "Fwd PSH Flags": flags["psh"] * (rng.random(n) < 0.5), "Bwd PSH Flags": 0.0, "Fwd URG Flags": 0.0, "Bwd URG Flags": 0.0,
            "Fwd Header Length": fwd * rng.choice([20, 32, 40], n), "Bwd Header Length": bwd * rng.choice([20, 32, 40], n),
            "Fwd Packets/s": fwd / np.maximum(dur_s, 1e-6), "Bwd Packets/s": bwd / np.maximum(dur_s, 1e-6),
            "Min Packet Length": np.minimum(fsize * 0.2, bsize * 0.1), "Max Packet Length": np.maximum(fsize, bsize) * 1.6,
            "Packet Length Mean": (tot_f + tot_b) / np.maximum(pkts, 1), "Packet Length Std": (fsize + bsize) * 0.4,
            "Packet Length Variance": ((fsize + bsize) * 0.4) ** 2,
            "FIN Flag Count": flags["fin"], "SYN Flag Count": flags["syn"], "RST Flag Count": flags["rst"],
            "PSH Flag Count": flags["psh"], "ACK Flag Count": flags["ack"], "URG Flag Count": flags["urg"],
            "CWE Flag Count": 0.0, "ECE Flag Count": flags["rst"] * (rng.random(n) < 0.2),
            "Down/Up Ratio": np.floor(bwd / np.maximum(fwd, 1)), "Average Packet Size": (tot_f + tot_b) / np.maximum(pkts, 1) * 1.1,
            "Avg Fwd Segment Size": fsize, "Avg Bwd Segment Size": bsize, "Fwd Header Length.1": None,
            "Fwd Avg Bytes/Bulk": 0.0, "Fwd Avg Packets/Bulk": 0.0, "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0, "Bwd Avg Packets/Bulk": 0.0, "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd, "Subflow Fwd Bytes": tot_f, "Subflow Bwd Packets": bwd, "Subflow Bwd Bytes": tot_b,
            "Init_Win_bytes_forward": np.where(rng.random(n) < 0.85, win, rng.choice([-1, 8192, 65535], n)),
            "Init_Win_bytes_backward": rng.choice([-1, 235, 2081, 28960, 65160], n),
            "act_data_pkt_fwd": np.floor(fwd * rng.uniform(0, 0.8, n)), "min_seg_size_forward": rng.choice([20, 32], n),
            "Active Mean": duration * rng.uniform(0, 0.2, n), "Active Std": duration * rng.uniform(0, 0.05, n),
            "Active Max": duration * rng.uniform(0, 0.3, n), "Active Min": duration * rng.uniform(0, 0.1, n),
            "Idle Mean": duration * rng.uniform(0, 0.8, n), "Idle Std": duration * rng.uniform(0, 0.1, n),
            "Idle Max": duration * rng.uniform(0, 0.9, n), "Idle Min": duration * rng.uniform(0, 0.5, n),
        }
        d["Fwd Header Length.1"] = d["Fwd Header Length"]
        df = pd.DataFrame({c: np.broadcast_to(np.asarray(d[c], dtype=float), (n,)) for c in ALL_COLUMNS})
        df["Label"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-class", type=int, default=2000)
    ap.add_argument("--out", default="data/synthetic_cicids.csv")
    a = ap.parse_args()
    generate(a.rows_per_class).to_csv(a.out, index=False)
    print("wrote", a.out)
