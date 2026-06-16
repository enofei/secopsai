"""
Synthetic CICIDS-Style Dataset Generator — SecOpsAI (v2)
Adds realistic noise, class overlap, and mislabeled-style edge cases
so the model has to learn genuine patterns instead of memorizing
clean boundaries.
"""

import pandas as pd
import numpy as np

np.random.seed(42)


def add_noise(df, numeric_cols, noise_level=0.25):
    """
    Adds Gaussian noise scaled to each column's own range.
    This blurs the artificial boundaries between classes,
    forcing the model to learn real patterns instead of
    memorizing clean thresholds.
    """
    for col in numeric_cols:
        std = df[col].std()
        noise = np.random.normal(0, std * noise_level, size=len(df))
        df[col] = df[col] + noise
        # Clip negative values where they don't make physical sense
        if df[col].min() >= 0:
            df[col] = df[col].clip(lower=0)
    return df


def generate_benign(n=5000):
    return pd.DataFrame({
        'Flow Duration':               np.random.randint(1000, 5000000, n),
        'Total Fwd Packets':           np.random.randint(2, 50, n),
        'Total Backward Packets':      np.random.randint(2, 50, n),
        'Total Length of Fwd Packets': np.random.randint(100, 50000, n),
        'Total Length of Bwd Packets': np.random.randint(100, 50000, n),
        'Fwd Packet Length Max':       np.random.randint(40, 1500, n),
        'Fwd Packet Length Min':       np.random.randint(20, 100, n),
        'Fwd Packet Length Mean':      np.random.uniform(100, 800, n),
        'Bwd Packet Length Max':       np.random.randint(40, 1500, n),
        'Bwd Packet Length Min':       np.random.randint(20, 100, n),
        'Bwd Packet Length Mean':      np.random.uniform(100, 800, n),
        'Flow Bytes/s':                np.random.uniform(1000, 500000, n),
        'Flow Packets/s':              np.random.uniform(1, 1000, n),
        'Flow IAT Mean':               np.random.uniform(1000, 100000, n),
        'Flow IAT Std':                np.random.uniform(500, 50000, n),
        'Fwd IAT Mean':                np.random.uniform(1000, 100000, n),
        'Bwd IAT Mean':                np.random.uniform(1000, 100000, n),
        'Fwd PSH Flags':               np.random.randint(0, 2, n),
        'Bwd PSH Flags':               np.random.randint(0, 2, n),
        'Packet Length Mean':          np.random.uniform(100, 800, n),
        'Packet Length Std':           np.random.uniform(50, 400, n),
        'Packet Length Variance':      np.random.uniform(100, 50000, n),
        'FIN Flag Count':              np.random.randint(0, 2, n),
        'SYN Flag Count':              np.random.randint(0, 2, n),
        'RST Flag Count':              np.random.randint(0, 1, n),
        'PSH Flag Count':              np.random.randint(0, 5, n),
        'ACK Flag Count':              np.random.randint(1, 10, n),
        'URG Flag Count':              np.zeros(n),
        'Average Packet Size':         np.random.uniform(100, 800, n),
        'Avg Fwd Segment Size':        np.random.uniform(100, 800, n),
        'Avg Bwd Segment Size':        np.random.uniform(100, 800, n),
        'Label': 'BENIGN'
    })


def generate_c2_beaconing(n=1000):
    # Widened ranges so they overlap into BENIGN's space at the edges
    return pd.DataFrame({
        'Flow Duration':               np.random.randint(700000, 1300000, n),
        'Total Fwd Packets':           np.random.randint(1, 8, n),
        'Total Backward Packets':      np.random.randint(1, 8, n),
        'Total Length of Fwd Packets': np.random.randint(40, 800, n),
        'Total Length of Bwd Packets': np.random.randint(40, 800, n),
        'Fwd Packet Length Max':       np.random.randint(60, 300, n),
        'Fwd Packet Length Min':       np.random.randint(40, 100, n),
        'Fwd Packet Length Mean':      np.random.uniform(50, 250, n),
        'Bwd Packet Length Max':       np.random.randint(60, 300, n),
        'Bwd Packet Length Min':       np.random.randint(40, 100, n),
        'Bwd Packet Length Mean':      np.random.uniform(50, 250, n),
        'Flow Bytes/s':                np.random.uniform(10, 1500, n),
        'Flow Packets/s':              np.random.uniform(0.001, 0.5, n),
        'Flow IAT Mean':               np.random.uniform(700000, 1300000, n),
        'Flow IAT Std':                np.random.uniform(1000, 60000, n),
        'Fwd IAT Mean':                np.random.uniform(700000, 1300000, n),
        'Bwd IAT Mean':                np.random.uniform(700000, 1300000, n),
        'Fwd PSH Flags':               np.random.randint(0, 2, n),
        'Bwd PSH Flags':               np.random.randint(0, 2, n),
        'Packet Length Mean':          np.random.uniform(50, 250, n),
        'Packet Length Std':           np.random.uniform(5, 100, n),
        'Packet Length Variance':      np.random.uniform(25, 5000, n),
        'FIN Flag Count':              np.random.randint(0, 2, n),
        'SYN Flag Count':              np.random.randint(0, 2, n),
        'RST Flag Count':              np.random.randint(0, 2, n),
        'PSH Flag Count':              np.random.randint(0, 3, n),
        'ACK Flag Count':              np.random.randint(0, 3, n),
        'URG Flag Count':              np.zeros(n),
        'Average Packet Size':         np.random.uniform(50, 250, n),
        'Avg Fwd Segment Size':        np.random.uniform(50, 250, n),
        'Avg Bwd Segment Size':        np.random.uniform(50, 250, n),
        'Label': 'C2-Beaconing'
    })


def generate_lateral_movement(n=1000):
    return pd.DataFrame({
        'Flow Duration':               np.random.randint(100, 20000, n),
        'Total Fwd Packets':           np.random.randint(1, 6, n),
        'Total Backward Packets':      np.random.randint(0, 4, n),
        'Total Length of Fwd Packets': np.random.randint(40, 300, n),
        'Total Length of Bwd Packets': np.random.randint(0, 200, n),
        'Fwd Packet Length Max':       np.random.randint(40, 200, n),
        'Fwd Packet Length Min':       np.random.randint(40, 100, n),
        'Fwd Packet Length Mean':      np.random.uniform(40, 150, n),
        'Bwd Packet Length Max':       np.random.randint(0, 150, n),
        'Bwd Packet Length Min':       np.random.randint(0, 50, n),
        'Bwd Packet Length Mean':      np.random.uniform(0, 100, n),
        'Flow Bytes/s':                np.random.uniform(2000, 80000, n),
        'Flow Packets/s':              np.random.uniform(50, 1200, n),
        'Flow IAT Mean':               np.random.uniform(100, 5000, n),
        'Flow IAT Std':                np.random.uniform(10, 1500, n),
        'Fwd IAT Mean':                np.random.uniform(100, 5000, n),
        'Bwd IAT Mean':                np.random.uniform(100, 5000, n),
        'Fwd PSH Flags':               np.random.randint(0, 2, n),
        'Bwd PSH Flags':               np.random.randint(0, 2, n),
        'Packet Length Mean':          np.random.uniform(40, 150, n),
        'Packet Length Std':           np.random.uniform(0, 50, n),
        'Packet Length Variance':      np.random.uniform(0, 2000, n),
        'FIN Flag Count':              np.random.randint(0, 2, n),
        'SYN Flag Count':              np.random.randint(0, 3, n),
        'RST Flag Count':              np.random.randint(0, 2, n),
        'PSH Flag Count':              np.random.randint(0, 2, n),
        'ACK Flag Count':              np.random.randint(0, 3, n),
        'URG Flag Count':              np.zeros(n),
        'Average Packet Size':         np.random.uniform(40, 150, n),
        'Avg Fwd Segment Size':        np.random.uniform(40, 150, n),
        'Avg Bwd Segment Size':        np.random.uniform(0, 100, n),
        'Label': 'Lateral-Movement'
    })


def generate_dns_tunnelling(n=1000):
    return pd.DataFrame({
        'Flow Duration':               np.random.randint(100, 30000, n),
        'Total Fwd Packets':           np.random.randint(3, 40, n),
        'Total Backward Packets':      np.random.randint(1, 10, n),
        'Total Length of Fwd Packets': np.random.randint(1000, 12000, n),
        'Total Length of Bwd Packets': np.random.randint(50, 1000, n),
        'Fwd Packet Length Max':       np.random.randint(150, 512, n),
        'Fwd Packet Length Min':       np.random.randint(80, 250, n),
        'Fwd Packet Length Mean':      np.random.uniform(120, 450, n),
        'Bwd Packet Length Max':       np.random.randint(40, 200, n),
        'Bwd Packet Length Min':       np.random.randint(30, 100, n),
        'Bwd Packet Length Mean':      np.random.uniform(35, 150, n),
        'Flow Bytes/s':                np.random.uniform(300, 8000, n),
        'Flow Packets/s':              np.random.uniform(5, 200, n),
        'Flow IAT Mean':               np.random.uniform(300, 8000, n),
        'Flow IAT Std':                np.random.uniform(50, 3000, n),
        'Fwd IAT Mean':                np.random.uniform(300, 8000, n),
        'Bwd IAT Mean':                np.random.uniform(300, 8000, n),
        'Fwd PSH Flags':               np.random.randint(0, 2, n),
        'Bwd PSH Flags':               np.random.randint(0, 2, n),
        'Packet Length Mean':          np.random.uniform(80, 400, n),
        'Packet Length Std':           np.random.uniform(30, 300, n),
        'Packet Length Variance':      np.random.uniform(300, 15000, n),
        'FIN Flag Count':              np.random.randint(0, 2, n),
        'SYN Flag Count':              np.random.randint(0, 2, n),
        'RST Flag Count':              np.random.randint(0, 2, n),
        'PSH Flag Count':              np.random.randint(0, 2, n),
        'ACK Flag Count':              np.random.randint(0, 3, n),
        'URG Flag Count':              np.zeros(n),
        'Average Packet Size':         np.random.uniform(80, 400, n),
        'Avg Fwd Segment Size':        np.random.uniform(120, 450, n),
        'Avg Bwd Segment Size':        np.random.uniform(35, 150, n),
        'Label': 'DNS-Tunnelling'
    })


def generate_slow_exfiltration(n=1000):
    return pd.DataFrame({
        'Flow Duration':               np.random.randint(5000000, 120000000, n),
        'Total Fwd Packets':           np.random.randint(2, 20, n),
        'Total Backward Packets':      np.random.randint(1, 10, n),
        'Total Length of Fwd Packets': np.random.randint(100, 3000, n),
        'Total Length of Bwd Packets': np.random.randint(50, 1500, n),
        'Fwd Packet Length Max':       np.random.randint(60, 300, n),
        'Fwd Packet Length Min':       np.random.randint(40, 100, n),
        'Fwd Packet Length Mean':      np.random.uniform(50, 250, n),
        'Bwd Packet Length Max':       np.random.randint(40, 200, n),
        'Bwd Packet Length Min':       np.random.randint(20, 80, n),
        'Bwd Packet Length Mean':      np.random.uniform(30, 150, n),
        'Flow Bytes/s':                np.random.uniform(0.1, 60, n),
        'Flow Packets/s':              np.random.uniform(0.0001, 0.1, n),
        'Flow IAT Mean':               np.random.uniform(3000000, 25000000, n),
        'Flow IAT Std':                np.random.uniform(100000, 4000000, n),
        'Fwd IAT Mean':                np.random.uniform(3000000, 25000000, n),
        'Bwd IAT Mean':                np.random.uniform(3000000, 25000000, n),
        'Fwd PSH Flags':               np.random.randint(0, 2, n),
        'Bwd PSH Flags':               np.random.randint(0, 2, n),
        'Packet Length Mean':          np.random.uniform(50, 250, n),
        'Packet Length Std':           np.random.uniform(5, 80, n),
        'Packet Length Variance':      np.random.uniform(25, 3000, n),
        'FIN Flag Count':              np.random.randint(0, 2, n),
        'SYN Flag Count':              np.random.randint(0, 2, n),
        'RST Flag Count':              np.random.randint(0, 2, n),
        'PSH Flag Count':              np.random.randint(0, 2, n),
        'ACK Flag Count':              np.random.randint(0, 3, n),
        'URG Flag Count':              np.zeros(n),
        'Average Packet Size':         np.random.uniform(50, 250, n),
        'Avg Fwd Segment Size':        np.random.uniform(50, 250, n),
        'Avg Bwd Segment Size':        np.random.uniform(30, 150, n),
        'Label': 'Slow-Exfiltration'
    })


def inject_label_overlap(df, swap_fraction=0.04):
    """
    Real-world datasets have ambiguous cases that look like one
    class but are labeled another (e.g., a slow benign download
    that resembles slow exfiltration). We simulate this by taking
    a small fraction of each attack class and giving it some
    BENIGN-like feature values, and vice versa, WITHOUT changing
    the label. This creates genuine classification difficulty.
    """
    rng = np.random.RandomState(7)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    n_swap = int(len(df) * swap_fraction)
    swap_idx = rng.choice(df.index, size=n_swap, replace=False)

    benign_sample = df[df['Label'] == 'BENIGN'][numeric_cols].sample(
        n=n_swap, replace=True, random_state=7
    ).values

    # Blend 40% of a random benign row's features into these rows
    df.loc[swap_idx, numeric_cols] = (
        0.6 * df.loc[swap_idx, numeric_cols].values + 0.4 * benign_sample
    )

    return df


def generate_full_dataset():
    print("Generating synthetic CICIDS-style dataset (v2 — with noise)...")

    benign  = generate_benign(5000)
    c2      = generate_c2_beaconing(1000)
    lateral = generate_lateral_movement(1000)
    dns     = generate_dns_tunnelling(1000)
    exfil   = generate_slow_exfiltration(1000)

    df = pd.concat([benign, c2, lateral, dns, exfil], ignore_index=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = add_noise(df, numeric_cols, noise_level=0.25)

    df = inject_label_overlap(df, swap_fraction=0.06)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")

    import os
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/tuesday.csv", index=False)
    print("Saved to data/raw/tuesday.csv")


if __name__ == "__main__":
    generate_full_dataset()
