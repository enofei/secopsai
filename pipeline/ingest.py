"""
Ingestion Pipeline — SecOpsAI
Reads raw CICIDS data, validates it, and saves a clean version.
"""

import pandas as pd
import numpy as np
import hashlib
import json
import os
from datetime import datetime

RAW_DATA_PATH = "data/raw/tuesday.csv"
PROCESSED_PATH = "data/processed/clean_data.csv"
INTEGRITY_LOG = "data/processed/integrity_log.json"

SELECTED_COLUMNS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Label'
]


def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def log_integrity(filepath, file_hash, row_count, label_counts):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "file": filepath,
        "sha256_hash": file_hash,
        "row_count": row_count,
        "label_distribution": label_counts
    }
    os.makedirs("data/processed", exist_ok=True)
    with open(INTEGRITY_LOG, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"[INTEGRITY] Log saved to {INTEGRITY_LOG}")
    print(f"[INTEGRITY] SHA-256: {file_hash}")


def load_and_clean(filepath):
    print(f"[INGEST] Loading {filepath}...")
    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"[INGEST] Raw shape: {df.shape}")
    print(f"[INGEST] Columns found: {list(df.columns[:5])}...")

    available = [c for c in SELECTED_COLUMNS if c in df.columns]
    df = df[available]

    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[INGEST] Dropped {before - after} rows with null values")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"[INGEST] Final shape after cleaning: {df.shape}")
    return df


def encode_labels(df):
    print("[INGEST] Encoding labels...")
    label_counts = df['Label'].value_counts().to_dict()
    print(f"[INGEST] Label distribution: {label_counts}")
    df['label_binary'] = (df['Label'] != 'BENIGN').astype(int)
    df['label_original'] = df['Label']
    df = df.drop(columns=['Label'])
    return df, label_counts


def run_ingestion():
    print("=" * 50)
    print("SecOpsAI Ingestion Pipeline Starting")
    print("=" * 50)

    print("[INTEGRITY] Computing file hash...")
    file_hash = compute_file_hash(RAW_DATA_PATH)

    df = load_and_clean(RAW_DATA_PATH)
    df, label_counts = encode_labels(df)
    log_integrity(RAW_DATA_PATH, file_hash, len(df), label_counts)

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[INGEST] Clean data saved to {PROCESSED_PATH}")
    print(f"[INGEST] Shape: {df.shape}")
    print("=" * 50)
    print("Ingestion Complete")
    print("=" * 50)
    return df


if __name__ == "__main__":
    run_ingestion()
