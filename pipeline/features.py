"""
Feature Engineering — SecOpsAI
Transforms clean network flow data into ML-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os


PROCESSED_PATH = "data/processed/clean_data.csv"
FEATURES_PATH = "data/processed/features.csv"
SCALER_PATH = "data/processed/scaler.pkl"


def compute_byte_ratio(df):
    total = df['Total Length of Fwd Packets'] + df['Total Length of Bwd Packets']
    total = total.replace(0, 1)
    df['byte_ratio'] = df['Total Length of Fwd Packets'] / total
    return df


def compute_timing_entropy(df):
    mean = df['Flow IAT Mean'].replace(0, 1)
    df['timing_entropy'] = df['Flow IAT Std'] / mean
    return df


def compute_packet_size_consistency(df):
    mean = df['Packet Length Mean'].replace(0, 1)
    df['size_consistency'] = df['Packet Length Std'] / mean
    return df


def compute_connection_symmetry(df):
    total_packets = df['Total Fwd Packets'] + df['Total Backward Packets']
    total_packets = total_packets.replace(0, 1)
    df['connection_symmetry'] = df['Total Fwd Packets'] / total_packets
    return df


def compute_flow_speed(df):
    duration = df['Flow Duration'].replace(0, 1)
    total_bytes = (df['Total Length of Fwd Packets'] +
                   df['Total Length of Bwd Packets'])
    df['flow_speed'] = total_bytes / duration
    return df


def select_final_features(df):
    feature_columns = [
        'byte_ratio', 'timing_entropy', 'size_consistency',
        'connection_symmetry', 'flow_speed',
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
        'Fwd IAT Mean', 'Bwd IAT Mean', 'Packet Length Mean',
        'Packet Length Std', 'Packet Length Variance', 'Average Packet Size',
        'SYN Flag Count', 'ACK Flag Count', 'PSH Flag Count', 'FIN Flag Count',
    ]
    available = [c for c in feature_columns if c in df.columns]
    return df[available]


def scale_features(X_df, fit=True):
    scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X_df)
        os.makedirs("data/processed", exist_ok=True)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"[FEATURES] Scaler saved to {SCALER_PATH}")
    else:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X_df)

    return pd.DataFrame(X_scaled, columns=X_df.columns)


def run_feature_engineering():
    print("=" * 50)
    print("SecOpsAI Feature Engineering Starting")
    print("=" * 50)

    print(f"[FEATURES] Loading {PROCESSED_PATH}...")
    df = pd.read_csv(PROCESSED_PATH)
    print(f"[FEATURES] Loaded shape: {df.shape}")

    y = df['label_binary']
    label_original = df['label_original']

    df = df.drop(columns=['label_binary', 'label_original'])

    print("[FEATURES] Engineering features...")
    df = compute_byte_ratio(df)
    df = compute_timing_entropy(df)
    df = compute_packet_size_consistency(df)
    df = compute_connection_symmetry(df)
    df = compute_flow_speed(df)

    X = select_final_features(df)
    print(f"[FEATURES] Feature set shape: {X.shape}")
    print(f"[FEATURES] Features: {list(X.columns)}")

    X_scaled = scale_features(X, fit=True)

    X_scaled['label_binary'] = y.values
    X_scaled['label_original'] = label_original.values

    X_scaled.to_csv(FEATURES_PATH, index=False)
    print(f"[FEATURES] Feature data saved to {FEATURES_PATH}")

    print("\n[FEATURES] Class distribution:")
    print(y.value_counts())
    print(f"\n[FEATURES] Attack types:")
    print(label_original.value_counts())

    print("=" * 50)
    print("Feature Engineering Complete")
    print("=" * 50)

    return X_scaled


if __name__ == "__main__":
    run_feature_engineering()
