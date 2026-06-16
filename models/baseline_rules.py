"""
Rule-Based Baseline Detector — SecOpsAI
Implements simple threshold rules for each attack type.
We build this FIRST so we have a benchmark to beat.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (classification_report, f1_score,
                             precision_score, recall_score,
                             confusion_matrix)
import json
import os


FEATURES_PATH = "data/processed/features.csv"
RESULTS_PATH  = "models/results/baseline_results.json"


def apply_rules(df):
    predictions = pd.Series(0, index=df.index)

    # Rule 1: C2 Beaconing
    c2_mask = (
        (df['Flow Bytes/s'] < 250) &
        (df['Flow Packets/s'] < 0.05) &
        (df['Packet Length Std'] < 25)
    )
    predictions[c2_mask] = 1

    # Rule 2: Lateral Movement
    lateral_mask = (
        (df['SYN Flag Count'] >= 1.5) &
        (df['Total Backward Packets'] < 0.5) &
        (df['Flow Duration'] < 1000)
    )
    predictions[lateral_mask] = 1

    # Rule 3: DNS Tunnelling
    dns_mask = (
        (df['byte_ratio'] > 0.85) &
        (df['Flow Packets/s'] < 150) &
        (df['Average Packet Size'] > 120)
    )
    predictions[dns_mask] = 1

    # Rule 4: Slow Exfiltration
    exfil_mask = (
        (df['Flow Bytes/s'] < 12) &
        (df['flow_speed'] < 12) &
        (df['Flow Duration'] > 5000000)
    )
    predictions[exfil_mask] = 1

    return predictions


def evaluate_rules(y_true, y_pred):
    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    cm        = confusion_matrix(y_true, y_pred).tolist()

    results = {
        "model": "rule_based_baseline",
        "f1_score":        round(f1, 4),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "confusion_matrix": cm
    }

    os.makedirs("models/results", exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_baseline():
    print("=" * 50)
    print("Rule-Based Baseline Detector")
    print("=" * 50)

    df = pd.read_csv(FEATURES_PATH)
    y_true = df['label_binary']
    X = df.drop(columns=['label_binary', 'label_original'])

    print("[BASELINE] Applying detection rules...")
    y_pred = apply_rules(X)

    results = evaluate_rules(y_true, y_pred)

    print(f"\n[BASELINE] Results:")
    print(f"  F1 Score:  {results['f1_score']}")
    print(f"  Precision: {results['precision']}")
    print(f"  Recall:    {results['recall']}")
    print(f"\n[BASELINE] Full Report:")
    print(classification_report(y_true, y_pred,
          target_names=['BENIGN', 'ATTACK']))
    print(f"\n[BASELINE] Results saved to {RESULTS_PATH}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_baseline()
