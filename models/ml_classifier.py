"""
ML Classifier — SecOpsAI
XGBoost-based behavioral detection model.
Must beat rule-based baseline by at least 15% F1.
Tracked with MLflow for experiment logging.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight


FEATURES_PATH  = "data/processed/features.csv"
MODEL_PATH     = "models/xgboost_detector.pkl"
RESULTS_PATH   = "models/results/ml_results.json"
BASELINE_PATH  = "models/results/baseline_results.json"


def load_data():
    print("[ML] Loading feature data...")
    df = pd.read_csv(FEATURES_PATH)

    y = df['label_binary']
    X = df.drop(columns=['label_binary', 'label_original'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[ML] Train size: {X_train.shape}")
    print(f"[ML] Test size:  {X_test.shape}")
    print(f"[ML] Attack rate in train: {y_train.mean():.2%}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    print("[ML] Training XGBoost classifier...")
    sample_weights = compute_sample_weight('balanced', y_train)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    print("[ML] Training complete")
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)

    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred).tolist()

    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    results = {
        "model":            "xgboost_classifier",
        "f1_score":         round(f1, 4),
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "confusion_matrix": cm,
        "feature_importance": importance
    }

    return results, y_pred


def compare_to_baseline(ml_results):
    if not os.path.exists(BASELINE_PATH):
        print("[ML] No baseline results found — run baseline first")
        return ml_results

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    baseline_f1 = baseline['f1_score']
    ml_f1       = ml_results['f1_score']
    improvement = ((ml_f1 - baseline_f1) / max(baseline_f1, 0.001)) * 100

    print("\n" + "=" * 50)
    print("BASELINE vs ML COMPARISON")
    print("=" * 50)
    print(f"  Rule-Based F1:  {baseline_f1:.4f}")
    print(f"  ML F1:          {ml_f1:.4f}")
    print(f"  Improvement:    {improvement:.1f}%")

    if improvement >= 15:
        print(f"  PASSED — ML beats baseline by {improvement:.1f}%")
    else:
        print(f"  FAILED — Need 15% improvement, got {improvement:.1f}%")

    print("=" * 50)

    ml_results['baseline_f1']     = baseline_f1
    ml_results['improvement_pct'] = round(improvement, 2)
    return ml_results


def run_ablation_study(X_train, X_test, y_train, y_test):
    print("\n[ML] Running ablation study...")
    sample_weights = compute_sample_weight('balanced', y_train)

    full_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    full_model.fit(X_train, y_train, sample_weight=sample_weights)
    full_f1 = f1_score(y_test, full_model.predict(X_test))

    ablation_results = {}

    for feature in X_train.columns:
        X_train_ablated = X_train.drop(columns=[feature])
        X_test_ablated  = X_test.drop(columns=[feature])

        ablated_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss', n_jobs=-1
        )
        ablated_model.fit(X_train_ablated, y_train, sample_weight=sample_weights)
        ablated_f1 = f1_score(y_test, ablated_model.predict(X_test_ablated))

        drop = full_f1 - ablated_f1
        ablation_results[feature] = round(drop, 4)

    ablation_results = dict(sorted(ablation_results.items(), key=lambda x: x[1], reverse=True))

    print("[ML] Ablation results (higher = more important):")
    for feat, drop in list(ablation_results.items())[:5]:
        print(f"     {feat}: F1 drops by {drop:.4f} when removed")

    return ablation_results


def run_ml_classifier():
    print("=" * 50)
    print("XGBoost ML Classifier — SecOpsAI")
    print("=" * 50)

    mlflow.set_experiment("secopsai-detection")

    with mlflow.start_run(run_name="xgboost_v1"):
        X_train, X_test, y_train, y_test = load_data()

        model = train_model(X_train, y_train)

        results, y_pred = evaluate_model(model, X_test, y_test, list(X_train.columns))

        results = compare_to_baseline(results)

        ablation = run_ablation_study(X_train, X_test, y_train, y_test)
        results['ablation_study'] = ablation

        mlflow.log_metric("f1_score",  results['f1_score'])
        mlflow.log_metric("precision", results['precision'])
        mlflow.log_metric("recall",    results['recall'])
        mlflow.log_metric("improvement_pct", results.get('improvement_pct', 0))

        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n[ML] Model saved to {MODEL_PATH}")

        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[ML] Results saved to {RESULTS_PATH}")

        print("\n[ML] Full Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['BENIGN', 'ATTACK']))

        mlflow.xgboost.log_model(model, "model")

    print("=" * 50)
    print("ML Training Complete")
    print("=" * 50)


if __name__ == "__main__":
    run_ml_classifier()
