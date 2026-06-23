"""
Iterative Adversarial Hardening — SecOpsAI (v2)
Performs 2 rounds of adversarial training. Round 2 generates
fresh adversarial examples against the Round 1 hardened model,
specifically targeting any new blind spots it introduced.
Also increases adversarial example volume significantly and
includes Greedy Leaf-Flip examples, which fully evaded v1 hardening.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import HopSkipJump, BoundaryAttack


FEATURES_PATH        = "data/processed/features.csv"
ORIGINAL_MODEL_PATH  = "models/xgboost_detector.pkl"
HARDENED_V2_PATH     = "models/xgboost_detector_hardened_v2.pkl"
RESULTS_PATH         = "adversarial/results/hardening_v2_results.json"


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    y = df['label_binary']
    X = df.drop(columns=['label_binary', 'label_original'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def greedy_leaf_flip(model, X_attack, feature_names):
    """Generates Greedy Leaf-Flip adversarial examples against a given model."""
    X_adv = X_attack.copy().values.astype(float)
    step_sizes = [0.1, 0.25, 0.5, 1.0, -0.1, -0.25, -0.5, -1.0]

    for i in range(len(X_adv)):
        row = X_adv[i].copy()
        current_prob = model.predict_proba(row.reshape(1, -1))[0][1]

        for _ in range(15):
            best_feat, best_step, best_prob = None, None, current_prob
            for feat_idx in range(len(feature_names)):
                for step in step_sizes:
                    trial = row.copy()
                    trial[feat_idx] += step
                    prob = model.predict_proba(trial.reshape(1, -1))[0][1]
                    if prob < best_prob:
                        best_prob, best_feat, best_step = prob, feat_idx, step
            if best_feat is None:
                break
            row[best_feat] += best_step
            current_prob = best_prob
            if current_prob < 0.5:
                break
        X_adv[i] = row
    return X_adv


def generate_adversarial_batch(model, X_train, y_train, feature_names, n_samples):
    """
    Generates a larger, diverse batch of adversarial examples
    using all 3 attacks that previously achieved full evasion:
    Greedy Leaf-Flip, HopSkipJump, and Boundary Attack.
    """
    art_classifier = XGBoostClassifier(
        model=model, nb_features=len(feature_names),
        nb_classes=2, clip_values=(-10, 10)
    )

    attack_idx = y_train[y_train == 1].index
    np.random.seed(np.random.randint(0, 10000))  # vary sample each round
    sample_idx = np.random.choice(attack_idx, size=min(n_samples, len(attack_idx)), replace=False)
    X_attack_sample = X_train.loc[sample_idx].reset_index(drop=True)

    print(f"[HARDEN-V2]   Generating Greedy Leaf-Flip examples ({len(X_attack_sample)} samples)...")
    X_adv_leaf = greedy_leaf_flip(model, X_attack_sample, feature_names)

    print(f"[HARDEN-V2]   Generating HopSkipJump examples...")
    hsj = HopSkipJump(classifier=art_classifier, targeted=False,
                       max_iter=10, max_eval=100, init_eval=10)
    X_adv_hsj = hsj.generate(x=X_attack_sample.values)

    print(f"[HARDEN-V2]   Generating Boundary Attack examples...")
    boundary = BoundaryAttack(estimator=art_classifier, targeted=False,
                               max_iter=50, delta=0.01, epsilon=0.01)
    X_adv_boundary = boundary.generate(x=X_attack_sample.values)

    X_adv_combined = np.vstack([X_adv_leaf, X_adv_hsj, X_adv_boundary])
    y_adv_combined = np.ones(len(X_adv_combined))

    return X_adv_combined, y_adv_combined


def train_xgb(X, y):
    sample_weights = compute_sample_weight('balanced', y)
    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    model.fit(X, y, sample_weight=sample_weights, verbose=False)
    return model


def evaluate_clean(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"[HARDEN-V2] {label} — Clean F1: {f1:.4f}, Accuracy: {acc:.4f}")
    return f1, acc


def run_iterative_hardening():
    print("=" * 50)
    print("Iterative Adversarial Hardening (2 Rounds) — SecOpsAI")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()
    feature_names = list(X_train.columns)

    with open(ORIGINAL_MODEL_PATH, 'rb') as f:
        original_model = pickle.load(f)

    # ── ROUND 1 ──────────────────────────────────────────────
    print("\n[HARDEN-V2] ROUND 1 — generating adversarial examples against ORIGINAL model")
    X_adv_r1, y_adv_r1 = generate_adversarial_batch(
        original_model, X_train, y_train, feature_names, n_samples=400
    )
    print(f"[HARDEN-V2] Round 1 generated {len(X_adv_r1)} adversarial examples")

    X_adv_r1_df = pd.DataFrame(X_adv_r1, columns=feature_names)
    y_adv_r1_series = pd.Series(y_adv_r1)

    X_round1 = pd.concat([X_train.reset_index(drop=True), X_adv_r1_df], ignore_index=True)
    y_round1 = pd.concat([y_train.reset_index(drop=True), y_adv_r1_series], ignore_index=True)

    print("[HARDEN-V2] Training Round 1 hardened model...")
    model_r1 = train_xgb(X_round1, y_round1)
    f1_r1, acc_r1 = evaluate_clean(model_r1, X_test, y_test, "Round 1")

    # ── ROUND 2 ──────────────────────────────────────────────
    print("\n[HARDEN-V2] ROUND 2 — generating FRESH adversarial examples against ROUND 1 model")
    X_adv_r2, y_adv_r2 = generate_adversarial_batch(
        model_r1, X_train, y_train, feature_names, n_samples=400
    )
    print(f"[HARDEN-V2] Round 2 generated {len(X_adv_r2)} adversarial examples")

    X_adv_r2_df = pd.DataFrame(X_adv_r2, columns=feature_names)
    y_adv_r2_series = pd.Series(y_adv_r2)

    X_round2 = pd.concat([X_round1, X_adv_r2_df], ignore_index=True)
    y_round2 = pd.concat([y_round1, y_adv_r2_series], ignore_index=True)

    print("[HARDEN-V2] Training Round 2 hardened model (final)...")
    model_r2 = train_xgb(X_round2, y_round2)
    f1_r2, acc_r2 = evaluate_clean(model_r2, X_test, y_test, "Round 2 (Final)")

    # Save final model
    os.makedirs("models", exist_ok=True)
    with open(HARDENED_V2_PATH, 'wb') as f:
        pickle.dump(model_r2, f)
    print(f"\n[HARDEN-V2] Final hardened model saved to {HARDENED_V2_PATH}")

    results = {
        "round1_clean_f1": round(float(f1_r1), 4),
        "round1_clean_accuracy": round(float(acc_r1), 4),
        "round1_adversarial_examples": len(X_adv_r1),
        "round2_clean_f1": round(float(f1_r2), 4),
        "round2_clean_accuracy": round(float(acc_r2), 4),
        "round2_adversarial_examples": len(X_adv_r2),
        "total_training_size": len(X_round2)
    }

    os.makedirs("adversarial/results", exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[HARDEN-V2] Results saved to {RESULTS_PATH}")
    print("=" * 50)
    print("Iterative Hardening Complete")
    print("=" * 50)


if __name__ == "__main__":
    run_iterative_hardening()
