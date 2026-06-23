"""
Adversarial Hardening — SecOpsAI
Retrains the XGBoost model on a mix of original training data
plus adversarial examples generated from the attacks that
succeeded, so the model learns to correctly classify them.
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


FEATURES_PATH      = "data/processed/features.csv"
MODEL_PATH          = "models/xgboost_detector.pkl"
HARDENED_MODEL_PATH = "models/xgboost_detector_hardened.pkl"
RESULTS_PATH        = "adversarial/results/hardening_results.json"


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    y = df['label_binary']
    X = df.drop(columns=['label_binary', 'label_original'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def generate_adversarial_training_examples(model, X_train, y_train, feature_names):
    """
    Generates adversarial examples from the TRAINING set using
    the same attacks that succeeded against the original model
    (HopSkipJump, Boundary Attack), plus the greedy leaf-flip
    method. These get added back into training, correctly
    labeled as attacks, so the model learns these regions
    of feature space are still malicious.
    """
    print("[HARDEN] Generating adversarial examples for retraining...")

    art_classifier = XGBoostClassifier(
        model=model,
        nb_features=len(feature_names),
        nb_classes=2,
        clip_values=(-10, 10)
    )

    # Take a sample of attack rows from training data to perturb
    attack_idx = y_train[y_train == 1].index
    np.random.seed(42)
    sample_idx = np.random.choice(attack_idx, size=min(150, len(attack_idx)), replace=False)
    X_attack_sample = X_train.loc[sample_idx].reset_index(drop=True)

    adversarial_sets = []

    # HopSkipJump adversarial examples
    print("[HARDEN] Generating HopSkipJump adversarial examples...")
    hsj = HopSkipJump(classifier=art_classifier, targeted=False,
                       max_iter=10, max_eval=100, init_eval=10)
    X_adv_hsj = hsj.generate(x=X_attack_sample.values)
    adversarial_sets.append(X_adv_hsj)

    # Boundary Attack adversarial examples
    print("[HARDEN] Generating Boundary Attack adversarial examples...")
    boundary = BoundaryAttack(estimator=art_classifier, targeted=False,
                               max_iter=50, delta=0.01, epsilon=0.01)
    X_adv_boundary = boundary.generate(x=X_attack_sample.values)
    adversarial_sets.append(X_adv_boundary)

    # Combine all adversarial examples — they keep their ORIGINAL
    # label of 1 (attack), teaching the model these perturbed
    # versions are still malicious
    X_adv_combined = np.vstack(adversarial_sets)
    y_adv_combined = np.ones(len(X_adv_combined))

    print(f"[HARDEN] Generated {len(X_adv_combined)} adversarial training examples")
    return X_adv_combined, y_adv_combined


def retrain_with_adversarial_examples(X_train, y_train, X_adv, y_adv, feature_names):
    """
    Combines original training data with adversarial examples
    and retrains the model from scratch on the augmented set.
    """
    print("[HARDEN] Retraining model with adversarial examples...")

    X_adv_df = pd.DataFrame(X_adv, columns=feature_names)
    y_adv_series = pd.Series(y_adv)

    X_combined = pd.concat([X_train.reset_index(drop=True), X_adv_df], ignore_index=True)
    y_combined = pd.concat([y_train.reset_index(drop=True), y_adv_series], ignore_index=True)

    print(f"[HARDEN] Original training size: {len(X_train)}")
    print(f"[HARDEN] Augmented training size: {len(X_combined)}")

    sample_weights = compute_sample_weight('balanced', y_combined)

    hardened_model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    hardened_model.fit(X_combined, y_combined, sample_weight=sample_weights, verbose=False)

    print("[HARDEN] Hardened model training complete")
    return hardened_model


def evaluate_clean_performance(model, X_test, y_test):
    """Confirms hardening didn't destroy normal detection ability."""
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"[HARDEN] Clean test F1 after hardening: {f1:.4f}")
    print(f"[HARDEN] Clean test accuracy after hardening: {acc:.4f}")
    return f1, acc


def run_hardening():
    print("=" * 50)
    print("Adversarial Hardening — SecOpsAI")
    print("=" * 50)

    X_train, X_test, y_train, y_test = load_data()
    feature_names = list(X_train.columns)

    with open(MODEL_PATH, 'rb') as f:
        original_model = pickle.load(f)

    X_adv, y_adv = generate_adversarial_training_examples(
        original_model, X_train, y_train, feature_names
    )

    hardened_model = retrain_with_adversarial_examples(
        X_train, y_train, X_adv, y_adv, feature_names
    )

    clean_f1, clean_acc = evaluate_clean_performance(hardened_model, X_test, y_test)

    os.makedirs("models", exist_ok=True)
    with open(HARDENED_MODEL_PATH, 'wb') as f:
        pickle.dump(hardened_model, f)
    print(f"[HARDEN] Hardened model saved to {HARDENED_MODEL_PATH}")

    results = {
        "clean_f1_after_hardening": round(float(clean_f1), 4),
        "clean_accuracy_after_hardening": round(float(clean_acc), 4),
        "adversarial_examples_added": len(X_adv)
    }

    os.makedirs("adversarial/results", exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[HARDEN] Results saved to {RESULTS_PATH}")
    print("=" * 50)
    print("Hardening Complete")
    print("=" * 50)


if __name__ == "__main__":
    run_hardening()
