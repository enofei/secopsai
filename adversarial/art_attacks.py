"""
Adversarial Robustness Testing — SecOpsAI
Attacks the trained XGBoost model using 5 distinct evasion
strategies covering both white-box and black-box threat models.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score, accuracy_score
from art.estimators.classification import XGBoostClassifier
from art.attacks.evasion import ZooAttack, BoundaryAttack, HopSkipJump


FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH    = "models/xgboost_detector.pkl"
RESULTS_PATH  = "adversarial/results/attack_results.json"


def load_model_and_data():
    print("[ADV] Loading model and data...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(FEATURES_PATH)
    y = df['label_binary']
    X = df.drop(columns=['label_binary', 'label_original'])

    np.random.seed(42)
    sample_idx = np.random.choice(X.index, size=200, replace=False)
    X_sample = X.loc[sample_idx].reset_index(drop=True)
    y_sample = y.loc[sample_idx].reset_index(drop=True)

    print(f"[ADV] Attack sample size: {X_sample.shape}")
    return model, X_sample, y_sample, list(X.columns)


def wrap_model_for_art(model, n_features):
    return XGBoostClassifier(
        model=model,
        nb_features=n_features,
        nb_classes=2,
        clip_values=(-10, 10)
    )


def evaluate_attack(art_classifier, X_adv, y_true, attack_name):
    y_pred_adv = np.argmax(art_classifier.predict(X_adv), axis=1)
    f1 = f1_score(y_true, y_pred_adv, zero_division=0)
    acc = accuracy_score(y_true, y_pred_adv)

    attack_mask = (y_true == 1).values
    if attack_mask.sum() > 0:
        evaded = (y_pred_adv[attack_mask] == 0).sum()
        evasion_rate = evaded / attack_mask.sum()
    else:
        evasion_rate = 0.0

    result = {
        "attack": attack_name,
        "f1_after_attack": round(float(f1), 4),
        "accuracy_after_attack": round(float(acc), 4),
        "evasion_rate": round(float(evasion_rate), 4)
    }
    print(f"[ADV] {attack_name}: F1={f1:.4f}, Evasion Rate={evasion_rate:.2%}")
    return result


def run_greedy_leaf_flip_attack(model, X_sample, y_sample, feature_names):
    """
    Greedy Leaf-Flip Attack — a white-box attack designed
    specifically for tree ensembles like XGBoost. Instead of
    using gradients (which trees don't have), it greedily tests
    perturbing each feature by small steps and keeps whichever
    single-feature change moves the prediction probability
    closest to the decision boundary. This mimics an attacker
    who has black-box query access and incrementally tunes
    inputs based on observed model confidence.
    """
    print("\n[ADV] Running Greedy Leaf-Flip Attack...")

    X_adv = X_sample.copy().values.astype(float)
    y_arr = y_sample.values
    attack_indices = np.where(y_arr == 1)[0]

    step_sizes = [0.1, 0.25, 0.5, 1.0, -0.1, -0.25, -0.5, -1.0]

    for idx in attack_indices:
        row = X_adv[idx].copy()
        current_prob = model.predict_proba(row.reshape(1, -1))[0][1]

        # Try perturbing each feature, keep the best improvement
        for _ in range(15):  # up to 15 greedy steps per sample
            best_feat = None
            best_step = None
            best_prob = current_prob

            for feat_idx in range(len(feature_names)):
                for step in step_sizes:
                    trial = row.copy()
                    trial[feat_idx] += step
                    prob = model.predict_proba(trial.reshape(1, -1))[0][1]
                    if prob < best_prob:
                        best_prob = prob
                        best_feat = feat_idx
                        best_step = step

            if best_feat is None:
                break  # No improvement found, stop early

            row[best_feat] += best_step
            current_prob = best_prob

            if current_prob < 0.5:
                break  # Successfully flipped to benign

        X_adv[idx] = row

    return X_adv


def run_zoo_attack(art_classifier, X_sample, y_sample):
    """
    ZOO Attack — fixed to use batch_size=1 as required by ART's
    current implementation for feature-vector inputs.
    """
    print("\n[ADV] Running ZOO Attack...")
    attack = ZooAttack(
        classifier=art_classifier,
        confidence=0.5,
        targeted=False,
        learning_rate=0.01,
        max_iter=15,
        binary_search_steps=2,
        initial_const=0.001,
        nb_parallel=5,
        batch_size=1,
        use_resize=False,
        use_importance=False
    )
    # Run on a smaller subset since batch_size=1 makes this slow
    X_small = X_sample.iloc[:30].values
    X_adv = attack.generate(x=X_small)
    return X_adv, X_small


def run_hopskipjump_attack(art_classifier, X_sample, y_sample):
    print("\n[ADV] Running HopSkipJump Attack...")
    attack = HopSkipJump(
        classifier=art_classifier,
        targeted=False,
        max_iter=10,
        max_eval=100,
        init_eval=10
    )
    X_adv = attack.generate(x=X_sample.values)
    return X_adv


def run_boundary_attack(art_classifier, X_sample, y_sample):
    print("\n[ADV] Running Boundary Attack...")
    attack = BoundaryAttack(
        estimator=art_classifier,
        targeted=False,
        max_iter=50,
        delta=0.01,
        epsilon=0.01
    )
    X_adv = attack.generate(x=X_sample.values)
    return X_adv


def run_manual_feature_perturbation(X_sample, y_sample, feature_names):
    print("\n[ADV] Running Manual Feature Perturbation Attack...")
    X_adv = X_sample.copy()
    key_features = ['Flow Bytes/s', 'Packet Length Mean',
                     'PSH Flag Count', 'Packet Length Std',
                     'Flow Packets/s']

    for feat in key_features:
        if feat in X_adv.columns:
            attack_mask = (y_sample == 1)
            noise = np.random.normal(0, 0.5, size=attack_mask.sum())
            X_adv.loc[attack_mask, feat] = (
                X_adv.loc[attack_mask, feat].values + noise
            )
    return X_adv.values


def run_adversarial_testing():
    print("=" * 50)
    print("Adversarial Robustness Testing — SecOpsAI")
    print("=" * 50)

    model, X_sample, y_sample, feature_names = load_model_and_data()
    art_classifier = wrap_model_for_art(model, len(feature_names))

    y_pred_clean = np.argmax(art_classifier.predict(X_sample.values), axis=1)
    clean_f1 = f1_score(y_sample, y_pred_clean, zero_division=0)
    print(f"\n[ADV] Clean (no attack) F1: {clean_f1:.4f}")

    results = {
        "clean_f1_before_attack": round(float(clean_f1), 4),
        "attacks": []
    }

    # Attack 1 — Greedy Leaf-Flip (white-box, tree-specific)
    try:
        X_adv = run_greedy_leaf_flip_attack(model, X_sample, y_sample, feature_names)
        result = evaluate_attack(art_classifier, X_adv, y_sample, "Greedy Leaf-Flip Attack")
        results["attacks"].append(result)
    except Exception as e:
        print(f"[ADV] Greedy Leaf-Flip Attack failed: {str(e)}")
        results["attacks"].append({"attack": "Greedy Leaf-Flip Attack", "error": str(e)})

    # Attack 2 — ZOO (black-box, query-based)
    try:
        X_adv_zoo, X_small = run_zoo_attack(art_classifier, X_sample, y_sample)
        y_small = y_sample.iloc[:30]
        result = evaluate_attack(art_classifier, X_adv_zoo, y_small, "ZOO Attack")
        results["attacks"].append(result)
    except Exception as e:
        print(f"[ADV] ZOO Attack failed: {str(e)}")
        results["attacks"].append({"attack": "ZOO Attack", "error": str(e)})

    # Attack 3 — HopSkipJump (black-box, decision-based)
    try:
        X_adv = run_hopskipjump_attack(art_classifier, X_sample, y_sample)
        result = evaluate_attack(art_classifier, X_adv, y_sample, "HopSkipJump Attack")
        results["attacks"].append(result)
    except Exception as e:
        print(f"[ADV] HopSkipJump Attack failed: {str(e)}")
        results["attacks"].append({"attack": "HopSkipJump Attack", "error": str(e)})

    # Attack 4 — Boundary Attack (black-box, decision-based)
    try:
        X_adv = run_boundary_attack(art_classifier, X_sample, y_sample)
        result = evaluate_attack(art_classifier, X_adv, y_sample, "Boundary Attack")
        results["attacks"].append(result)
    except Exception as e:
        print(f"[ADV] Boundary Attack failed: {str(e)}")
        results["attacks"].append({"attack": "Boundary Attack", "error": str(e)})

    # Attack 5 — Manual Feature Perturbation (simple realistic attacker)
    X_adv_manual = run_manual_feature_perturbation(X_sample, y_sample, feature_names)
    result = evaluate_attack(art_classifier, X_adv_manual, y_sample, "Manual Feature Perturbation")
    results["attacks"].append(result)

    os.makedirs("adversarial/results", exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[ADV] Results saved to {RESULTS_PATH}")
    print("=" * 50)
    print("Adversarial Testing Complete (Pre-Hardening)")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_adversarial_testing()
