"""
Post-Hardening v2 Adversarial Testing — SecOpsAI
Final attack run against the iteratively hardened model.
Produces the definitive before/after comparison for the report.
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


FEATURES_PATH    = "data/processed/features.csv"
MODEL_V2_PATH    = "models/xgboost_detector_hardened_v2.pkl"
PRE_RESULTS_PATH = "adversarial/results/attack_results.json"
POST_V2_PATH     = "adversarial/results/attack_results_post_hardening_v2.json"
FINAL_COMPARISON = "adversarial/results/final_comparison.json"


def load_model_and_data():
    print("[FINAL] Loading v2 hardened model...")
    with open(MODEL_V2_PATH, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(FEATURES_PATH)
    y = df['label_binary']
    X = df.drop(columns=['label_binary', 'label_original'])

    # Same seed as all previous tests — identical sample for fair comparison
    np.random.seed(42)
    sample_idx = np.random.choice(X.index, size=200, replace=False)
    X_sample = X.loc[sample_idx].reset_index(drop=True)
    y_sample = y.loc[sample_idx].reset_index(drop=True)

    return model, X_sample, y_sample, list(X.columns)


def wrap_model_for_art(model, n_features):
    return XGBoostClassifier(
        model=model, nb_features=n_features,
        nb_classes=2, clip_values=(-10, 10)
    )


def evaluate_attack(art_classifier, X_adv, y_true, attack_name):
    y_pred_adv = np.argmax(art_classifier.predict(X_adv), axis=1)
    f1  = f1_score(y_true, y_pred_adv, zero_division=0)
    acc = accuracy_score(y_true, y_pred_adv)

    attack_mask = (np.array(y_true) == 1)
    if attack_mask.sum() > 0:
        evaded = (y_pred_adv[attack_mask] == 0).sum()
        evasion_rate = evaded / attack_mask.sum()
    else:
        evasion_rate = 0.0

    result = {
        "attack": attack_name,
        "f1_after_attack":       round(float(f1), 4),
        "accuracy_after_attack": round(float(acc), 4),
        "evasion_rate":          round(float(evasion_rate), 4)
    }
    print(f"[FINAL] {attack_name}: F1={f1:.4f}, Evasion Rate={evasion_rate:.2%}")
    return result


def run_greedy_leaf_flip(model, X_sample, y_sample, feature_names):
    print("\n[FINAL] Running Greedy Leaf-Flip Attack...")
    X_adv = X_sample.copy().values.astype(float)
    step_sizes = [0.1, 0.25, 0.5, 1.0, -0.1, -0.25, -0.5, -1.0]
    attack_indices = np.where(np.array(y_sample) == 1)[0]

    for idx in attack_indices:
        row = X_adv[idx].copy()
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
        X_adv[idx] = row
    return X_adv


def run_zoo_attack(art_classifier, X_sample):
    print("\n[FINAL] Running ZOO Attack...")
    attack = ZooAttack(
        classifier=art_classifier, confidence=0.5, targeted=False,
        learning_rate=0.01, max_iter=15, binary_search_steps=2,
        initial_const=0.001, nb_parallel=5, batch_size=1,
        use_resize=False, use_importance=False
    )
    X_small = X_sample.iloc[:30].values
    return attack.generate(x=X_small), X_small


def run_hopskipjump(art_classifier, X_sample):
    print("\n[FINAL] Running HopSkipJump Attack...")
    attack = HopSkipJump(
        classifier=art_classifier, targeted=False,
        max_iter=10, max_eval=100, init_eval=10
    )
    return attack.generate(x=X_sample.values)


def run_boundary_attack(art_classifier, X_sample):
    print("\n[FINAL] Running Boundary Attack...")
    attack = BoundaryAttack(
        estimator=art_classifier, targeted=False,
        max_iter=50, delta=0.01, epsilon=0.01
    )
    return attack.generate(x=X_sample.values)


def run_manual_perturbation(X_sample, y_sample):
    print("\n[FINAL] Running Manual Feature Perturbation Attack...")
    X_adv = X_sample.copy()
    key_features = ['Flow Bytes/s', 'Packet Length Mean',
                     'PSH Flag Count', 'Packet Length Std', 'Flow Packets/s']
    for feat in key_features:
        if feat in X_adv.columns:
            attack_mask = (y_sample == 1)
            noise = np.random.normal(0, 0.5, size=attack_mask.sum())
            X_adv.loc[attack_mask, feat] = X_adv.loc[attack_mask, feat].values + noise
    return X_adv.values


def build_final_comparison(pre_results, post_results):
    """
    Produces the definitive before/after comparison table
    across all 3 stages: original, v1 hardened, v2 hardened.
    """
    comparison = {
        "requirement": "Survive at least 3 of 5 attacks (evasion rate < 50%)",
        "attacks_survived_post_hardening_v2": 0,
        "comparison": []
    }

    for pre, post in zip(pre_results["attacks"], post_results["attacks"]):
        pre_evasion  = pre.get("evasion_rate")
        post_evasion = post.get("evasion_rate")
        survived     = (post_evasion is not None and post_evasion < 0.5)

        if survived:
            comparison["attacks_survived_post_hardening_v2"] += 1

        comparison["comparison"].append({
            "attack":                pre["attack"],
            "evasion_rate_original": pre_evasion,
            "evasion_rate_v2":       post_evasion,
            "improvement":           round((pre_evasion or 0) - (post_evasion or 0), 4),
            "survived":              survived,
            "verdict":               "PASS" if survived else "FAIL"
        })

    total = comparison["attacks_survived_post_hardening_v2"]
    comparison["overall_verdict"] = "PASSED" if total >= 3 else "NEEDS_MORE_HARDENING"
    return comparison


def run_final_testing():
    print("=" * 55)
    print("Final Post-Hardening v2 Attack Test — SecOpsAI")
    print("=" * 55)

    model, X_sample, y_sample, feature_names = load_model_and_data()
    art_classifier = wrap_model_for_art(model, len(feature_names))

    y_clean_pred = np.argmax(art_classifier.predict(X_sample.values), axis=1)
    clean_f1 = f1_score(y_sample, y_clean_pred, zero_division=0)
    print(f"\n[FINAL] Clean (no attack) F1: {clean_f1:.4f}")

    results = {"clean_f1": round(float(clean_f1), 4), "attacks": []}

    # Attack 1 — Greedy Leaf-Flip
    try:
        X_adv = run_greedy_leaf_flip(model, X_sample, y_sample, feature_names)
        results["attacks"].append(evaluate_attack(art_classifier, X_adv, y_sample, "Greedy Leaf-Flip Attack"))
    except Exception as e:
        results["attacks"].append({"attack": "Greedy Leaf-Flip Attack", "error": str(e)})

    # Attack 2 — ZOO
    try:
        X_adv_zoo, X_small = run_zoo_attack(art_classifier, X_sample)
        y_small = y_sample.iloc[:30]
        results["attacks"].append(evaluate_attack(art_classifier, X_adv_zoo, y_small, "ZOO Attack"))
    except Exception as e:
        results["attacks"].append({"attack": "ZOO Attack", "error": str(e)})

    # Attack 3 — HopSkipJump
    try:
        X_adv = run_hopskipjump(art_classifier, X_sample)
        results["attacks"].append(evaluate_attack(art_classifier, X_adv, y_sample, "HopSkipJump Attack"))
    except Exception as e:
        results["attacks"].append({"attack": "HopSkipJump Attack", "error": str(e)})

    # Attack 4 — Boundary Attack
    try:
        X_adv = run_boundary_attack(art_classifier, X_sample)
        results["attacks"].append(evaluate_attack(art_classifier, X_adv, y_sample, "Boundary Attack"))
    except Exception as e:
        results["attacks"].append({"attack": "Boundary Attack", "error": str(e)})

    # Attack 5 — Manual Perturbation
    X_adv_manual = run_manual_perturbation(X_sample, y_sample)
    results["attacks"].append(evaluate_attack(art_classifier, X_adv_manual, y_sample, "Manual Feature Perturbation"))

    os.makedirs("adversarial/results", exist_ok=True)
    with open(POST_V2_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    # Load pre-hardening results and build final comparison
    with open(PRE_RESULTS_PATH) as f:
        pre_results = json.load(f)

    comparison = build_final_comparison(pre_results, results)

    with open(FINAL_COMPARISON, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print final summary table
    print("\n" + "=" * 55)
    print("FINAL ADVERSARIAL ROBUSTNESS SUMMARY")
    print("=" * 55)
    print(f"{'Attack':<30} {'Before':>8} {'After':>8} {'Verdict':>8}")
    print("-" * 55)
    for entry in comparison["comparison"]:
        before = f"{entry['evasion_rate_original']:.0%}" if entry['evasion_rate_original'] is not None else "N/A"
        after  = f"{entry['evasion_rate_v2']:.0%}"       if entry['evasion_rate_v2']       is not None else "N/A"
        print(f"{entry['attack']:<30} {before:>8} {after:>8} {entry['verdict']:>8}")
    print("-" * 55)
    print(f"Attacks survived: {comparison['attacks_survived_post_hardening_v2']} / 5")
    print(f"Overall verdict:  {comparison['overall_verdict']}")
    print("=" * 55)


if __name__ == "__main__":
    run_final_testing()
