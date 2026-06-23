"""
Post-Hardening Adversarial Testing — SecOpsAI
Re-runs the same 5 attacks against the hardened model to measure
improvement. This produces the before/after comparison required
for the Adversarial Robustness Report.
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


FEATURES_PATH        = "data/processed/features.csv"
HARDENED_MODEL_PATH  = "models/xgboost_detector_hardened.pkl"
PRE_RESULTS_PATH     = "adversarial/results/attack_results.json"
POST_RESULTS_PATH    = "adversarial/results/attack_results_post_hardening.json"
COMPARISON_PATH      = "adversarial/results/hardening_comparison.json"


def load_model_and_data():
    print("[ADV-POST] Loading hardened model and data...")
    with open(HARDENED_MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv(FEATURES_PATH)
    y = df['label_binary']
    X = df.drop(columns=['label_binary', 'label_original'])

    # SAME random seed as pre-hardening test — identical sample
    # so the comparison is apples-to-apples
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
    print(f"[ADV-POST] {attack_name}: F1={f1:.4f}, Evasion Rate={evasion_rate:.2%}")
    return result


def run_greedy_leaf_flip_attack(model, X_sample, y_sample, feature_names):
    print("\n[ADV-POST] Running Greedy Leaf-Flip Attack...")
    X_adv = X_sample.copy().values.astype(float)
    y_arr = y_sample.values
    attack_indices = np.where(y_arr == 1)[0]
    step_sizes = [0.1, 0.25, 0.5, 1.0, -0.1, -0.25, -0.5, -1.0]

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


def run_zoo_attack(art_classifier, X_sample, y_sample):
    print("\n[ADV-POST] Running ZOO Attack...")
    attack = ZooAttack(
        classifier=art_classifier, confidence=0.5, targeted=False,
        learning_rate=0.01, max_iter=15, binary_search_steps=2,
        initial_const=0.001, nb_parallel=5, batch_size=1,
        use_resize=False, use_importance=False
    )
    X_small = X_sample.iloc[:30].values
    X_adv = attack.generate(x=X_small)
    return X_adv, X_small


def run_hopskipjump_attack(art_classifier, X_sample, y_sample):
    print("\n[ADV-POST] Running HopSkipJump Attack...")
    attack = HopSkipJump(classifier=art_classifier, targeted=False,
                          max_iter=10, max_eval=100, init_eval=10)
    return attack.generate(x=X_sample.values)


def run_boundary_attack(art_classifier, X_sample, y_sample):
    print("\n[ADV-POST] Running Boundary Attack...")
    attack = BoundaryAttack(estimator=art_classifier, targeted=False,
                             max_iter=50, delta=0.01, epsilon=0.01)
    return attack.generate(x=X_sample.values)


def run_manual_feature_perturbation(X_sample, y_sample, feature_names):
    print("\n[ADV-POST] Running Manual Feature Perturbation Attack...")
    X_adv = X_sample.copy()
    key_features = ['Flow Bytes/s', 'Packet Length Mean',
                     'PSH Flag Count', 'Packet Length Std', 'Flow Packets/s']
    for feat in key_features:
        if feat in X_adv.columns:
            attack_mask = (y_sample == 1)
            noise = np.random.normal(0, 0.5, size=attack_mask.sum())
            X_adv.loc[attack_mask, feat] = X_adv.loc[attack_mask, feat].values + noise
    return X_adv.values


def run_post_hardening_testing():
    print("=" * 50)
    print("Post-Hardening Adversarial Testing — SecOpsAI")
    print("=" * 50)

    model, X_sample, y_sample, feature_names = load_model_and_data()
    art_classifier = wrap_model_for_art(model, len(feature_names))

    y_pred_clean = np.argmax(art_classifier.predict(X_sample.values), axis=1)
    clean_f1 = f1_score(y_sample, y_pred_clean, zero_division=0)
    print(f"\n[ADV-POST] Clean (no attack) F1: {clean_f1:.4f}")

    results = {"clean_f1_before_attack": round(float(clean_f1), 4), "attacks": []}

    try:
        X_adv = run_greedy_leaf_flip_attack(model, X_sample, y_sample, feature_names)
        results["attacks"].append(evaluate_attack(art_classifier, X_adv, y_sample, "Greedy Leaf-Flip Attack"))
    except Exception as e:
        results["attacks"].append({"attack": "Greedy Leaf-Flip Attack", "error": str(e)})

    try:
        X_adv_zoo, X_small = run_zoo_attack(art_classifier, X_sample, y_sample)
        y_small = y_sample.iloc[:30]
        results["attacks"].append(evaluate_attack(art_classifier, X_adv_zoo, y_small, "ZOO Attack"))
    except Exception as e:
        results["attacks"].append({"attack": "ZOO Attack", "error": str(e)})

    try:
        X_adv = run_hopskipjump_attack(art_classifier, X_sample, y_sample)
        results["attacks"].append(evaluate_attack(art_classifier, X_adv, y_sample, "HopSkipJump Attack"))
    except Exception as e:
        results["attacks"].append({"attack": "HopSkipJump Attack", "error": str(e)})

    try:
        X_adv = run_boundary_attack(art_classifier, X_sample, y_sample)
        results["attacks"].append(evaluate_attack(art_classifier, X_adv, y_sample, "Boundary Attack"))
    except Exception as e:
        results["attacks"].append({"attack": "Boundary Attack", "error": str(e)})

    X_adv_manual = run_manual_feature_perturbation(X_sample, y_sample, feature_names)
    results["attacks"].append(evaluate_attack(art_classifier, X_adv_manual, y_sample, "Manual Feature Perturbation"))

    os.makedirs("adversarial/results", exist_ok=True)
    with open(POST_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    # Build comparison report
    with open(PRE_RESULTS_PATH) as f:
        pre_results = json.load(f)

    comparison = {"attacks_survived_post_hardening": 0, "comparison": []}

    for pre_attack, post_attack in zip(pre_results["attacks"], results["attacks"]):
        pre_evasion = pre_attack.get("evasion_rate", None)
        post_evasion = post_attack.get("evasion_rate", None)
        survived = (post_evasion is not None and post_evasion < 0.5)

        if survived:
            comparison["attacks_survived_post_hardening"] += 1

        comparison["comparison"].append({
            "attack": pre_attack["attack"],
            "evasion_rate_before": pre_evasion,
            "evasion_rate_after": post_evasion,
            "survived_post_hardening": survived
        })

    with open(COMPARISON_PATH, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\n[ADV-POST] Results saved to {POST_RESULTS_PATH}")
    print(f"[ADV-POST] Comparison saved to {COMPARISON_PATH}")
    print(f"\n[ADV-POST] Attacks survived (evasion < 50%): {comparison['attacks_survived_post_hardening']} / 5")
    print("=" * 50)
    print("Post-Hardening Testing Complete")
    print("=" * 50)


if __name__ == "__main__":
    run_post_hardening_testing()
