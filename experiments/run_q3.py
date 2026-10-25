

import os
import sys
import json
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import load_and_prepare_data  # my roll no is M26MAC003 --seed- 26003
from src.bagging import CustomBaggingClassifier
from src.boosting import CustomSAMMEAdaBoost
from src.metrics import compute_bias_variance
from src.utils import (
    setup_logger,
    plot_q3_bias_variance_depth,
    plot_q3_decomposition_bars
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_resampled_train_sets(X_all, y_all, train_idx, seed=26003, R=30):
    rng = np.random.default_rng(seed + 999)
    train_target = y_all[train_idx]
    unique_identities = np.unique(train_target)
    
    identity_to_train_indices = {}
    for c in unique_identities:
        identity_to_train_indices[c] = train_idx[np.where(train_target == c)[0]]

    resampled_sets = []
    for r in range(R):
        rep_indices = []
        for c in unique_identities:
            orig_6 = identity_to_train_indices[c]
            sampled_6 = rng.choice(orig_6, size=6, replace=True)
            rep_indices.extend(sampled_6)
        
        rep_idx = np.array(rep_indices, dtype=int)
        resampled_sets.append((X_all[rep_idx], y_all[rep_idx]))
        
    return resampled_sets


def run_q3():
    log, log_file = setup_logger("logs/q3_bias_variance.log")
    log("=" * 80)
    log("QUESTION 3: BIAS AND VARIANCE AS MEASURED QUANTITIES")
    log("Student: Bhisham Sharma | Roll Number: M26MAC003")
    log("=" * 80)

    # 1. Load data
    data = load_and_prepare_data(roll_number="M26MAC003")
    X_test, y_test = data["X_test"], data["y_test"]
    X_test_masked = data["X_test_masked"]
    X_all, y_all = data["X_all"], data["y_all"]
    train_idx = data["train_idx"]
    seed = data["seed"]

    # Load frozen settings from Q1 and Q2
    with open("data/q1_summary.json", "r") as f:
        q1_summary = json.load(f)
    best_tree_params = q1_summary["best_tree_params"]
    final_B = q1_summary["final_B"]

    with open("data/q2_summary.json", "r") as f:
        q2_summary = json.load(f)
    best_boost_depth = q2_summary["best_depth"]
    best_boost_rounds = q2_summary["best_rounds"]

    log("Selected Frozen Settings:")
    log(f"   Single Tree:  max_depth={best_tree_params['max_depth']}, min_samples_leaf={best_tree_params['min_samples_leaf']}")
    log(f"   Final Bagging: B={final_B}")
    log(f"   Selected Boosting: depth={best_boost_depth}, rounds={best_boost_rounds}\n")

    R = 30
    log(f"Generating {R} resampled training sets (identity-stratified with replacement)...")
    resampled_sets = generate_resampled_train_sets(X_all, y_all, train_idx, seed=seed, R=R)

    N = len(y_test)
    K = 40

    # Checkpoint path to allow resuming
    ckpt_path = "data/q3_checkpoint.pkl"
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f_ckpt:
            ckpt = pickle.load(f_ckpt)
        start_r = ckpt["completed_r"]
        p_tree = ckpt["p_tree"]
        p_bag = ckpt["p_bag"]
        p_boost = ckpt["p_boost"]
        p_bag_100 = ckpt["p_bag_100"]
        accs_tree = ckpt["accs_tree"]
        accs_bag = ckpt["accs_bag"]
        accs_boost = ckpt["accs_boost"]
        accs_bag_100 = ckpt["accs_bag_100"]
        log(f"[Resuming from checkpoint: {start_r}/{R} refits already complete]")
    else:
        start_r = 0
        p_tree = np.zeros((R, N, K), dtype=float)
        p_bag = np.zeros((R, N, K), dtype=float)
        p_boost = np.zeros((R, N, K), dtype=float)
        p_bag_100 = np.zeros((R, N, K), dtype=float)
        accs_tree = []
        accs_bag = []
        accs_boost = []
        accs_bag_100 = []

    
    # Q 3(a): 
    
    log("\n--- Part 3(a): Fitting Selected Models across 30 Replicates ---")
    log(f"{'Replicate':<12} {'Tree Acc':<12} {'Bagging Acc':<14} {'Boosting Acc':<14} {'Elapsed(s)':<12}")
    log("-" * 65)

    for r in range(start_r, R):
        t0 = time.time()
        X_r, y_r = resampled_sets[r]

        # Single Tree
        tree = DecisionTreeClassifier(**best_tree_params, random_state=seed + r * 10)
        tree.fit(X_r, y_r)
        proba_t = tree.predict_proba(X_test)
        if proba_t.shape[1] < K:
            full_p = np.zeros((N, K))
            full_p[:, tree.classes_] = proba_t
            proba_t = full_p
        p_tree[r] = proba_t
        acc_t = float(np.mean(np.argmax(proba_t, axis=1) == y_test))
        accs_tree.append(acc_t)

        #  Bagging: Fit 100 trees
        bag = CustomBaggingClassifier(
            base_estimator_params=best_tree_params,
            n_estimators=100,
            random_state=seed + r * 100
        )
        bag.fit(X_r, y_r)
        
        # Sliced final_B model
        sub_bag = bag.get_sub_ensemble(final_B)
        proba_b = sub_bag.predict_proba(X_test)
        p_bag[r] = proba_b
        acc_b = float(np.mean(np.argmax(proba_b, axis=1) == y_test))
        accs_bag.append(acc_b)

        # 100-tree model
        proba_b100 = bag.predict_proba(X_test)
        p_bag_100[r] = proba_b100
        accs_bag_100.append(float(np.mean(np.argmax(proba_b100, axis=1) == y_test)))

        #  Boosting (depth 4, rounds 100)
        boost = CustomSAMMEAdaBoost(
            base_depth=best_boost_depth,
            n_rounds=best_boost_rounds,
            random_state=seed + r * 10
        )
        boost.fit(X_r, y_r)
        proba_boost = boost.predict_proba(X_test)
        p_boost[r] = proba_boost
        acc_boost = float(np.mean(np.argmax(proba_boost, axis=1) == y_test))
        accs_boost.append(acc_boost)

        elapsed = time.time() - t0
        log(f"Rep {r+1:<8} {acc_t:<12.4f} {acc_b:<14.4f} {acc_boost:<14.4f} {elapsed:<12.2f}")

        # Save checkpoint after each replicate
        with open(ckpt_path, "wb") as f_ckpt:
            pickle.dump({
                "completed_r": r + 1,
                "p_tree": p_tree,
                "p_bag": p_bag,
                "p_boost": p_boost,
                "p_bag_100": p_bag_100,
                "accs_tree": accs_tree,
                "accs_bag": accs_bag,
                "accs_boost": accs_boost,
                "accs_bag_100": accs_bag_100
            }, f_ckpt)

    # Compute Part 3(a) Bias-Variance
    bv_tree = compute_bias_variance(p_tree, y_test, n_classes=K)
    bv_bag = compute_bias_variance(p_bag, y_test, n_classes=K)
    bv_boost = compute_bias_variance(p_boost, y_test, n_classes=K)

    log("\n" + "=" * 80)
    log(f"{'Model':<22} {'Bias^2':<12} {'Variance':<12} {'Total Loss':<12} {'Mean Test Acc':<16}")
    log("-" * 80)
    log(f"{'Selected Single Tree':<22} {bv_tree['bias_squared']:<12.4f} {bv_tree['variance']:<12.4f} {bv_tree['total_expected_error']:<12.4f} {np.mean(accs_tree):.4f} +/- {np.std(accs_tree):.4f}")
    log(f"{f'Bagging (B={final_B})':<22} {bv_bag['bias_squared']:<12.4f} {bv_bag['variance']:<12.4f} {bv_bag['total_expected_error']:<12.4f} {np.mean(accs_bag):.4f} +/- {np.std(accs_bag):.4f}")
    log(f"{f'Boosting (d={best_boost_depth},r={best_boost_rounds})':<22} {bv_boost['bias_squared']:<12.4f} {bv_boost['variance']:<12.4f} {bv_boost['total_expected_error']:<12.4f} {np.mean(accs_boost):.4f} +/- {np.std(accs_boost):.4f}")
    log("=" * 80 + "\n")

   
    # Q 3(b): 

    log("--- Part 3(b): Depth Variation Study (Single Tree vs 100-Tree Bagging) ---")
    depth_configs = [
        {"name": "Shallow (Depth 4)", "params": {"max_depth": 4, "min_samples_leaf": 1}},
        {"name": "Selected (None, leaf=2)", "params": best_tree_params},
        {"name": "Deep (Depth 16)", "params": {"max_depth": 16, "min_samples_leaf": 1}},
    ]

    depth_study_results = []
    tree_biases, tree_vars = [], []
    bag_biases, bag_vars = [], []
    depth_labels = [c["name"] for c in depth_configs]

    for d_idx, cfg in enumerate(depth_configs):
        c_name = cfg["name"]
        c_params = cfg["params"]
        log(f"Evaluating {c_name}: {c_params}...")

        if d_idx == 1:
            bv_t = bv_tree
            bv_b = compute_bias_variance(p_bag_100, y_test, n_classes=K)
            mean_acc_t = float(np.mean(accs_tree))
            mean_acc_b = float(np.mean(accs_bag_100))
        else:
            p_t_d = np.zeros((R, N, K), dtype=float)
            accs_t_d = []
            p_b_d = np.zeros((R, N, K), dtype=float)
            accs_b_d = []

            for r in range(R):
                X_r, y_r = resampled_sets[r]
                # Single tree
                t = DecisionTreeClassifier(**c_params, random_state=seed + r * 10)
                t.fit(X_r, y_r)
                proba_t = t.predict_proba(X_test)
                if proba_t.shape[1] < K:
                    full_p = np.zeros((N, K))
                    full_p[:, t.classes_] = proba_t
                    proba_t = full_p
                p_t_d[r] = proba_t
                accs_t_d.append(float(np.mean(np.argmax(proba_t, axis=1) == y_test)))

                # 100-tree bagging
                b_m = CustomBaggingClassifier(
                    base_estimator_params=c_params,
                    n_estimators=100,
                    random_state=seed + r * 100
                )
                b_m.fit(X_r, y_r)
                proba_b = b_m.predict_proba(X_test)
                p_b_d[r] = proba_b
                accs_b_d.append(float(np.mean(np.argmax(proba_b, axis=1) == y_test)))

            bv_t = compute_bias_variance(p_t_d, y_test, n_classes=K)
            bv_b = compute_bias_variance(p_b_d, y_test, n_classes=K)
            mean_acc_t = float(np.mean(accs_t_d))
            mean_acc_b = float(np.mean(accs_b_d))

        var_removed = bv_t["variance"] - bv_b["variance"]
        pct_var_removed = (var_removed / bv_t["variance"]) * 100.0 if bv_t["variance"] > 0 else 0.0
        acc_improvement = mean_acc_b - mean_acc_t

        log(f"   Single Tree: Bias^2={bv_t['bias_squared']:.4f}, Var={bv_t['variance']:.4f}, Acc={mean_acc_t*100:.2f}%")
        log(f"   100-Bagging: Bias^2={bv_b['bias_squared']:.4f}, Var={bv_b['variance']:.4f}, Acc={mean_acc_b*100:.2f}%")
        log(f"   Variance Removed: {var_removed:.4f} ({pct_var_removed:.1f}%), Acc Gain: +{acc_improvement*100:.2f}%\n")

        tree_biases.append(bv_t["bias_squared"])
        tree_vars.append(bv_t["variance"])
        bag_biases.append(bv_b["bias_squared"])
        bag_vars.append(bv_b["variance"])

        depth_study_results.append({
            "name": c_name,
            "params": c_params,
            "tree_bias": bv_t["bias_squared"],
            "tree_var": bv_t["variance"],
            "tree_acc": mean_acc_t,
            "bag_bias": bv_b["bias_squared"],
            "bag_var": bv_b["variance"],
            "bag_acc": mean_acc_b,
            "var_removed": var_removed,
            "pct_var_removed": pct_var_removed,
            "acc_improvement": acc_improvement
        })

    # Plot depth variation and stacked decomposition
    plot_q3_bias_variance_depth(
        depth_labels=depth_labels,
        tree_bias=tree_biases,
        tree_var=tree_vars,
        bag_bias=bag_biases,
        bag_var=bag_vars,
        save_path="figures/q3_bias_variance_depth.png"
    )

    plot_q3_decomposition_bars(
        models=["Single Tree", f"Bagging (B={final_B})", "Boosting (SAMME)"],
        biases=[bv_tree["bias_squared"], bv_bag["bias_squared"], bv_boost["bias_squared"]],
        variances=[bv_tree["variance"], bv_bag["variance"], bv_boost["variance"]],
        save_path="figures/q3_decomposition_bars.png"
    )

    
    # Part 3(d): Master Comparison Table
    
    log("--- Part 3(d): Master Synthesis Table ---")
    clean_acc_tree = q1_summary["robustness"]["tree_clean_acc"]
    mask_acc_tree = q1_summary["robustness"]["tree_mask_acc"]
    clean_acc_bag = q1_summary["robustness"]["bag_clean_acc"]
    mask_acc_bag = q1_summary["robustness"]["bag_mask_acc"]
    clean_acc_boost = q2_summary["robustness"]["boost_clean_acc"]
    mask_acc_boost = q2_summary["robustness"]["boost_mask_acc"]

    log(f"{'Model':<25} {'Clean Test Acc':<16} {'Masked Test Acc':<17} {'Bias^2':<12} {'Variance':<12}")
    log("-" * 82)
    log(f"{'Selected Single Tree':<25} {clean_acc_tree:<16.4f} {mask_acc_tree:<17.4f} {bv_tree['bias_squared']:<12.4f} {bv_tree['variance']:<12.4f}")
    log(f"{f'Bagging (B={final_B})':<25} {clean_acc_bag:<16.4f} {mask_acc_bag:<17.4f} {bv_bag['bias_squared']:<12.4f} {bv_bag['variance']:<12.4f}")
    log(f"{f'Boosting (d={best_boost_depth},r={best_boost_rounds})':<25} {clean_acc_boost:<16.4f} {mask_acc_boost:<17.4f} {bv_boost['bias_squared']:<12.4f} {bv_boost['variance']:<12.4f}")
    log("-" * 82 + "\n")

    summary_q3 = {
        "part_a": {
            "single_tree": {
                "bias_squared": bv_tree["bias_squared"],
                "variance": bv_tree["variance"],
                "total_loss": bv_tree["total_expected_error"],
                "acc_mean": float(np.mean(accs_tree)),
                "acc_std": float(np.std(accs_tree))
            },
            "bagging": {
                "bias_squared": bv_bag["bias_squared"],
                "variance": bv_bag["variance"],
                "total_loss": bv_bag["total_expected_error"],
                "acc_mean": float(np.mean(accs_bag)),
                "acc_std": float(np.std(accs_bag))
            },
            "boosting": {
                "bias_squared": bv_boost["bias_squared"],
                "variance": bv_boost["variance"],
                "total_loss": bv_boost["total_expected_error"],
                "acc_mean": float(np.mean(accs_boost)),
                "acc_std": float(np.std(accs_boost))
            }
        },
        "part_b": depth_study_results,
        "part_d_table": [
            {
                "model": "Selected Single Tree",
                "clean_acc": clean_acc_tree,
                "masked_acc": mask_acc_tree,
                "bias_squared": bv_tree["bias_squared"],
                "variance": bv_tree["variance"]
            },
            {
                "model": f"Bagging (B={final_B})",
                "clean_acc": clean_acc_bag,
                "masked_acc": mask_acc_bag,
                "bias_squared": bv_bag["bias_squared"],
                "variance": bv_bag["variance"]
            },
            {
                "model": f"Boosting (d={best_boost_depth},r={best_boost_rounds})",
                "clean_acc": clean_acc_boost,
                "masked_acc": mask_acc_boost,
                "bias_squared": bv_boost["bias_squared"],
                "variance": bv_boost["variance"]
            }
        ]
    }

    with open("data/q3_summary.json", "w") as f:
        json.dump(summary_q3, f, indent=2, cls=NumpyEncoder)

    log("[Success] Q3 experiments complete. Summary saved to data/q3_summary.json")
    log_file.close()


if __name__ == "__main__":
    run_q3()
