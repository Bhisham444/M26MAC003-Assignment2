

import os
import sys
import json
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

from src.dataset import load_and_prepare_data
from src.bagging import CustomBaggingClassifier
from src.boosting import CustomSAMMEAdaBoost
from src.utils import (
    setup_logger,
    plot_q2_diagnostics,
    plot_q2_heavy_sample,
    plot_q2_masked_comparison_face
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


def run_q2():
    log, log_file = setup_logger("logs/q2_samme_boosting.log")
    log("=" * 80)
    log("QUESTION 2: MULTICLASS BOOSTING FROM SCRATCH (SAMME ADABOOST)")
    log("Student: Bhisham Sharma | Roll Number: M26MAC003")
    log("=" * 80)

     #Load data
    data = load_and_prepare_data(roll_number="M26MAC003")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    X_test_masked = data["X_test_masked"]
    seed = data["seed"]

    # Load frozen Q1 settings
    with open("data/q1_summary.json", "r") as f:
        q1_summary = json.load(f)
    best_tree_params = q1_summary["best_tree_params"]
    final_B = q1_summary["final_B"]

    with open("data/q1_checkpoint.pkl", "rb") as f:
        q1_ckpt = pickle.load(f)
    frozen_tree = q1_ckpt["best_tree"]
    frozen_bagging = q1_ckpt["trained_bagging_models"][final_B]

   
    # Q 2(a): 3-Class Toy Example 
   
    log("\n--- Part 2(a): 3-Class Synthetic Example (Round 1 Step-by-Step Logging) ---")
    
    toy_rng = np.random.default_rng(seed)
    X_toy = np.array([
        [1.0, 2.0], [1.5, 1.8], [2.0, 2.5],  # Class 0
        [5.0, 5.0], [5.5, 4.5], [6.0, 5.5],  # Class 1
        [1.0, 6.0], [2.0, 7.0], [1.5, 6.5],  # Class 2
    ])
    y_toy = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    N_toy = len(y_toy)
    K_toy = 3

    log(f"Toy Dataset: N={N_toy} samples, K={K_toy} classes.")
    
    
    # 1. Initial weights:  = 1/N
    w_initial = np.ones(N_toy) / float(N_toy)
    log(f"1. Initial Sample Weights w_1:\n   {np.round(w_initial, 4).tolist()}")

    # 2. Fit weak learner 
    stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
    stump.fit(X_toy, y_toy, sample_weight=w_initial)
    toy_preds = stump.predict(X_toy)
    log(f"2. Weak Learner Predictions h_1(x):\n   {toy_preds.tolist()}")
    log(f"   Ground Truth Labels y:\n   {y_toy.tolist()}")

    misclassified = (toy_preds != y_toy).astype(float)
    log(f"   Misclassified Indicators I(y != h_1):\n   {misclassified.astype(int).tolist()}")

    # 3. Weighted error epsilon_1
    weighted_err = np.sum(w_initial * misclassified) / np.sum(w_initial)
    log(f"3. Weighted Training Error epsilon_1 = {weighted_err:.4f}")
    log(f"   Random multiclass error bound (1 - 1/K) = {1 - 1/K_toy:.4f}")

    # 4. Compute alpha_1
    alpha_1 = np.log((1.0 - weighted_err) / weighted_err) + np.log(K_toy - 1)
    log(f"4. Learner Weight alpha_1 = ln(({1-weighted_err:.4f})/{weighted_err:.4f}) + ln({K_toy-1}) = {alpha_1:.4f}")

    # 5. Update weights:
    w_unnorm = w_initial * np.exp(alpha_1 * misclassified)
    w_updated = w_unnorm / np.sum(w_unnorm)
    log(f"5. Updated Normalized Sample Weights w_2:\n   {np.round(w_updated, 4).tolist()}")
    log(f"   Sum of updated weights: {np.sum(w_updated):.6f}")

    # Verify CustomSAMMEAdaBoost 
    toy_booster = CustomSAMMEAdaBoost(base_depth=1, n_rounds=1, random_state=seed)
    toy_booster.fit(X_toy, y_toy)
    assert np.isclose(toy_booster.estimator_weights_[0], alpha_1), "Alpha mismatch!"
    log("   [Verified] CustomSAMMEAdaBoost class update exactly matches manual computation!\n")

   
    # Part 2(b):
   
    log("--- Part 2(b): SAMME Grid Search on Olivetti Faces ---")
    depths = [1, 2, 3, 4]
    rounds_list = [10, 25, 50, 100]

    best_val_acc = -1.0
    best_depth = None
    best_rounds = None
    best_booster = None
    boosting_grid = []

    log(f"{'Depth':<8} {'Rounds':<8} {'TrainAcc':<10} {'ValAcc':<10} {'TestAcc':<10} {'ActualRounds':<14}")
    log("-" * 65)

    for d in depths:
        for r in rounds_list:
            booster = CustomSAMMEAdaBoost(base_depth=d, n_rounds=r, random_state=seed)
            booster.fit(X_train, y_train)

            train_acc = booster.score(X_train, y_train)
            val_acc = booster.score(X_val, y_val)
            test_acc = booster.score(X_test, y_test)
            actual_r = len(booster.estimators_)

            log(f"{d:<8} {r:<8} {train_acc:<10.4f} {val_acc:<10.4f} {test_acc:<10.4f} {actual_r:<14}")
            boosting_grid.append({
                "depth": d,
                "rounds": r,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "actual_rounds": actual_r
            })

         
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_depth = d
                best_rounds = r
                best_booster = booster

    log("-" * 65)
    log(f"Selected Boosting Model (Validation-driven): Depth={best_depth}, Rounds={best_rounds}")
    log(f"Selected Booster Metrics: Train Acc={best_booster.score(X_train, y_train):.4f}, Val Acc={best_val_acc:.4f}, Test Acc={best_booster.score(X_test, y_test):.4f}\n")

    # Plot 4 diagnostic curves for selected booster
    plot_rounds = list(range(1, len(best_booster.estimator_errors_) + 1))
    plot_q2_diagnostics(
        rounds=plot_rounds,
        weighted_errors=best_booster.estimator_errors_,
        learner_weights=best_booster.estimator_weights_,
        max_weights=best_booster.max_sample_weight_history_[:len(plot_rounds)],
        top10_weights=best_booster.top10_weight_history_[:len(plot_rounds)],
        save_path="figures/q2_diagnostics.png"
    )

    
    # Q 2(c): 
   
    log("--- Part 2(c): Head-to-Head Comparison: Bagging vs. Boosting ---")
    comp_rounds = [25, 50, 100]
    comparison_table = []

    log(f"{'Learners':<10} {'Model':<12} {'TrainAcc':<10} {'ValAcc':<10} {'TestAcc':<10} {'TrainTime(s)':<14}")
    log("-" * 70)

    # Benchmark single tree training time for accurate sequential bagging time
    t_tree_start = time.time()
    DecisionTreeClassifier(**best_tree_params, random_state=seed).fit(X_train, y_train)
    single_tree_time = time.time() - t_tree_start

    for n_learners in comp_rounds:
        # Bagging (reuse models  in Q1)
        bag_m = q1_ckpt["trained_bagging_models"].get(n_learners)
        if bag_m is None:
            t0 = time.time()
            bag_m = CustomBaggingClassifier(
                base_estimator_params=best_tree_params,
                n_estimators=n_learners,
                random_state=seed + n_learners
            ).fit(X_train, y_train)
            t_bag = time.time() - t0
        else:
            t_bag = single_tree_time * n_learners
            
        bag_train = bag_m.score(X_train, y_train)
        bag_val = bag_m.score(X_val, y_val)
        bag_test = bag_m.score(X_test, y_test)

        log(f"{n_learners:<10} {'Bagging':<12} {bag_train:<10.4f} {bag_val:<10.4f} {bag_test:<10.4f} {t_bag:<14.2f}")

        # Boosting
        t0 = time.time()
        boost_m = CustomSAMMEAdaBoost(
            base_depth=best_depth,
            n_rounds=n_learners,
            random_state=seed + n_learners
        ).fit(X_train, y_train)
        t_boost = time.time() - t0
        boost_train = boost_m.score(X_train, y_train)
        boost_val = boost_m.score(X_val, y_val)
        boost_test = boost_m.score(X_test, y_test)

        log(f"{n_learners:<10} {'Boosting':<12} {boost_train:<10.4f} {boost_val:<10.4f} {boost_test:<10.4f} {t_boost:<14.2f}")

        comparison_table.append({
            "learners": n_learners,
            "bag_train": bag_train, "bag_val": bag_val, "bag_test": bag_test, "bag_time": t_bag,
            "boost_train": boost_train, "boost_val": boost_val, "boost_test": boost_test, "boost_time": t_boost
        })

    log("-" * 70)

    # Inspect one training face that receives very large boosting weight
    final_weights = best_booster.sample_weights_history_[-1]
    heaviest_idx = int(np.argmax(final_weights))
    heaviest_weight = float(final_weights[heaviest_idx])
    heaviest_class = int(y_train[heaviest_idx])
    
    log(f"\nHeavyweight Sample Analysis:")
    log(f"Training Sample #{heaviest_idx} (Class {heaviest_class}) has highest final weight: {heaviest_weight:.4f} (Uniform baseline was {1/len(y_train):.4f})")
    
    heavy_history = best_booster.get_sample_prediction_history(heaviest_idx, heaviest_class)
    misclassified_count = sum(heavy_history["is_misclassified"])
    log(f"Misclassified in {misclassified_count} out of {len(heavy_history['is_misclassified'])} boosting rounds ({misclassified_count/len(heavy_history['is_misclassified'])*100:.1f}%).")

    plot_q2_heavy_sample(
        face_image=X_train[heaviest_idx],
        sample_idx=heaviest_idx,
        true_label=heaviest_class,
        weight_history=heavy_history["weights"],
        is_misclassified_history=heavy_history["is_misclassified"],
        save_path="figures/q2_heavy_sample_history.png"
    )

    
    # Q. 2(d): 
    log("\n--- Part 2(d): Robustness on Clean and Masked Test Sets ---")
    
    # 1. Single tree
    t_clean_acc = float(frozen_tree.score(X_test, y_test))
    t_clean_conf = float(np.mean(np.max(frozen_tree.predict_proba(X_test), axis=1)))
    t_mask_acc = float(frozen_tree.score(X_test_masked, y_test))
    t_mask_conf = float(np.mean(np.max(frozen_tree.predict_proba(X_test_masked), axis=1)))

    # 2. Bagging
    bag_clean_acc = float(frozen_bagging.score(X_test, y_test))
    bag_clean_conf = float(np.mean(np.max(frozen_bagging.predict_proba(X_test), axis=1)))
    bag_mask_acc = float(frozen_bagging.score(X_test_masked, y_test))
    bag_mask_conf = float(np.mean(np.max(frozen_bagging.predict_proba(X_test_masked), axis=1)))

    # 3. Boosting
    boost_clean_acc = float(best_booster.score(X_test, y_test))
    boost_clean_conf = float(np.mean(np.max(best_booster.predict_proba(X_test), axis=1)))
    boost_mask_acc = float(best_booster.score(X_test_masked, y_test))
    boost_mask_conf = float(np.mean(np.max(best_booster.predict_proba(X_test_masked), axis=1)))

    log(f"{'Model':<18} {'Clean Acc':<12} {'Clean Conf':<12} {'Masked Acc':<12} {'Masked Conf':<12}")
    log("-" * 68)
    log(f"{'Single Tree':<18} {t_clean_acc:<12.4f} {t_clean_conf:<12.4f} {t_mask_acc:<12.4f} {t_mask_conf:<12.4f}")
    log(f"{f'Bagging (B={final_B})':<18} {bag_clean_acc:<12.4f} {bag_clean_conf:<12.4f} {bag_mask_acc:<12.4f} {bag_mask_conf:<12.4f}")
    log(f"{f'Boosting (d={best_depth},r={best_rounds})':<18} {boost_clean_acc:<12.4f} {boost_clean_conf:<12.4f} {boost_mask_acc:<12.4f} {boost_mask_conf:<12.4f}")
    log("-" * 68)

    # Find one face where bagging and boosting respond differently to the missing region
    
    bag_clean_preds = frozen_bagging.predict(X_test)
    bag_mask_preds = frozen_bagging.predict(X_test_masked)
    boost_clean_preds = best_booster.predict(X_test)
    boost_mask_preds = best_booster.predict(X_test_masked)

    diff_idx = None
    for i in range(len(y_test)):
        # case where bagging survives mask or behaves differently
        bag_correct_mask = (bag_mask_preds[i] == y_test[i])
        boost_correct_mask = (boost_mask_preds[i] == y_test[i])
        if bag_correct_mask != boost_correct_mask:
            diff_idx = i
            break

    if diff_idx is None:
        
        diff_idx = int(np.where(bag_mask_preds != boost_mask_preds)[0][0])

    log(f"Discrepant Robustness Case: Test Sample #{diff_idx} (True Class {y_test[diff_idx]})")
    log(f"   Clean:  Bagging={bag_clean_preds[diff_idx]} | Boosting={boost_clean_preds[diff_idx]}")
    log(f"   Masked: Bagging={bag_mask_preds[diff_idx]} | Boosting={boost_mask_preds[diff_idx]}")

    plot_q2_masked_comparison_face(
        clean_img=X_test[diff_idx],
        masked_img=X_test_masked[diff_idx],
        sample_idx=diff_idx,
        y_true=int(y_test[diff_idx]),
        bag_pred_clean=int(bag_clean_preds[diff_idx]),
        bag_pred_mask=int(bag_mask_preds[diff_idx]),
        boost_pred_clean=int(boost_clean_preds[diff_idx]),
        boost_pred_mask=int(boost_mask_preds[diff_idx]),
        save_path="figures/q2_masked_comparison_face.png"
    )

    # Save summary and models
    q2_summary = {
        "toy_verification": {
            "w_initial": w_initial.tolist(),
            "toy_preds": toy_preds.tolist(),
            "weighted_err": float(weighted_err),
            "alpha_1": float(alpha_1),
            "w_updated": w_updated.tolist()
        },
        "best_depth": best_depth,
        "best_rounds": best_rounds,
        "boosting_grid": boosting_grid,
        "comparison_table": comparison_table,
        "heavyweight_sample": {
            "index": heaviest_idx,
            "true_label": heaviest_class,
            "final_weight": heaviest_weight,
            "misclassified_count": misclassified_count,
            "total_rounds": len(heavy_history["is_misclassified"])
        },
        "robustness": {
            "tree_clean_acc": t_clean_acc, "tree_clean_conf": t_clean_conf,
            "tree_mask_acc": t_mask_acc, "tree_mask_conf": t_mask_conf,
            "bag_clean_acc": bag_clean_acc, "bag_clean_conf": bag_clean_conf,
            "bag_mask_acc": bag_mask_acc, "bag_mask_conf": bag_mask_conf,
            "boost_clean_acc": boost_clean_acc, "boost_clean_conf": boost_clean_conf,
            "boost_mask_acc": boost_mask_acc, "boost_mask_conf": boost_mask_conf,
            "divergent_sample": {
                "test_idx": diff_idx,
                "y_true": int(y_test[diff_idx]),
                "bag_clean": int(bag_clean_preds[diff_idx]),
                "bag_mask": int(bag_mask_preds[diff_idx]),
                "boost_clean": int(boost_clean_preds[diff_idx]),
                "boost_mask": int(boost_mask_preds[diff_idx])
            }
        }
    }

    with open("data/q2_checkpoint.pkl", "wb") as f_ckpt:
        pickle.dump({
            "best_booster": best_booster,
            "best_depth": best_depth,
            "best_rounds": best_rounds
        }, f_ckpt)

    with open("data/q2_summary.json", "w") as f:
        json.dump(q2_summary, f, indent=2, cls=NumpyEncoder)

    log("\n[Success] Q2 experiments complete. Summary saved to data/q2_summary.json")
    log_file.close()


if __name__ == "__main__":
    run_q2()
