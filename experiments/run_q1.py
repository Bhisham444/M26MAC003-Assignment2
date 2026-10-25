

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.tree import DecisionTreeClassifier

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import load_and_prepare_data
from src.bagging import CustomBaggingClassifier
from src.utils import setup_logger, plot_q1_oob_disagreement_margin, plot_q1_error_faces


def run_q1():
    log, log_file = setup_logger("logs/q1_baseline_bagging.log")
    log("=" * 80)
    log("QUESTION 1: DECISION TREE BASELINE AND BAGGING FROM SCRATCH")
    log("Student: Bhisham Sharma | Roll Number: M26MAC003")
    log("=" * 80)

    # 1. Load data with  seed M26MAC003
    data = load_and_prepare_data(roll_number="M26MAC003")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    X_test_masked = data["X_test_masked"]
    seed = data["seed"]

    log(f"Dataset split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    log(f"Features: {X_train.shape[1]} raw pixels. Deterministic seed: {seed}\n")

    
    # Q. 1(a): 
   
    log("--- Part 1(a): Single Decision Tree Baseline Grid Search ---")
    max_depths = [2, 4, 8, 16, None]
    min_samples_leafs = [1, 2, 4]

    best_val_acc = -1.0
    best_params = None
    best_tree = None
    grid_results = []

    log(f"{'Depth':<8} {'MinLeaf':<8} {'TrainAcc':<10} {'ValAcc':<10} {'TestAcc':<10} {'#Leaves':<10} {'MeanSamp/Leaf':<14} {'Frac1Samp':<10}")
    log("-" * 82)

    for md in max_depths:
        for msl in min_samples_leafs:
            tree = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=seed)
            tree.fit(X_train, y_train)

            train_acc = float(tree.score(X_train, y_train))
            val_acc = float(tree.score(X_val, y_val))
            test_acc = float(tree.score(X_test, y_test))

            # Leaf statistics
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            leaf_mask = (children_left == -1) & (children_right == -1)
            
            n_leaves = int(np.sum(leaf_mask))
            samples_at_leaves = tree.tree_.n_node_samples[leaf_mask]
            mean_samples_per_leaf = float(np.mean(samples_at_leaves))
            frac_single_sample = float(np.mean(samples_at_leaves == 1))

            md_str = str(md) if md is not None else "None"
            log(f"{md_str:<8} {msl:<8} {train_acc:<10.4f} {val_acc:<10.4f} {test_acc:<10.4f} {n_leaves:<10} {mean_samples_per_leaf:<14.2f} {frac_single_sample:<10.4f}")

            res = {
                "max_depth": md,
                "min_samples_leaf": msl,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "n_leaves": n_leaves,
                "mean_samples_per_leaf": mean_samples_per_leaf,
                "frac_single_sample": frac_single_sample
            }
            grid_results.append(res)

            # Select model  on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {"max_depth": md, "min_samples_leaf": msl}
                best_tree = tree

    log("-" * 82)
    log(f"Selected Frozen Setting (Validation-driven): max_depth={best_params['max_depth']}, min_samples_leaf={best_params['min_samples_leaf']}")
    log(f"Frozen Single Tree Metrics: Train Acc={best_tree.score(X_train, y_train):.4f}, Val Acc={best_val_acc:.4f}, Test Acc={best_tree.score(X_test, y_test):.4f}\n")

    
    # Q. 1(b) & 1(c): 
    log("--- Part 1(b) & 1(c): Custom Bagging Classifier Evaluation ---")
    B_list = [1, 5, 10, 25, 50, 100, 200]
    N = X_train.shape[0]
    theoretical_distinct = N * (1.0 - (1.0 - 1.0 / N) ** N) # ~ 151.90 for N=240
    log(f"Theoretical expected distinct samples per bootstrap: {theoretical_distinct:.2f} ({theoretical_distinct / N * 100:.2f}% of {N})\n")

    bagging_results = []
    oob_accs = []
    disagreements = []
    margins = []
    val_accs = []

    log(f"{'B':<6} {'TrainAcc':<10} {'ValAcc':<10} {'TestAcc':<10} {'AvgDistinct':<14} {'OOBAcc':<10} {'Disagreement':<14} {'VoteMargin':<12}")
    log("-" * 88)

    checkpoint_path = "data/q1_checkpoint.pkl"
    trained_bagging_models = {}

    log(f"Training master Bagging ensemble of B_max={max(B_list)} trees...")
    master_bag = CustomBaggingClassifier(
        base_estimator_params=best_params,
        n_estimators=max(B_list),
        random_state=seed + 42
    )
    master_bag.fit(X_train, y_train)

    for B in B_list:
        bag = master_bag.get_sub_ensemble(B)
        trained_bagging_models[B] = bag

        train_acc = float(bag.score(X_train, y_train))
        val_acc = float(bag.score(X_val, y_val))
        test_acc = float(bag.score(X_test, y_test))

        avg_distinct = bag.average_distinct_samples_per_bag()
        oob_acc, oob_evaluated = bag.compute_oob_accuracy(X_train, y_train)
        disagreement = bag.compute_pairwise_disagreement(X_val)
        _, mean_margin = bag.compute_vote_margin(X_val)

        oob_accs.append(oob_acc)
        disagreements.append(disagreement)
        margins.append(mean_margin)
        val_accs.append(val_acc)

        log(f"{B:<6} {train_acc:<10.4f} {val_acc:<10.4f} {test_acc:<10.4f} {avg_distinct:<14.2f} {oob_acc:<10.4f} {disagreement:<14.4f} {mean_margin:<12.4f}")

        bagging_results.append({
            "B": B,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "avg_distinct": avg_distinct,
            "oob_acc": oob_acc,
            "disagreement": disagreement,
            "mean_margin": mean_margin
        })

    # Save checkpoint
    with open(checkpoint_path, "wb") as f_ckpt:
        pickle.dump({
            "best_tree": best_tree,
            "best_params": best_params,
            "grid_results": grid_results,
            "trained_bagging_models": trained_bagging_models,
            "bagging_results": bagging_results,
            "oob_accs": oob_accs,
            "disagreements": disagreements,
            "margins": margins,
            "val_accs": val_accs
        }, f_ckpt)

    log("-" * 88)
    
    # Plot OOB, Disagreement, and Vote Margin
    plot_q1_oob_disagreement_margin(
        B_list=B_list,
        oob_accs=oob_accs,
        disagreements=disagreements,
        margins=margins,
        val_accs=val_accs,
        save_path="figures/q1_oob_disagreement_margin.png"
    )

    
    best_B_idx = int(np.argmax(val_accs))
    final_B = B_list[best_B_idx]
    final_bagged_model = trained_bagging_models[final_B]
    log(f"Selected Final Ensemble Size (Validation-driven): B = {final_B} (Val Acc: {val_accs[best_B_idx]:.4f})\n")

    
    # Q.1(d):
   
    log("--- Part 1(d): Clean vs. Masked Robustness & Error Analysis ---")
    
    # Clean Test
    tree_clean_acc = float(best_tree.score(X_test, y_test))
    tree_clean_conf = float(np.mean(np.max(best_tree.predict_proba(X_test), axis=1)))
    
    bag_clean_acc = float(final_bagged_model.score(X_test, y_test))
    bag_clean_conf = float(np.mean(np.max(final_bagged_model.predict_proba(X_test), axis=1)))

    # Masked Test
    tree_mask_acc = float(best_tree.score(X_test_masked, y_test))
    tree_mask_conf = float(np.mean(np.max(best_tree.predict_proba(X_test_masked), axis=1)))

    bag_mask_acc = float(final_bagged_model.score(X_test_masked, y_test))
    bag_mask_conf = float(np.mean(np.max(final_bagged_model.predict_proba(X_test_masked), axis=1)))

    log(f"{'Model':<20} {'Clean Acc':<12} {'Clean Conf':<12} {'Masked Acc':<12} {'Masked Conf':<12}")
    log("-" * 70)
    log(f"{'Single Tree':<20} {tree_clean_acc:<12.4f} {tree_clean_conf:<12.4f} {tree_mask_acc:<12.4f} {tree_mask_conf:<12.4f}")
    log(f"{f'Bagging (B={final_B})':<20} {bag_clean_acc:<12.4f} {bag_clean_conf:<12.4f} {bag_mask_acc:<12.4f} {bag_mask_conf:<12.4f}")
    log("-" * 70)

    #  face case study on clean test set
    tree_preds = best_tree.predict(X_test)
    bag_preds = final_bagged_model.predict(X_test)
    bag_votes = final_bagged_model.predict_vote_counts(X_test)

    corrected_cases = [] # tree wrong, bagging correct
    both_wrong_cases = [] # both wrong

    for i in range(len(y_test)):
        is_tree_right = (tree_preds[i] == y_test[i])
        is_bag_right = (bag_preds[i] == y_test[i])
        case_info = {
            "test_idx": i,
            "y_true": int(y_test[i]),
            "tree_pred": int(tree_preds[i]),
            "bag_pred": int(bag_preds[i]),
            "votes": bag_votes[i],
            "image": X_test[i]
        }
        if (not is_tree_right) and is_bag_right and len(corrected_cases) < 2:
            corrected_cases.append(case_info)
        elif (not is_tree_right) and (not is_bag_right) and len(both_wrong_cases) < 2:
            both_wrong_cases.append(case_info)

    log("\nDiagnostic Cases Identified:")
    for idx, c in enumerate(corrected_cases):
        log(f"Corrected Case {idx+1}: Test Sample #{c['test_idx']} | True: {c['y_true']} | Tree: {c['tree_pred']} | Bag: {c['bag_pred']} (Votes: {c['votes'][c['bag_pred']]}/{final_B})")
    for idx, c in enumerate(both_wrong_cases):
        log(f"Both Wrong Case {idx+1}: Test Sample #{c['test_idx']} | True: {c['y_true']} | Tree: {c['tree_pred']} | Bag: {c['bag_pred']} (Votes: {c['votes'][c['bag_pred']]}/{final_B})")

    # Plot error faces
    plot_q1_error_faces(corrected_cases, both_wrong_cases, save_path="figures/q1_error_faces.png")

    # Save summary dictionary for CVPR report
    summary = {
        "best_tree_params": best_params,
        "grid_results": grid_results,
        "bagging_results": bagging_results,
        "final_B": final_B,
        "robustness": {
            "tree_clean_acc": tree_clean_acc,
            "tree_clean_conf": tree_clean_conf,
            "tree_mask_acc": tree_mask_acc,
            "tree_mask_conf": tree_mask_conf,
            "bag_clean_acc": bag_clean_acc,
            "bag_clean_conf": bag_clean_conf,
            "bag_mask_acc": bag_mask_acc,
            "bag_mask_conf": bag_mask_conf
        },
        "theoretical_distinct": theoretical_distinct
    }

    with open("data/q1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("\n[Success] Q1 experiments complete. Summary saved to data/q1_summary.json")
    log_file.close()


if __name__ == "__main__":
    run_q1()
