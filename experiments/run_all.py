

import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_q1 import run_q1
from experiments.run_q2 import run_q2
from experiments.run_q3 import run_q3


def main():
    start_total = time.time()
    print("=" * 80)
    print("MACHINE LEARNING ASSIGNMENT 2")
    print("Student: Bhisham Sharma | Roll Number: M26MAC003")
    print("Deterministic Seed: 26003 ")
    print("=" * 80)

    # Question 1
    print("\n>>> EXECUTING QUESTION 1: DECISION TREES & BAGGING FROM SCRATCH")
    t0 = time.time()
    run_q1()
    print(f">>> Question 1 completed in {time.time() - t0:.2f} seconds.\n")

    # Question 2
    print("\n>>> EXECUTING QUESTION 2: MULTICLASS BOOSTING (SAMME) FROM SCRATCH")
    t0 = time.time()
    run_q2()
    print(f">>> Question 2 completed in {time.time() - t0:.2f} seconds.\n")

    # Question 3
    print("\n>>> EXECUTING QUESTION 3: BIAS-VARIANCE DECOMPOSITION (30 REFITS)")
    t0 = time.time()
    run_q3()
    print(f">>> Question 3 completed in {time.time() - t0:.2f} seconds.\n")

    print("=" * 80)
    print(f"PIPELINE FULLY EXECUTED IN {time.time() - start_total:.2f} SECONDS.")
    print("Generated Artifacts:")
    print("  - Data & Summaries: data/split_indices.npz, data/q1_summary.json, data/q2_summary.json, data/q3_summary.json")
    print("  - Figures: figures/ (7 publication-ready PNG plots)")
    print("  - Execution Logs: logs/q1_baseline_bagging.log, logs/q2_samme_boosting.log, logs/q3_bias_variance.log")
    print("=" * 80)


if __name__ == "__main__":
    main()
