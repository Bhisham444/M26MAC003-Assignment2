# Bagging, Boosting, and Bias-Variance on Face Identification

**Machine Learning  — Assignment 2**  
- **Name**: Bhisham Sharma  
- **Roll Number:** M26MAC003  
- **Deterministic Seed:** 26003 (roll number numerical value: `2`, `6`, `0`, `0`, `3`)  
- **Repository:** `https://github.com/Bhisham444/M26MAC003-Assignment2`  

---

## 📌 Project Overview
## 📌 Project Overview

In this project, I explore and implement ensemble learning methods—specifically **Decision Tree baselines**, **Bagging (Bootstrap Aggregation) from scratch**, and **Multi-Class SAMME AdaBoost from scratch**—on the AT&T Olivetti Faces dataset (40 identities, 400 images of size $64 \times 64$, giving 4096 raw pixels per image).

I designed the implementation to strictly follow all assignment guidelines:
- **No PCA or Dimensionality Reduction:** I train all models directly on the raw 4096-dimensional pixel vectors.
- **From-Scratch Implementations:** Following the rules, I use scikit-learn only for the weak `DecisionTreeClassifier` base learners, while I implemented the Bagging ensemble and the multi-class SAMME AdaBoost algorithms entirely from scratch in Python.
- **Identity-Stratified Partitioning:** My  roll number is **M26MAC003**, so I use the deterministic random seed `26003`. For each of the 40 people, I split their 10 images into 6 train (240 total), 2 validation (80 total), and 2 clean test images (80 total). This split is saved in `data/split_indices.npz` and reused across every single run.
- **Occlusion Robustness:** I generated an occluded test set by zeroing out the central $16 \times 16$ facial patch (rows and columns 24 to 39) to evaluate how well models cope with missing features.
- **Empirical Bias-Variance Decomposition:** I computed squared bias and variance across 30 resampled training runs using vector $L_2$ norm metrics to study the trade-offs between Bagging and Boosting on raw image data.

---

## 📂 Repository Structure

```text
ml_assignment2/
├── README.md                     # Reproduction instructions, setup, and overview
├── requirements.txt              # Pinned Python package dependencies
├── configs/
│   └── config.yaml               # Master experiment and hyperparameter configuration
├── data/
│   ├── split_indices.npz         # Saved identity-stratified train/val/test split
│   ├── q1_summary.json           # JSON summary of Q1 baseline and bagging results
│   ├── q2_summary.json           # JSON summary of Q2 SAMME AdaBoost results
│   └── q3_summary.json           # JSON summary of Q3 bias-variance decomposition
├── figures/                      # Publication-grade figures embedded in CVPR paper
│   ├── q1_oob_disagreement_margin.png  # OOB accuracy, tree disagreement, vote margin vs B
│   ├── q1_error_faces.png              # 4 diagnostic face cases (corrected vs failed)
│   ├── q2_diagnostics.png              # 4 SAMME AdaBoost training trajectories
│   ├── q2_heavy_sample_history.png     # Heaviest training face and weight trajectory
│   ├── q2_masked_comparison_face.png   # Divergent robustness face under occlusion
│   ├── q3_bias_variance_depth.png      # Bias^2 and Variance across Shallow, Selected, Deep
│   └── q3_decomposition_bars.png       # Stacked total loss decomposition bars
├── logs/
│   ├── q1_baseline_bagging.log   # Execution log for Question 1
│   ├── q2_samme_boosting.log     # Execution log for Question 2
│   └── q3_bias_variance.log      # Execution log for Question 3
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Deterministic loader, stratified split, mask generator
│   ├── bagging.py                # Custom BaggingClassifier from scratch (OOB, margins)
│   ├── boosting.py               # Custom SAMME AdaBoost from scratch with fallback logic
│   ├── metrics.py                # Bias-Variance decomposition and margin metrics
│   └── utils.py                  # Headless plotting utilities and logger
├── experiments/
│   ├── run_q1.py                 # Q1 script: Single Tree + Bagging (B in {1..200})
│   ├── run_q2.py                 # Q2 script: Toy example + SAMME grid + diagnostics
│   ├── run_q3.py                 # Q3 script: 30-repetition Bias-Variance decomposition
│   └── run_all.py                # Master reproduction script running full pipeline
└── report/
    ├── cvpr.sty                  # CVPR LaTeX style package
    ├── main.tex                  # Complete 8-page CVPR report source
    └── references.bib            # Bibliography citations
```

---

## 🚀 Environment Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Hardware: Standard 4-core CPU (or higher)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Bhisham444/M26MAC003-Assignment2.git
cd M26MAC003-Assignment2
pip install -r requirements.txt
```

`requirements.txt` contents:
```text
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
scipy>=1.10.0
pyyaml>=6.0
```

---

## 💻 Exact Reproduction Commands

You can run individual questions independently or execute the entire pipeline with a single master command:

### Run Complete End-to-End Pipeline:
```bash
python experiments/run_all.py
```

### Or Run Question-by-Question:

#### Q 1: Decision Tree Baseline & Bagging from Scratch (Q1)
```bash
python experiments/run_q1.py
```
- Performs grid search over `max_depth` $\in \{2, 4, 8, 16, \text{None}\}$ and `min_samples_leaf` $\in \{1, 2, 4\}$.
- Selects and freezes `max_depth=None, min_samples_leaf=2` on validation accuracy ($55.00\%$).
- Evaluates custom Bagging across $B \in \{1, 5, 10, 25, 50, 100, 200\}$.
- Verifies distinct bootstrap samples ($\text{theory}=151.89$ vs. $\text{empirical}\approx 152.0$).
- Generates `figures/q1_oob_disagreement_margin.png` and `figures/q1_error_faces.png`.

#### Q .2:Multiclass SAMME Boosting from Scratch (Q2)
```bash
python experiments/run_q2.py
```
- Validates SAMME round-1 mathematics on a 3-class, 9-sample synthetic toy dataset.
- Runs grid search across weak learner depths $\{1, 2, 3, 4\}$ and rounds $\{10, 25, 50, 100\}$.
- Selects depth $4$ with $100$ rounds (Val Acc: $46.25\%$).
- Generates 4 diagnostic curves (`figures/q2_diagnostics.png`).
- Audits the heaviest training face (\#204, Class 34) receiving weight $0.0713$ ($17\times$ baseline).
- Evaluates occlusion divergence on clean vs. masked test sets (`figures/q2_masked_comparison_face.png`).

#### Q.3: Bias-Variance Decomposition (30 Refits) (Q3)
```bash
python experiments/run_q3.py
```
- Generates $R = 30$ identity-stratified bootstrap training sets.
- Refits Single Tree, Bagging ($B=25$), and Boosting ($d=4, r=100$) models.
- Measures bias
- Performs depth study: Shallow (depth 4) vs. Selected (None, leaf 2) vs. Deep (None, leaf 1).
- Generates `figures/q3_bias_variance_depth.png` and `figures/q3_decomposition_bars.png`.

---

## 📊 Summary of Key Experimental Results

### 1. Robustness & Bias-Variance Master Table
| Model | Clean Test Acc | Masked Test Acc | $\text{Bias}^2$ | Variance | Total Expected Error |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Selected Single Tree** | 50.00% | 5.00% | 0.8124 | 0.4912 | 1.3036 |
| **Bagging ($B=25$)** | **78.75%** | **55.00%** | **0.3241** | **0.0815** | **0.4056** |
| **Boosting ($d=4, r=100$)** | 43.75% | 32.50% | 0.7410 | 0.2185 | 0.9595 |

### 2. Core Takeaways
### 2. Core Takeaways

1. **Bagging Slashes Single-Tree Variance by Over 83%:** In high-dimensional pixel space (4096 dimensions) with few training faces per person, a single decision tree has severe instability ($\text{Variance} = 0.4912$). By averaging independent trees trained on bootstrap samples, Bagging reduces variance down to $0.0815$, boosting clean test accuracy from $50.00\%$ to $78.75\%$.

2. **Single Trees Collapse under Occlusion:** When the central $16 \times 16$ facial patch is blacked out, a single decision tree's accuracy plunges from $50.0\%$ to just $5.0\%$ (near random chance), yet its prediction confidence remains artificially high ($100\%$). In contrast, Bagging retains $55.0\%$ accuracy under occlusion because different bootstrap trees split on diverse peripheral facial features (forehead, jawline, hair).

3. **Boosting Traps on Hard Outliers:** SAMME AdaBoost struggles on small-sample raw image data. Because it exponentially inflates the weights of misclassified images, subsequent trees get stuck repeatedly trying to fit a few noisy or ambiguous faces (e.g., training face #204 absorbed over $17\times$ the baseline uniform weight). This causes high bias and degrades generalization compared to Bagging.

4. **Is Deepest Always Best for Bagging? No:** Conventional intuition suggests growing trees to maximum depth (`max_depth=None, min_samples_leaf=1`). However, my empirical results show that unconstrained deep trees generate singleton leaves ($9.09\%$ of leaves contain only 1 sample) that overfit to camera noise. Applying light regularization (`min_samples_leaf=2`) eliminates these noise splits and gives better ensemble generalization and stability than growing trees to 100% purity.

---

## 📄 CVPR Report Compilation
The report is formatted in two-column CVPR style (`report/main.tex`).
To compile using LaTeX:
```bash
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
