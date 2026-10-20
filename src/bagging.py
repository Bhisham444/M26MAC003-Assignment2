"""
Custom implementation of BaggingClassifier from scratch.
Supports bootstrap sampling, Out-Of-Bag (OOB) evaluation,
pairwise tree disagreement, and vote margin computation.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional


class CustomBaggingClassifier:
    def __init__(
        self,
        base_estimator_params: dict,
        n_estimators: int = 50,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            base_estimator_params: Dictionary of parameters for DecisionTreeClassifier
                                  (e.g., max_depth, min_samples_leaf)
            n_estimators: Number of bootstrap trees (B)
            random_state: Seed for reproducibility
        """
        self.base_estimator_params = base_estimator_params
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_: List[DecisionTreeClassifier] = []
        self.bootstrap_indices_: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits n_estimators trees on bootstrap samples of (X, y).
        """
        n_samples, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        rng = np.random.default_rng(self.random_state)
        self.bootstrap_indices_ = [
            rng.choice(n_samples, size=n_samples, replace=True)
            for _ in range(self.n_estimators)
        ]
        tree_seeds = [
            None if self.random_state is None else int(rng.integers(0, 1000000))
            for _ in range(self.n_estimators)
        ]

        def _fit_single_tree(b):
            bag_idx = self.bootstrap_indices_[b]
            tree = DecisionTreeClassifier(
                **self.base_estimator_params,
                random_state=tree_seeds[b]
            )
            tree.fit(X[bag_idx], y[bag_idx])
            return tree

        with ThreadPoolExecutor(max_workers=4) as executor:
            self.estimators_ = list(executor.map(_fit_single_tree, range(self.n_estimators)))

        return self

    def get_sub_ensemble(self, B: int):
        """
        Returns a sub-ensemble containing the first B estimators.
        Avoids redundant training when evaluating ensembles of increasing sizes.
        """
        assert B <= len(self.estimators_), f"Requested {B} trees, but model only has {len(self.estimators_)}"
        sub = CustomBaggingClassifier(
            base_estimator_params=self.base_estimator_params,
            n_estimators=B,
            random_state=None
        )
        sub.estimators_ = self.estimators_[:B]
        sub.bootstrap_indices_ = self.bootstrap_indices_[:B]
        sub.classes_ = self.classes_
        sub.n_classes_ = self.n_classes_
        sub.n_features_ = self.n_features_
        return sub

    def predict_individual(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predictions of all individual estimators.
        Shape: (n_estimators, n_samples)
        """
        preds = np.zeros((self.n_estimators, X.shape[0]), dtype=int)
        for b, tree in enumerate(self.estimators_):
            preds[b] = tree.predict(X)
        return preds

    def predict_vote_counts(self, X: np.ndarray) -> np.ndarray:
        """
        Returns vote counts for each class for every sample in X.
        Shape: (n_samples, n_classes)
        """
        individual_preds = self.predict_individual(X) # (B, N)
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes_), dtype=int)
        
        for k_idx, k in enumerate(self.classes_):
            votes[:, k_idx] = np.sum(individual_preds == k, axis=0)
            
        return votes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns normalized vote fractions across classes for each sample.
        Shape: (n_samples, n_classes)
        """
        votes = self.predict_vote_counts(X)
        return votes / float(self.n_estimators)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Majority vote prediction.
        """
        votes = self.predict_vote_counts(X)
        winning_class_indices = np.argmax(votes, axis=1)
        return self.classes_[winning_class_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns classification accuracy.
        """
        return np.mean(self.predict(X) == y)

    def compute_oob_accuracy(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[float, int]:
        """
        Computes Out-Of-Bag (OOB) accuracy on the training set.
        Returns:
            (oob_accuracy, num_evaluated_samples)
        """
        n_samples = X_train.shape[0]
        oob_votes = np.zeros((n_samples, self.n_classes_), dtype=int)
        oob_counts = np.zeros(n_samples, dtype=int)

        # For each tree, find samples that were NOT in its bootstrap bag
        for b in range(self.n_estimators):
            in_bag = np.zeros(n_samples, dtype=bool)
            in_bag[self.bootstrap_indices_[b]] = True
            oob_indices = np.where(~in_bag)[0]

            if len(oob_indices) > 0:
                preds = self.estimators_[b].predict(X_train[oob_indices])
                for idx, p in zip(oob_indices, preds):
                    c_idx = np.where(self.classes_ == p)[0][0]
                    oob_votes[idx, c_idx] += 1
                    oob_counts[idx] += 1

        # Evaluate accuracy only on samples with at least one OOB tree
        valid_oob_mask = oob_counts > 0
        if not np.any(valid_oob_mask):
            return 0.0, 0

        oob_preds = self.classes_[np.argmax(oob_votes[valid_oob_mask], axis=1)]
        oob_acc = np.mean(oob_preds == y_train[valid_oob_mask])
        return float(oob_acc), int(np.sum(valid_oob_mask))

    def compute_pairwise_disagreement(self, X: np.ndarray) -> float:
        """
        Computes the average pairwise disagreement between trees on X:
        Disagreement = (2 / (B*(B-1))) * sum_{i < j} mean(h_i(X) != h_j(X))
        """
        B = self.n_estimators
        if B <= 1:
            return 0.0

        individual_preds = self.predict_individual(X) # (B, N)
        total_disagreements = 0
        pair_count = 0

        for i in range(B):
            for j in range(i + 1, B):
                mismatch_fraction = np.mean(individual_preds[i] != individual_preds[j])
                total_disagreements += mismatch_fraction
                pair_count += 1

        return float(total_disagreements / pair_count)

    def compute_vote_margin(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the vote margin for each sample in X:
        Vote Margin = (winning vote fraction) - (runner-up vote fraction)
        
        Returns:
            margins: array of margins per sample (N,)
            mean_margin: scalar float
        """
        votes = self.predict_vote_counts(X) # (N, n_classes)
        # Sort vote counts in descending order along class axis
        sorted_votes = np.sort(votes, axis=1)[:, ::-1]
        
        top1 = sorted_votes[:, 0] / float(self.n_estimators)
        top2 = sorted_votes[:, 1] / float(self.n_estimators) if self.n_classes_ > 1 else np.zeros_like(top1)
        margins = top1 - top2
        return margins, float(np.mean(margins))

    def average_distinct_samples_per_bag(self) -> float:
        """
        Computes the empirical average number of distinct samples in one bootstrap sample.
        """
        distinct_counts = [len(np.unique(bag)) for bag in self.bootstrap_indices_]
        return float(np.mean(distinct_counts))
