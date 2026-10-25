

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, List, Tuple, Optional


class CustomSAMMEAdaBoost:
    def __init__(
        self,
        base_depth: int = 1,
        n_rounds: int = 50,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            base_depth: Max depth of the DecisionTreeClassifier weak learner
            n_rounds: Maximum number of boosting rounds (T)
            random_state: Seed for reproducibility
        """
        self.base_depth = base_depth
        self.n_rounds = n_rounds
        self.random_state = random_state
        
        self.estimators_: List[DecisionTreeClassifier] = []
        self.estimator_weights_: List[float] = [] # alpha_t
        self.estimator_errors_: List[float] = []  # epsilon_t
        
        # Diagnostic tracking per round
        self.sample_weights_history_: List[np.ndarray] = []
        self.max_sample_weight_history_: List[float] = []
        self.top10_weight_history_: List[float] = []
        self.learner_predictions_history_: List[np.ndarray] = []
        
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits SAMME AdaBoost on (X, y).
        """
        n_samples, self.n_features_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        K = self.n_classes_
        
        rng = np.random.default_rng(self.random_state)
        
        # Initialize sample weights uniformly: w_1(i) = 1 / N
        w = np.ones(n_samples, dtype=float) / float(n_samples)
        
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.sample_weights_history_ = []
        self.max_sample_weight_history_ = []
        self.top10_weight_history_ = []
        self.learner_predictions_history_ = []

        random_error_bound = 1.0 - (1.0 / K) # Error of random guessing
        eps_numerical = 1e-12

        for t in range(self.n_rounds):
            # Record current weights before update
            self.sample_weights_history_.append(w.copy())
            self.max_sample_weight_history_.append(float(np.max(w)))
            # Total weight of top 10 heaviest training images
            sorted_w = np.sort(w)[::-1]
            self.top10_weight_history_.append(float(np.sum(sorted_w[:min(10, n_samples)])))
            
            # Step 1: Fit weak learner h_t(x) using sample weights w
            tree_seed = None if self.random_state is None else int(rng.integers(0, 1000000))
            learner = DecisionTreeClassifier(
                max_depth=self.base_depth,
                random_state=tree_seed
            )
            learner.fit(X, y, sample_weight=w)
            
            # Step 2: Predictions and weighted error
            y_pred = learner.predict(X)
            self.learner_predictions_history_.append(y_pred)
            
            misclassified = (y != y_pred).astype(float)
            weighted_err = np.sum(w * misclassified) / np.sum(w)
            
            # Explicit handling: weak learner not better than random guessing
            if weighted_err >= random_error_bound:
                print(f"[AdaBoost Info] Round {t+1}: Weighted error {weighted_err:.4f} >= random threshold {random_error_bound:.4f}. Terminating boosting early.")
                if len(self.estimators_) == 0:
                    # Keep at least this first learner with small positive weight so model is usable
                    self.estimators_.append(learner)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(weighted_err)
                break
                
            # If error is practically zero, avoid division by zero
            if weighted_err <= eps_numerical:
                alpha = np.log((1.0 - eps_numerical) / eps_numerical) + np.log(K - 1)
                self.estimators_.append(learner)
                self.estimator_weights_.append(float(alpha))
                self.estimator_errors_.append(float(weighted_err))
                print(f"[AdaBoost Info] Round {t+1}: Perfect training error reached (0.0). Terminating early.")
                break
                
            # Step 3: Compute alpha_t = ln((1 - eps) / eps) + ln(K - 1)
            alpha = np.log((1.0 - weighted_err) / weighted_err) + np.log(K - 1)
            
            self.estimators_.append(learner)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(float(weighted_err))
            
            # Step 4: Update weights
            # w_{t+1}(i) = w_t(i) * exp(alpha * I(y_i != y_pred_i))
            w = w * np.exp(alpha * misclassified)
            
            # Step 5: Normalize weights
            w = w / np.sum(w)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Computes multiclass SAMME scores:
        S_k(x) = sum_t alpha_t * I(h_t(x) == k)
        Returns:
            S of shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        S = np.zeros((n_samples, self.n_classes_), dtype=float)
        
        for learner, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = learner.predict(X)
            for k_idx, k in enumerate(self.classes_):
                S[:, k_idx] += alpha * (pred == k).astype(float)
                
        return S

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns normalized weighted vote confidence:
        q_k(x) = S_k(x) / sum_t alpha_t
        """
        S = self.decision_function(X)
        total_alpha = np.sum(self.estimator_weights_)
        if total_alpha <= 0:
            return np.ones_like(S) / float(self.n_classes_)
        return S / total_alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predicted class label: argmax_k S_k(x)
        """
        S = self.decision_function(X)
        best_indices = np.argmax(S, axis=1)
        return self.classes_[best_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns classification accuracy.
        """
        return float(np.mean(self.predict(X) == y))

    def get_sample_prediction_history(self, sample_idx: int, y_true: int) -> Dict:
        """
        For a given training sample index, returns its prediction history across rounds:
        - predictions of each learner
        - whether it was misclassified at each round
        - its weight progression over rounds
        """
        history = {
            "sample_idx": sample_idx,
            "true_label": y_true,
            "predictions": [preds[sample_idx] for preds in self.learner_predictions_history_],
            "is_misclassified": [preds[sample_idx] != y_true for preds in self.learner_predictions_history_],
            "weights": [w[sample_idx] for w in self.sample_weights_history_],
            "alphas": list(self.estimator_weights_),
            "errors": list(self.estimator_errors_),
        }
        return history
