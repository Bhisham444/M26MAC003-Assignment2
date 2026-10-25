
import numpy as np
from typing import Tuple, Dict


def one_hot_encode(y: np.ndarray, n_classes: int = 40) -> np.ndarray:
    """
    Converts 1D array of class labels into (N, n_classes) one-hot matrix.
    """
    N = len(y)
    one_hot = np.zeros((N, n_classes), dtype=float)
    one_hot[np.arange(N), y] = 1.0
    return one_hot


def compute_bias_variance(p_runs: np.ndarray, y_true: np.ndarray, n_classes: int = 40) -> Dict[str, float]:
    """
    Computes Bias^2 and Variance across R repetitions.
    
    Args:
        p_runs: Array of shape (R, N, K), containing predicted probability/vote vectors
                for repetition r, sample i, class k.
        y_true: Ground truth labels of shape (N,)
        n_classes: Total number of classes (K)
        
    Returns:
        Dictionary with:
            - bias_squared: float
            - variance: float
            - total_loss: float (bias^2 + variance)
    """
    R, N, K = p_runs.shape
    assert len(y_true) == N, f"Length mismatch: {len(y_true)} vs {N}"
    
    # One-hot target matrix e_i: shape (N, K)
    e = one_hot_encode(y_true, n_classes=K)
    
    # Mean prediction 
    p_bar = np.mean(p_runs, axis=0) # (N, K)
    
    # Bias^2 = 
    diff_bias = p_bar - e # (N, K)
    bias_squared = float(np.mean(np.sum(diff_bias ** 2, axis=1)))
    
    # Variance 
    # p_runs is (R, N, K), p_bar is (N, K) -> broadcast along R
    diff_var = p_runs - p_bar[np.newaxis, :, :] # (R, N, K)
    variance = float(np.mean(np.sum(diff_var ** 2, axis=2)))
    
    return {
        "bias_squared": bias_squared,
        "variance": variance,
        "total_expected_error": bias_squared + variance
    }


def compute_vote_margin(votes: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Votes shape: (N, K)
    Margin = fraction(winner) - fraction(runner_up)
    """
    N = votes.shape[0]
    total_votes = np.sum(votes, axis=1, keepdims=True)
    total_votes[total_votes == 0] = 1.0 # avoid div by zero
    fractions = votes / total_votes
    
    sorted_fracs = np.sort(fractions, axis=1)[:, ::-1]
    winner = sorted_fracs[:, 0]
    runner_up = sorted_fracs[:, 1] if sorted_fracs.shape[1] > 1 else np.zeros_like(winner)
    margins = winner - runner_up
    return margins, float(np.mean(margins))
