"""
Dataset loading and preprocessing module for Olivetti Faces.
Includes deterministic identity-stratified train/val/test splitting
and generation of the masked test set for robustness evaluation.
"""

import os
import re
import numpy as np
from sklearn.datasets import fetch_olivetti_faces


def extract_numeric_seed(roll_number: str) -> int:
    """
    Extracts all numeric characters from the roll number in order
    and returns them as an integer seed.
    Example: 'M26MAC003' -> 26003
    """
    digits = re.findall(r'\d', roll_number)
    if not digits:
        raise ValueError(f"No numeric characters found in roll number: {roll_number}")
    return int("".join(digits))


def get_stratified_split(target: np.ndarray, seed: int):
    """
    Within each of the 40 identities, randomly splits the 10 images into
    6 train, 2 validation, and 2 test images using numpy.random.default_rng(seed).
    
    Returns:
        train_idx, val_idx, test_idx (1D numpy arrays of indices)
    """
    rng = np.random.default_rng(seed)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    unique_classes = np.unique(target)
    for c in unique_classes:
        c_indices = np.where(target == c)[0]
        # Randomly permute the 10 images of this identity
        shuffled = rng.permutation(c_indices)
        train_indices.extend(shuffled[:6])
        val_indices.extend(shuffled[6:8])
        test_indices.extend(shuffled[8:10])
        
    return (np.array(train_indices, dtype=int),
            np.array(val_indices, dtype=int),
            np.array(test_indices, dtype=int))


def create_masked_images(images_flat: np.ndarray, row_start=24, row_end=40, col_start=24, col_end=40) -> np.ndarray:
    """
    Creates a masked copy of face images by setting the centred 16x16 block
    (rows 24 to 39, columns 24 to 39) to zero.
    
    Args:
        images_flat: Array of shape (N, 4096)
        row_start, row_end: Row bounds for slicing [24, 40)
        col_start, col_end: Column bounds for slicing [24, 40)
        
    Returns:
        masked_flat: Array of shape (N, 4096) with center 16x16 zeroed out
    """
    masked = images_flat.copy().reshape(-1, 64, 64)
    masked[:, row_start:row_end, col_start:col_end] = 0.0
    return masked.reshape(-1, 4096)


def load_and_prepare_data(roll_number: str = "M26MAC003", split_save_path: str = "data/split_indices.npz"):
    """
    Loads Olivetti Faces, generates or loads the deterministic split,
    and returns (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_test_masked, y_test).
    """
    seed = extract_numeric_seed(roll_number)
    faces = fetch_olivetti_faces()
    X_all = faces.data   # shape: (400, 4096), float32 in [0, 1]
    y_all = faces.target # shape: (400,), int 0..39
    
    # Check if split file already exists
    if split_save_path and os.path.exists(split_save_path):
        data = np.load(split_save_path)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        test_idx = data["test_idx"]
        saved_seed = int(data["seed"])
        if saved_seed != seed:
            print(f"[Warning] Saved seed ({saved_seed}) differs from requested seed ({seed}). Regenerating split.")
            train_idx, val_idx, test_idx = get_stratified_split(y_all, seed)
            np.savez(split_save_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=seed, roll_number=roll_number)
    else:
        train_idx, val_idx, test_idx = get_stratified_split(y_all, seed)
        if split_save_path:
            os.makedirs(os.path.dirname(split_save_path), exist_ok=True)
            np.savez(split_save_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=seed, roll_number=roll_number)
            print(f"[Info] Split saved successfully to {split_save_path}")

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    # Create masked test set (centred 16x16 zeroed out)
    X_test_masked = create_masked_images(X_test, 24, 40, 24, 40)
    
    return {
        "seed": seed,
        "roll_number": roll_number,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_masked": X_test_masked,
        "X_all": X_all,
        "y_all": y_all,
    }


if __name__ == "__main__":
    data_dict = load_and_prepare_data()
    print(f"Roll Number: {data_dict['roll_number']} -> Seed: {data_dict['seed']}")
    print(f"Train set: {data_dict['X_train'].shape}, classes: {len(np.unique(data_dict['y_train']))}")
    print(f"Val set:   {data_dict['X_val'].shape}, classes: {len(np.unique(data_dict['y_val']))}")
    print(f"Test set:  {data_dict['X_test'].shape}, classes: {len(np.unique(data_dict['y_test']))}")
    print(f"Masked Test set: {data_dict['X_test_masked'].shape}")
    print("Checking mask: center pixels zeroed?", (data_dict['X_test_masked'].reshape(-1, 64, 64)[:, 24:40, 24:40] == 0).all())
