import os
from pathlib import Path

import numpy as np
import lasio
import torch
from sklearn.preprocessing import StandardScaler


def load_log_data(
    path: str | None = None,
    random_state: int | None = None,
    split_method: str = "random",   # "random" or "depth"
    gap: int = 0                    # number of samples as a buffer between splits
):
    """
    Load well-log data from a LAS file and return train/val/test tensors.

    Parameters
    ----------
    path : str, optional
        Path to the LAS file. Can be absolute or relative.
        If None, a default file path inside the project structure is used.
    random_state : int, optional
        Seed for the random train/val/test split. If None, a fresh RNG is used.
    split_method : str
        Split strategy: "random" or "depth".
        - "random": random permutation split
        - "depth": contiguous intervals after sorting by depth
    gap : int
        Buffer (in number of samples) inserted between train/val and val/test
        for the "depth" split to reduce neighbor leakage.

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), x_scaler, y_scaler
        Torch tensors and scalers for training, validation, and testing.
    """

    # ----------------------------------------------------------------------
    # Resolve LAS file path
    # ----------------------------------------------------------------------
    if path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        repo_root = Path(project_root)

        candidates = list(repo_root.glob("**/1_9-7*LOGS*.LAS"))
        if not candidates:
            candidates = list(repo_root.glob("datasets/well_data/**/*.LAS")) + \
                         list(repo_root.glob("datasets/well_data/*.LAS"))

        if not candidates:
            raise FileNotFoundError(
                f"Could not find LAS file under {project_root}. "
                f"Expected something like '1_9-7*LOGS*.LAS' in datasets/well_data."
            )

        las_path = str(candidates[0])
        print(f"[load_log_data] Using LAS file: {las_path}")

    else:
        if not os.path.isabs(path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            las_path = os.path.join(current_dir, path)
        else:
            las_path = path
        print(f"[load_log_data] Using LAS file from argument: {las_path}")

    # ----------------------------------------------------------------------
    # Read LAS file into a DataFrame
    # ----------------------------------------------------------------------
    try:
        las = lasio.read(las_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read LAS file at {las_path}: {e}")

    df = las.df().reset_index()

    # ----------------------------------------------------------------------
    # Automatically detect depth column (common names: DEPT, DEPTH)
    # ----------------------------------------------------------------------
    depth_col = None
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("dept") or cl == "depth":
            depth_col = c
            break

    if depth_col is None:
        raise KeyError(
            f"No depth column found in LAS. Available columns: {list(df.columns)}"
        )

    # ----------------------------------------------------------------------
    # Select target log curve (AC). Raise error if not found.
    # ----------------------------------------------------------------------
    target_col = "AC"
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in LAS file. "
            f"Available columns: {list(df.columns)}"
        )

    # Features = all columns except the target
    feature_cols = [col for col in df.columns if col != target_col]

    # Remove depth from the feature list so we don't leak positional info as a feature
    if depth_col in feature_cols:
        feature_cols.remove(depth_col)

    # ----------------------------------------------------------------------
    # Replace typical null values in LAS files with NaN
    # ----------------------------------------------------------------------
    df = df.replace([-999.25, -999.0, -9999.25], np.nan)

    # Keep only relevant columns: depth, target, and features
    df_model = df[[depth_col, target_col] + feature_cols]

    # Remove NaN and infinite values
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()

    # Sort by depth (required for depth-based split; harmless for random split)
    df_model = df_model.sort_values(depth_col).reset_index(drop=True)

    # ----------------------------------------------------------------------
    # Convert to numpy arrays
    # ----------------------------------------------------------------------
    depth = df_model[depth_col].values
    y = df_model[target_col].values
    X = df_model[feature_cols].values

    # Extra safety: remove any remaining invalid rows
    valid_indices = ~(
        np.isnan(X).any(axis=1)
        | np.isinf(X).any(axis=1)
        | np.isnan(y)
        | np.isinf(y)
    )
    print(f"Removing {(~valid_indices).sum()} rows with NaN/inf values")

    X = X[valid_indices]
    y = y[valid_indices]
    depth = depth[valid_indices]

    print(f"Data shape after NaN removal: X={X.shape}, y={y.shape}")

    # ----------------------------------------------------------------------
    # Split indices: 70% train, 15% val, 15% test
    # ----------------------------------------------------------------------
    N = X.shape[0]
    train_end = int(0.7 * N)
    val_end = int(0.85 * N)

    if split_method == "random":
        # Random permutation split (seeded if random_state is provided)
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(N)

        train_idx = indices[:train_end]
        val_idx   = indices[train_end:val_end]
        test_idx  = indices[val_end:]

    elif split_method == "depth":
        # Depth-based split: contiguous intervals (data already sorted by depth)
        # gap is a buffer (in samples) to reduce neighbor leakage between sets
        train_idx = np.arange(0, train_end)

        val_start = min(train_end + gap, N)
        val_len = val_end - train_end
        val_idx = np.arange(val_start, min(val_start + val_len, N))

        # Start test after validation, with another gap if possible
        test_start = (val_idx[-1] + 1 + gap) if len(val_idx) > 0 else val_start
        test_start = min(test_start, N)
        test_idx = np.arange(test_start, N)

    else:
        raise ValueError("split_method must be 'random' or 'depth'")

    # Create split datasets (shared for both split methods)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Optional safety check: ensure val/test are non-empty (especially with large gap)
    if X_val.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Validation or test set became empty. Reduce 'gap' or adjust split fractions.")

    # ----------------------------------------------------------------------
    # Standardize feature matrix X (fit on train only)
    # ----------------------------------------------------------------------
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)

    # ----------------------------------------------------------------------
    # Standardize target y (fit on train only)
    # ----------------------------------------------------------------------
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

    # Handle potential NaNs after scaling (should be rare)
    if np.isnan(y_train_scaled).any():
        print("Warning: NaN values detected after scaling y_train")
        valid_mask = ~np.isnan(y_train_scaled.flatten())
        y_train_scaled = y_train_scaled[valid_mask]
        X_train = X_train[valid_mask]

    y_train = y_train_scaled
    y_val   = y_scaler.transform(y_val.reshape(-1, 1))
    y_test  = y_scaler.transform(y_test.reshape(-1, 1))

    # ----------------------------------------------------------------------
    # Convert arrays to torch tensors
    # ----------------------------------------------------------------------
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val,   dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), x_scaler, y_scaler
