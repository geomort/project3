def load_log_data(path: str = None):
    """
    Load well-log data from a LAS file and return train/val/test tensors.

    Parameters
    ----------
    path : str, optional
        Path to the LAS file, relative to this file or absolute.
        If None, uses default path to 1_9-7_LOGS.LAS

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), x_scaler, y_scaler
    """

    # Build absolute path relative to this file if needed
    if path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        las_path = os.path.join(current_dir, "../../datasets/Well_data/1_9-7_LOGS.LAS")
    elif not os.path.isabs(path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        las_path = os.path.join(current_dir, path)
    else:
        las_path = path

    if not os.path.exists(las_path):
        raise FileNotFoundError(f"LAS file not found at: {las_path}")

    # Read LAS and få depth som egen kolonne
    las = lasio.read(las_path)
    df = las.df().reset_index()  # depth blir egen kolonne

    # Finn depth-kolonnen (DEPT, DEPTH, etc.)
    depth_col = None
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("dept") or cl == "depth":
            depth_col = c
            break

    if depth_col is None:
        raise KeyError(
            f"Finner ingen depth-kolonne i LAS-data. Kolonner: {list(df.columns)}"
        )

    # Velg target
    target_col = "AC"
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in LAS file. "
            f"Available columns: {list(df.columns)}"
        )

    # Alle kolonner utenom target er først features
    feature_cols = [col for col in df.columns if col != target_col]

    # Fjern depth fra features
    if depth_col in feature_cols:
        feature_cols.remove(depth_col)

    # Håndter null-verdier/-999 etc.
    df = df.replace([-999.25, -999.0, -9999.25], np.nan)

    # Vi holder på depth, target og features
    df_model = df[[depth_col, target_col] + feature_cols]

    # Dropp NaN/inf
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna()

    # Sorter etter dyp (viktig før vi splitter!)
    df_model = df_model.sort_values(depth_col).reset_index(drop=True)

    # Lag numpy-arrays
    depth = df_model[depth_col].values          # (N,)
    y = df_model[target_col].values             # (N,)
    X = df_model[feature_cols].values           # (N, n_features)

    # Sjekk for NaN/inf i X/y
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

    # 🚨 HER ER DEN VIKTIGE ENDRINGEN: depth-basert, ikke tilfeldig

    N = X.shape[0]
    train_end = int(0.7 * N)   # øverste 70 % av dypene
    val_end   = int(0.85 * N)  # neste 15 % til val, siste 15 % til test

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    # Skalering av X
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)

    # Skalering av y
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))

    if np.isnan(y_train_scaled).any():
        print(f"Warning: {np.isnan(y_train_scaled).sum()} NaN values found after y_train scaling")
        print(
            f"y_train stats before scaling: min={np.nanmin(y_train)}, max={np.nanmax(y_train)}, "
            f"mean={np.nanmean(y_train)}, std={np.nanstd(y_train)}"
        )
        valid_mask = ~np.isnan(y_train_scaled.flatten())
        y_train_scaled = y_train_scaled[valid_mask]
        X_train = X_train[valid_mask]

    y_train = y_train_scaled
    y_val   = y_scaler.transform(y_val.reshape(-1, 1))
    y_test  = y_scaler.transform(y_test.reshape(-1, 1))

    # Til PyTorch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val,   dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), x_scaler, y_scaler
