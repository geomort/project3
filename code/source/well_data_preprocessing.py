import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

path = "datasets/Well_data/log_2.csv"

def load_log_data(path: str):
    df = pd.read_csv(path)

    # Target: DPOR
    y = df["DPOR"].values
    X = df.drop(columns=["DPOR"]).values  # use all other columns as features

    # Train / val / test-split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
    y_val   = y_scaler.transform(y_val.reshape(-1,1))
    y_test  = y_scaler.transform(y_test.reshape(-1,1))
    
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)


    # torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val   = torch.tensor(y_val,   dtype=torch.float32).view(-1, 1)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_dataloaders(path: str, batch_size: int = 64):
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_log_data(path)

    train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    input_dim = X_tr.shape[1]

    return train_dl, val_dl, test_dl, input_dim