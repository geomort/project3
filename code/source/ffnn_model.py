import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .well_data_preprocessing import load_log_data

def get_well_dataloaders(
    csv_path: str,
    batch_size: int = 64,
    shuffle_train: bool = True,
):
    """
    Loading well data and returning dataloaders for training, validation, and testing.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_log_data(csv_path)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]

    return train_dl, val_dl, test_dl, input_dim


class WellLogFFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims=(64, 64),
        output_dim: int = 1,
        dropout: float = 0.2,
        activation: nn.Module = nn.ReLU,
    ):
        """
        Feedforward neural network for well log data regression.
        Parameters:
            input_dim (int): Number of input features.
            hidden_dims (tuple): Sizes of hidden layers.        
            output_dim (int): Number of output features.
            dropout (float): Dropout rate between layers.
            activation (nn.Module): Activation function class to use.
        """
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        # last layer to output dimension without activation
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    # example usage
    csv_path = "datasets/Well_data/log_2.csv"
    train_dl, val_dl, test_dl, input_dim = get_well_dataloaders(csv_path, batch_size=64)

    model = WellLogFFNN(input_dim=input_dim, hidden_dims=(64, 64), dropout=0.2)
    print(model)

