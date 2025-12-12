import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    from .ffnn_well_data_preprocessing import load_log_data
except ImportError:
    from ffnn_well_data_preprocessing import load_log_data


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


