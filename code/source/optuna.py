
from source.cnn_model import FlexibleCNN, train_model

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import optuna

def create_objective(num_classes, input_size_img, train_dataset, validation_dataset, num_epochs, device="cpu"):
    print(f"[INFO] num_classes = {num_classes}")


    def objective(trial):
        # --- Hyperparameters ---
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout_fc = trial.suggest_categorical('dropout_fc', [0.3, 0.5])
        dropout_conv = trial.suggest_categorical('dropout_conv', [0.0, 0.3])


        n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)  # 2 to 4 layers
        conv_layers = [
            (
                trial.suggest_categorical(f'conv{i+1}_filters', [32, 64, 128, 256]),
                3  # fixed kernel size
            )
            for i in range(n_conv_layers)
        ]

        # optional 
        use_hidden = trial.suggest_categorical('use_hidden_layer', [True, False])
        fc_layers = [num_classes]  # start with classification layer only
        if use_hidden:
            hidden_neurons = trial.suggest_categorical('fc_hidden_neurons', [32, 64, 128, 256])
            fc_layers.insert(0, hidden_neurons)  # add hidden layer before output


        # Model
        model = FlexibleCNN(
            input_size=input_size_img,
            num_classes=num_classes,
            conv_layers=conv_layers,
            fc_layers=fc_layers,
            activation=nn.ReLU,
            dropout_fc=dropout_fc,
            dropout_conv=dropout_conv,
            use_batchnorm=True,
            pool_type="max",
            global_pool="avg"
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        batch_size = 32
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dl = DataLoader(validation_dataset, batch_size=batch_size)

        history = train_model(model, num_epochs, train_dl=train_dl, valid_dl=valid_dl,
                            loss_fn=loss_fn, optimizer=optimizer, device=device,
                            trial=trial)

        return max(history["valid_acc"])

    return objective


