


import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from torchinfo import summary
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader



#def get_labels_from_subset(subset):
#    labels = []
#    for _, label in subset:  # iterate through subset
#        labels.append(label)
#    return labels

class RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, self.label_map[y]


def create_objective(input_size_img, train_dataset, validation_dataset, device="cpu"):
    # Extract labels from subsets
    train_labels = [y for _, y in train_dataset]
    val_labels = [y for _, y in validation_dataset]

    # Build label map
    unique_labels = sorted(set(train_labels) | set(val_labels))
    label_map = {old_label: new_idx for new_idx, old_label in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    print(f"[INFO] Original labels: {unique_labels}")
    print(f"[INFO] Remapping: {label_map}")
    print(f"[INFO] num_classes = {num_classes}")

    # Wrap subsets with remapping
    train_dataset = RemappedSubset(train_dataset, label_map)
    validation_dataset = RemappedSubset(validation_dataset, label_map)

    def objective(trial):
        # Hyperparameters
        learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.001, 0.01])
        dropout_fc = trial.suggest_categorical('dropout_fc', [0.1, 0.5])
        dropout_conv = trial.suggest_categorical('dropout_conv', [0.0, 0.3])
        activation_name = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'ELU'])
        activation_choice = nn.ReLU if activation_name == 'ReLU' else nn.ELU if activation_name == 'ELU' else nn.LeakyReLU
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])

        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        conv_layers = [(trial.suggest_categorical(f'conv{i+1}_filters', [16, 32, 64, 128]), 3) for i in range(n_conv_layers)]
        n_fc_layers = trial.suggest_int('n_fc_layers', 1, 2)
        fc_layers = [trial.suggest_categorical(f'fc{i+1}_neurons', [16, 32, 64, 128]) for i in range(n_fc_layers)]

        # Model
        model = FlexibleCNN(input_size=input_size_img, num_classes=num_classes,
                            conv_layers=conv_layers, fc_layers=fc_layers,
                            activation=activation_choice, dropout_fc=dropout_fc,
                            dropout_conv=dropout_conv, use_batchnorm=use_batchnorm,
                            pool_type="max", global_pool="avg", show_summary=False)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dl = DataLoader(validation_dataset, batch_size=batch_size)

        history = train_model(model, num_epochs=30, train_dl=train_dl, valid_dl=valid_dl,
                              loss_fn=loss_fn, optimizer=optimizer, device=device,
                              verbose=False, patience_val=5, trial=trial)

        return min(history["valid_acc"])
    return objective

"""
def remap_subset_labels(subset, label_map):
    # Wrap subset to remap labels dynamically
    class RemappedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, label_map):
            self.subset = subset
            self.label_map = label_map

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return x, self.label_map[y]

    return RemappedSubset(subset, label_map)
"""



def train_model(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer, device="cpu",
                verbose=False, patience_val=5, trial=None):
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    best_valid_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_val)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_correct = 0.0, 0
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y_batch.size(0)
            train_correct += (pred.argmax(dim=1) == y_batch).sum().item()

        epoch_train_loss = train_loss / len(train_dl.dataset)
        epoch_train_acc = train_correct / len(train_dl.dataset)

        # Validation
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                valid_loss += loss.item() * y_batch.size(0)
                valid_correct += (pred.argmax(dim=1) == y_batch).sum().item()

        epoch_valid_loss = valid_loss / len(valid_dl.dataset)
        epoch_valid_acc = valid_correct / len(valid_dl.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["valid_loss"].append(epoch_valid_loss)
        history["valid_acc"].append(epoch_valid_acc)

        scheduler.step(epoch_valid_loss)

        # Report to Optuna for pruning
        if trial is not None:
            trial.report(epoch_valid_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()  

        # Early stopping
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_val:
                break

    model.load_state_dict(best_model_wts)
    return history






class FlexibleCNN(nn.Module):
    def __init__(self, input_size, num_classes,
                 conv_layers=[(32, 3), (64, 3), (128, 3)],  # [(out_channels, kernel_size), ...]
                 fc_layers=[256, 128],                     # [units for each FC layer]
                 activation=nn.ReLU,                       # Activation class
                 dropout_fc=0.5,                           # Dropout for FC layers
                 dropout_conv=0.0,                         #  Optional dropout for conv layers
                 use_batchnorm=True,
                 pool_type="max",                          # "max" or "avg"
                 global_pool="avg",                        #  "avg" or "max" for final pooling
                 show_summary=True):
        super().__init__()

        # Choose pooling layers
        pool_layer = nn.MaxPool2d(2) if pool_type == "max" else nn.AvgPool2d(2)
        global_pool_layer = nn.AdaptiveAvgPool2d((1, 1)) if global_pool == "avg" else nn.AdaptiveMaxPool2d((1, 1))

        # Build convolutional feature extractor
        layers = []
        in_channels = input_size[0]
        for out_channels, kernel_size in conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation())
            if dropout_conv > 0:
                layers.append(nn.Dropout2d(dropout_conv))  # ✅ Dropout for conv layers
            layers.append(pool_layer)
            in_channels = out_channels

        layers.append(global_pool_layer)  # ✅ Global pooling
        self.features = nn.Sequential(*layers)

        # Classifier
        in_features = conv_layers[-1][0]
        fc = []
        for units in fc_layers:
            fc.append(nn.Linear(in_features, units))
            fc.append(activation())
            if dropout_fc > 0:
                fc.append(nn.Dropout(dropout_fc))
            in_features = units
        fc.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*fc)

        # Optional summary
        if show_summary:
            summary(self, input_size=(1, *input_size), col_names=["input_size", "output_size", "num_params"])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

