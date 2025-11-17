


import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from torchinfo import summary


def train_model(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer, device="cpu", verbose=False, patience=5):
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    best_valid_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    # Scheduler: reduce LR if validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_correct = 0.0, 0

        train_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for x_batch, y_batch in train_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y_batch.size(0)
            train_correct += (pred.argmax(dim=1) == y_batch).sum().item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = train_loss / len(train_dl.dataset)
        epoch_train_acc = train_correct / len(train_dl.dataset)

        # Validation
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        valid_bar = tqdm(valid_dl, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=False)

        with torch.no_grad():
            for x_batch, y_batch in valid_bar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

                valid_loss += loss.item() * y_batch.size(0)
                valid_correct += (pred.argmax(dim=1) == y_batch).sum().item()
                valid_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_valid_loss = valid_loss / len(valid_dl.dataset)
        epoch_valid_acc = valid_correct / len(valid_dl.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["valid_loss"].append(epoch_valid_loss)
        history["valid_acc"].append(epoch_valid_acc)

        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                  f"Valid Loss: {epoch_valid_loss:.4f}, Valid Acc: {epoch_valid_acc:.4f}")

        # Step scheduler based on validation loss
        scheduler.step(epoch_valid_loss)

        # Early stopping check
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_valid_loss:.4f}")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)
    return history



class FlexibleCNN(nn.Module):
    def __init__(self, input_size, num_classes,
                 conv_layers=[(32, 3), (64, 3), (128, 3)],  # [(out_channels, kernel_size), ...]
                 fc_layers=[256, 128],                     # [units for each FC layer]
                 activation=nn.ReLU,                       # Activation class
                 dropout_fc=0.5,                           # Dropout for FC layers
                 dropout_conv=0.0,                         # ✅ Optional dropout for conv layers
                 use_batchnorm=True,
                 pool_type="max",                          # ✅ "max" or "avg"
                 global_pool="avg",                        # ✅ "avg" or "max" for final pooling
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


class FlexibleCNN_backup(nn.Module):
    def __init__(self, input_size, num_classes,
                 conv_layers=[(16, 5), (64, 5)],  # [(out_channels, kernel_size), ...]
                 fc_layers=[128],                # [units for each FC layer]
                 activation=nn.ReLU,             # activation class
                 dropout=0.5):
        super().__init__()

        # Build convolutional feature extractor
        layers = []
        in_channels = input_size[0]
        for out_channels, kernel_size in conv_layers:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(activation())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        # Add Adaptive Pooling to avoid huge flatten size
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # Global pooling
        self.features = nn.Sequential(*layers)

        # Classifier: now input is just number of channels (last conv layer)
        in_features = conv_layers[-1][0]
        fc = []
        for units in fc_layers:
            fc.append(nn.Linear(in_features, units))
            fc.append(activation())
            if dropout:
                fc.append(nn.Dropout(dropout))
            in_features = units
        fc.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten only channels (batch_size, channels)
        return self.classifier(x)


class CNN(nn.Module):
    def __init__(self, num_classes, input_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Compute in_features dynamically based on input_size
        self._to_linear = self._get_flatten_size(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def _get_flatten_size(self, input_size):
        x = torch.randn(1, *input_size)  # dynamic input size
        out = self.features(x)
        return out.numel()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)