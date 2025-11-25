import torch.nn as nn
import torch
import copy
import optuna
from tqdm import tqdm



class FlexibleCNN(nn.Module):
    def __init__(self, input_size, num_classes,
                 conv_layers=[(32, 3), (64, 3), (128, 3)],  # [(out_channels, kernel_size), ...]
                 fc_layers=[256, 128],                     # [units for each FC layer]
                 activation=nn.ReLU,                       # Activation class
                 dropout_fc=0.5,                           # Dropout for FC layers
                 dropout_conv=0.0,                         #  Optional dropout for conv layers
                 use_batchnorm=True,
                 pool_type="max",                          # "max" or "avg"
                 global_pool="avg"):
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


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)




def train_model(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer, device="cpu",
                verbose=False, patience_val=10, trial=None):
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    best_valid_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_val)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_correct = 0.0, 0

        train_bar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=True)
        for batch in train_bar:
            if isinstance(batch, dict):
                if 'image' in batch:
                    x_batch = batch['image']
                elif 'images' in batch:
                    x_batch = batch['images']
                else:
                    raise KeyError("Neither 'image' nor 'images' found in batch.")

                if 'label' in batch:
                    y_batch = batch['label']
                elif 'labels' in batch:
                    y_batch = batch['labels']
                else:
                    raise KeyError("Neither 'label' nor 'labels' found in batch.")
            else:
                x_batch, y_batch = batch[0], batch[1]

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y_batch.size(0)
            train_correct += (pred.argmax(dim=1) == y_batch).sum().item()

            current_acc = train_correct / ((train_bar.n + 1) * train_dl.batch_size)
            train_bar.set_postfix(loss=loss.item(), acc=f"{current_acc:.4f}")

        epoch_train_loss = train_loss / len(train_dl.dataset)
        epoch_train_acc = train_correct / len(train_dl.dataset)

        # Validation
        model.eval()
        valid_loss, valid_correct = 0.0, 0
        valid_bar = tqdm(valid_dl, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=True)
        with torch.no_grad():
            for batch in valid_bar:
                if isinstance(batch, dict):
                    if 'image' in batch:
                        x_batch = batch['image']
                    elif 'images' in batch:
                        x_batch = batch['images']
                    else:
                        raise KeyError("Neither 'image' nor 'images' found in batch.")

                    if 'label' in batch:
                        y_batch = batch['label']
                    elif 'labels' in batch:
                        y_batch = batch['labels']
                    else:
                        raise KeyError("Neither 'label' nor 'labels' found in batch.")
                else:
                    x_batch, y_batch = batch[0], batch[1]

                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)

                valid_loss += loss.item() * y_batch.size(0)
                valid_correct += (pred.argmax(dim=1) == y_batch).sum().item()

                current_val_acc = valid_correct / ((valid_bar.n + 1) * valid_dl.batch_size)
                valid_bar.set_postfix(loss=loss.item(), acc=f"{current_val_acc:.4f}")

        epoch_valid_loss = valid_loss / len(valid_dl.dataset)
        epoch_valid_acc = valid_correct / len(valid_dl.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["valid_loss"].append(epoch_valid_loss)
        history["valid_acc"].append(epoch_valid_acc)

        scheduler.step(epoch_valid_loss)

        # Optuna pruning
        if trial is not None:
            trial.report(epoch_valid_acc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping based on accuracy
        if epoch_valid_acc > best_valid_acc:
            best_valid_acc = epoch_valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_val:
                tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
                break

        tqdm.write(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {epoch_train_acc:.4f} | Valid Acc: {epoch_valid_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return history


