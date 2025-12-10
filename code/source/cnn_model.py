import torch.nn as nn
import torch
import copy
import optuna
from tqdm import tqdm



class FlexibleCNN(nn.Module):

    """
    Docstrings made with Copilot and edited
    A configurable Convolutional Neural Network for image classification.

    Parameters
    ----------
    input_size : tuple
        Input image size as (channels, height, width).
    num_classes : int
        Number of output classes.
    conv_layers : list of tuple, optional
        Convolutional layers defined as [(out_channels, kernel_size), ...].
    fc_layers : list of int, optional
        Fully connected layers defined as [units per layer].
    activation : callable, optional
        Activation function class (e.g., nn.ReLU).
    dropout_fc : float, optional
        Dropout probability for fully connected layers.
    dropout_conv : float, optional
        Dropout probability for convolutional layers.
    use_batchnorm : bool, optional
        Whether to include BatchNorm after each convolution.
    pool_type : str, optional
        Pooling type: "max" or "avg".
    global_pool : str, optional
        Global pooling type: "avg" or "max".

    Methods
    -------
    forward(x):
        Forward pass through the network.

    Notes
    -----
    - Convolutional layers are followed by activation, optional BatchNorm, optional dropout, and pooling.
    - Global pooling reduces spatial dimensions before fully connected layers.
    """

    def __init__(self, input_size, num_classes,
                 conv_layers=[(32, 3), (64, 3), (128, 3)],  
                 fc_layers=[256, 128],                     
                 activation=nn.ReLU,                       
                 dropout_fc=0.5,                           
                 dropout_conv=0.0,                         
                 use_batchnorm=True,
                 pool_type="max",                          
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
                layers.append(nn.Dropout2d(dropout_conv)) 
            layers.append(pool_layer)
            in_channels = out_channels

        layers.append(global_pool_layer) 
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
    """
    Docstrings made with Copilot and edited
    Train a classification model with validation, LR scheduling, early stopping, and optional Optuna pruning.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    num_epochs : int
        Maximum number of training epochs.
    train_dl : torch.utils.data.DataLoader
        Dataloader for training data.
    valid_dl : torch.utils.data.DataLoader
        Dataloader for validation data.
    loss_fn : callable
        Loss function (e.g., nn.CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : str, optional
        Device for training ('cpu' or 'cuda').
    verbose : bool, optional
        Unused here; progress is shown via tqdm bars.
    patience_val : int, optional
        Patience for ReduceLROnPlateau and early stopping (based on validation accuracy).
    trial : optuna.trial.Trial, optional
        Optuna trial for reporting and pruning.

    Returns
    -------
    dict
        Training history with keys: 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'.

    Notes
    -----
    - Supports batches as dicts with keys ('image'/'images', 'label'/'labels') or tuples (x, y).
    - Uses ReduceLROnPlateau on validation loss.
    - Early stops when validation accuracy does not improve for `patience_val` epochs.
    - Restores the best model weights (by validation accuracy) before returning.
    - Displays per-epoch progress with tqdm and prints final metrics per epoch
    """

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



def get_batch_size(n_images_train=None):
    """
    Docstrings made with Copilot and edited
    Determine an appropriate batch size based on GPU memory and dataset size.

    Parameters
    ----------
    n_images_train : int or None
        Number of training images. If None, only hardware-based defaults are used.

    Returns
    -------
    int
        Recommended batch size:
        - GPU > 12 GB → 128
        - GPU ≤ 12 GB → 64
        - No GPU → 32
        - Adjusted for small datasets (<20 → 16, <100 → 32).
    """

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
        if gpu_mem > 12:
            default_large = 128
        else:
            default_large = 64
    else:
        default_large = 32

    if n_images_train is None:
        return default_large
    elif n_images_train < 20:
        return 16
    elif n_images_train < 100:
        return 32
    else:
        return default_large