import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from src.qml_model import predict_proba_qml


def compute_metrics(y_true: np.ndarray, probs: np.ndarray):
    preds = np.argmax(probs, axis=1)
    acc_value = accuracy_score(y_true, preds)
    f1_value = f1_score(y_true, preds, average='macro')
    onehot = np.eye(probs.shape[1], dtype=float)[y_true]
    mse_value = mean_squared_error(onehot, probs)

    return acc_value, f1_value, mse_value


def evaluate_torch_loss(model, x_eval, y_eval, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        x_tensor = torch.tensor(x_eval, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_eval, dtype=torch.long, device=device)
        loss = loss_fn(model(x_tensor), y_tensor)

    return float(loss.item())


def evaluate_qml_log_loss(model, x_eval, y_eval, device):
    probs = predict_proba_qml(model, x_eval, device)

    return float(log_loss(y_eval, probs, labels=np.arange(probs.shape[1])))


def train_model(
    model,
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    epochs=10,
    batch_size=64,
    lr=1e-3,
    device=None,
    trainable_params=None,
    freeze_params=None,
    label_prefix='Adam',
    use_validation=False
):
    if device is None:
        device = next(model.parameters()).device

    if use_validation and (x_val is None or y_val is None):
        raise ValueError('x_val and y_val are required when use_validation=True')

    if freeze_params is not None:
        for param in freeze_params:
            param.requires_grad = False

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True
    )

    if trainable_params is None:
        trainable_params = model.parameters()

    trainable_params = list(trainable_params)
    opt = torch.optim.Adam(trainable_params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

            running += float(loss.item())

        train_loss = running / max(len(train_loader), 1)

        if use_validation:
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(x_val, dtype=torch.float32, device=device)
                yv = torch.tensor(y_val, dtype=torch.long, device=device)
                val_loss = float(loss_fn(model(xv), yv).item())

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f'{label_prefix} Epoch {epoch + 1:02d} | '
                    f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}'
                )
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f'{label_prefix} Epoch {epoch + 1:02d} | '
                    f'Train Loss: {train_loss:.4f}'
                )

    if use_validation and best_state is not None:
        model.load_state_dict(best_state)

    if freeze_params is not None:
        for param in freeze_params:
            param.requires_grad = True

    return model