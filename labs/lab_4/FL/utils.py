import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------
# Device selection utility
# ------------------------------------------------------------
def get_device():
    """
    Return the available torch device.
    Prioritizes GPU (cuda) if available; otherwise falls back to CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Binary prediction and evaluation helper
# ------------------------------------------------------------
def prediction_binary(model, loader, loss_fn, device):
    """
    Evaluate a binary classification model on a given DataLoader.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (DataLoader): DataLoader providing evaluation data.
        loss_fn: Loss function (e.g., nn.BCELoss).
        device (torch.device): Computation device.

    Returns:
        val_loss (float): Average validation loss.
        auc (float): Area Under the ROC Curve (AUC).
        apr (float): Area Under the Precision-Recall Curve (APR).
    """
    model.to(device)
    model.eval()

    preds, labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)

            output = model(data)[:, 0]  # assume output shape: (batch_size, 1)
            loss = loss_fn(output, target)
            val_loss += loss.item()

            preds.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())

    val_loss /= len(loader)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    auc = roc_auc_score(labels, preds)
    precision, recall, _ = metrics.precision_recall_curve(labels, preds)
    apr = metrics.auc(recall, precision)

    return val_loss, auc, apr


# ------------------------------------------------------------
# Validation evaluation for each client
# ------------------------------------------------------------
def evaluate_models(client_id, loaders, model, loss_fn, device, df, best_auc, train_loss, path):
    """
    Evaluate a client's model on its validation data.
    If the new validation AUC exceeds the best so far, save the model.

    Args:
        client_id (int): ID of the client.
        loaders (list): List of [train_loader, val_loader, test_loader] for each client.
        model (torch.nn.Module): Trained model to evaluate.
        loss_fn: Loss function used.
        device (torch.device): Computation device.
        df (pd.DataFrame): DataFrame to append validation results.
        best_auc (list): List tracking best AUC per client.
        train_loss (float): Training loss for the current round.
        path (str): Directory to save best-performing models.

    Returns:
        df (pd.DataFrame): Updated results DataFrame.
        best_auc[client_id] (float): Updated best AUC for this client.
        val_auc (float): Current validation AUC.
        val_apr (float): Current validation APR.
    """
    val_loss, val_auc, val_apr = prediction_binary(model, loaders[client_id][1], loss_fn, device)

    # Save model if it achieves a new best AUC
    if val_auc > best_auc[client_id]:
        best_auc[client_id] = val_auc
        torch.save(model, f'./trained_models/{path}/node{client_id}')

    # Log metrics
    new_entry = {
        "Train_Loss": train_loss,
        "Val_Loss": val_loss,
        "Val_AUC": val_auc,
        "Val_APR": val_apr
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    return df, best_auc[client_id], val_auc, val_apr


# ------------------------------------------------------------
# Test evaluation for each client
# ------------------------------------------------------------
def evaluate_models_test(client_id, loaders, model, loss_fn, device):
    """
    Evaluate a model on a client's test data.

    Args:
        client_id (int): ID of the client.
        loaders (list): List of [train_loader, val_loader, test_loader] for each client.
        model (torch.nn.Module): Model to evaluate.
        loss_fn: Loss function.
        device (torch.device): Computation device.

    Returns:
        val_loss (float): Test loss.
        val_auc (float): Test AUC.
        val_apr (float): Test APR.
    """
    val_loss, val_auc, val_apr = prediction_binary(model, loaders[client_id][2], loss_fn, device)
    return val_loss, val_auc, val_apr
