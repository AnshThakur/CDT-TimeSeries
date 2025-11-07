import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# ------------------------------------------------------------
# Data loader for time-series dataset (not dataset-specific)
# ------------------------------------------------------------
def get_loaders_time_series(
    path,
    train_batch=64,
    val_batch=64,
    test_batch=64,
    sampler=True,
    pre_process="minmax",
    ds_half=0,
):
    """
    Create train/val/test DataLoaders for a time-series dataset stored as pickled files.

    Expected files under `path`:
      - train_raw.p  -> stored as (X_train, y_train)
      - val_raw.p    -> stored as (X_val, y_val)
      - test_raw.p   -> stored as {"data": (X_test, y_test)}

    Notes:
      - This function does NOT drop or reorder feature dimensions; it preserves the input shape.
      - `pre_process` controls normalization: "minmax", "std", or "none".
      - If `ds_half` is 1 or 2, uses first or second half of the training set respectively (useful for experiments).
      - If `sampler=True`, a weighted sampler is used to balance classes in training.
      - The function returns only (train_loader, val_loader, test_loader).

    Returns:
      train_loader, val_loader, test_loader
    """

    # ---------- Load training data ----------
    with open(os.path.join(path, "train_raw.p"), "rb") as f:
        train_obj = pickle.load(f)

    train_X, train_y = train_obj[0], np.array(train_obj[1])

    # ---------- Optionally use half of the training data ----------
    assert ds_half in (0, 1, 2)
    if ds_half:
        mid = len(train_X) // 2
        if ds_half == 1:
            print("Using first half as the training set")
            train_X, train_y = train_X[:mid], train_y[:mid]
        else:
            print("Using second half as the training set")
            train_X, train_y = train_X[mid:], train_y[mid:]

    # ---------- Normalization (fit on training data) ----------
    normalizer = DataNormalization(train_X, pre_process=pre_process)
    train_X = normalizer(train_X)

    # Build training dataset
    train_dataset = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))

    # ---------- Optional weighted sampler to address class imbalance ----------
    if sampler:
        unique_classes, counts = np.unique(train_y, return_counts=True)
        class_sample_count = dict(zip(unique_classes, counts))
        # weights: inverse frequency per class
        class_weights = {c: 1.0 / max(1, class_sample_count[c]) for c in unique_classes}
        sample_weights = np.array([class_weights[int(lbl)] for lbl in train_y], dtype=np.float64)
        sample_weights = torch.from_numpy(sample_weights)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights.type(torch.DoubleTensor), len(sample_weights)
        )
        train_loader = DataLoader(train_dataset, batch_size=train_batch, sampler=weighted_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)

    # ---------- Validation data ----------
    with open(os.path.join(path, "val_raw.p"), "rb") as f:
        val_obj = pickle.load(f)

    val_X, val_y = val_obj[0], np.array(val_obj[1])
    val_X = normalizer(val_X)  # apply same normalization fitted on train
    val_dataset = TensorDataset(torch.FloatTensor(val_X), torch.LongTensor(val_y))
    val_loader = DataLoader(val_dataset, batch_size=val_batch)

    # ---------- Test data ----------
    with open(os.path.join(path, "test_raw.p"), "rb") as f:
        test_obj = pickle.load(f)

    # Expecting structure: {"data": (X_test, y_test)} or similar
    test_X = None
    test_y = None
    if isinstance(test_obj, dict) and "data" in test_obj:
        test_X = test_obj["data"][0]
        test_y = np.array(test_obj["data"][1])
    else:
        # fallback to tuple-like structure (X, y)
        try:
            test_X, test_y = test_obj[0], np.array(test_obj[1])
        except Exception as e:
            raise RuntimeError("Unable to parse test_raw.p format") from e

    test_X = normalizer(test_X)
    test_dataset = TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))
    test_loader = DataLoader(test_dataset, batch_size=test_batch)

    # ---------- Return only the loaders as requested ----------
    return train_loader, val_loader, test_loader


# ------------------------------------------------------------
# DataNormalization helper (preserves feature dimensions)
# ------------------------------------------------------------
class DataNormalization:
    """
    Simple normalization helper for time-series arrays.

    Accepts:
      - X_train: either a numpy array (N x T x F) or a list of arrays [T_i x F].
      - pre_process: "none", "std" (z-score), or "minmax".

    The computed statistics (mean/std/min/max) are stored and reused for val/test.
    """

    def __init__(self, X_train, pre_process="std", eps=1e-9, prpr_ax=(0, 1)):
        self.eps = eps
        assert pre_process in ("none", "std", "minmax"), "pre_process must be one of: none/std/minmax"
        self.pre_process = pre_process

        if self.pre_process == "none":
            # Nothing to compute
            self.mean = None
            self.std = None
            self.min = None
            self.max = None
            self.dif = None
            return

        # Determine data source and axis to compute statistics
        if isinstance(X_train, list) and len(X_train) > 0 and X_train[0].ndim == 2:
            # list of (T_i x F) arrays: concatenate across time dimension
            X_source = np.concatenate(X_train, axis=0)  # shape (sum T_i, F)
            axis = 0
        elif isinstance(X_train, np.ndarray) and X_train.ndim == 3:
            # array shape: (N, T, F)
            X_source = X_train
            axis = prpr_ax
        else:
            raise NotImplementedError("Normalization supports list of T_i x F arrays or ndarray N x T x F")

        # compute statistics along specified axes (keeps feature dimension shape)
        self.mean = np.mean(X_source, axis=axis, keepdims=True)
        self.std = np.std(X_source, axis=axis, keepdims=True) + self.eps
        self.max = np.max(X_source, axis=axis, keepdims=True)
        self.min = np.min(X_source, axis=axis, keepdims=True)
        self.dif = (self.max - self.min) + self.eps

    def norm_std(self, data):
        if isinstance(data, list):
            return [(item - self.mean) / self.std for item in data]
        else:
            return (data - self.mean) / self.std

    def norm_minmax(self, data):
        if isinstance(data, list):
            return [(item - self.min) / self.dif for item in data]
        else:
            return (data - self.min) / self.dif

    def recover_minmax(self, data):
        if isinstance(data, list):
            return [(item * self.dif) + self.min for item in data]
        else:
            return (data * self.dif) + self.min

    def recover_std(self, data):
        if isinstance(data, list):
            return [(item * self.std) + self.mean for item in data]
        else:
            return (data * self.std) + self.mean

    def __call__(self, data):
        if self.pre_process == "none":
            return data
        if self.pre_process == "minmax":
            return self.norm_minmax(data)
        elif self.pre_process == "std":
            return self.norm_std(data)
        else:
            raise NotImplementedError(f"Unknown pre_process: {self.pre_process}")

    def recover(self, data):
        if self.pre_process == "none":
            return data
        if self.pre_process == "minmax":
            return self.recover_minmax(data)
        elif self.pre_process == "std":
            return self.recover_std(data)
        else:
            raise NotImplementedError(f"Unknown pre_process: {self.pre_process}")
