"""Training service helpers for node-level graph models.

Purpose:
    Encapsulate fit-loop logic, early stopping, and dataset split handling.
Inputs/Outputs:
    Input is a torch model plus graph snapshot dataset; output is TrainResult,
    split indices, classification thresholds, and the test loader.
Assumptions:
    Regression targets come from graph dataset tensors where the last horizon is
    commonly used for class labeling; feature index conventions are defined in
    ``src/data/graph_dataset.py`` (e.g., log-return at feature position 0).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .labels import thresholds_from_train, to_3class_labels


@dataclass
class TrainResult:
    best_state_dict: dict
    history: dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_time_splits(n: int, train_frac: float = 0.7, val_frac: float = 0.15):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    idx = np.arange(n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def train_node_model(
    model,
    dataset,
    task,
    batch_size,
    epochs,
    lr,
    weight_decay,
    patience,
    label_mode="std",
    threshold_k=0.5,
    device=None,
    splits=None,
    print_every=1,
    logger=None,
):
    device = device or pick_device()
    tr_idx, va_idx, te_idx = splits if splits is not None else make_time_splits(len(dataset))
    lo = hi = None
    if task == "classify":
        train_future = np.stack([dataset[i].y.detach().cpu().numpy() for i in tr_idx], axis=0)
        lo, hi = thresholds_from_train(train_future, mode=label_mode, k=threshold_k)

    def transform_item(data):
        if not hasattr(data, "y_reg"):
            data.y_reg = data.y.clone()
        if task == "classify":
            y = data.y_reg.detach().cpu().numpy()
            y = y[:, -1] if y.ndim == 2 else y
            data.y = torch.tensor(to_3class_labels(y, lo, hi), dtype=torch.long)
        return data

    tr_loader = DataLoader(
        [transform_item(dataset[i]) for i in tr_idx], batch_size=batch_size, shuffle=True
    )
    va_loader = DataLoader(
        [transform_item(dataset[i]) for i in va_idx], batch_size=batch_size, shuffle=False
    )
    te_loader = DataLoader(
        [transform_item(dataset[i]) for i in te_idx], batch_size=batch_size, shuffle=False
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() if task == "classify" else nn.MSELoss()
    if task not in {"classify", "regress"}:
        raise ValueError("task must be 'classify' or 'regress'")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = float("inf")
    best_state = None
    bad = 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        nb = 0
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.price, batch.text, batch.text_mask, batch.edge_index, batch.edge_attr)
            loss = (
                criterion(out, batch.y.view(-1))
                if task == "classify"
                else criterion(out, batch.y_reg)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
            nb += 1
        tr_loss /= max(1, nb)
        model.eval()
        va_loss = 0.0
        nb = 0
        with torch.no_grad():
            for batch in va_loader:
                batch = batch.to(device)
                out = model(
                    batch.price, batch.text, batch.text_mask, batch.edge_index, batch.edge_attr
                )
                loss = (
                    criterion(out, batch.y.view(-1))
                    if task == "classify"
                    else criterion(out, batch.y_reg)
                )
                va_loss += loss.item()
                nb += 1
        va_loss /= max(1, nb)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        if print_every and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
            (logger.info if logger else print)(
                f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}"
            )
        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            if print_every:
                (logger.info if logger else print)(
                    f"Early stopping at epoch {epoch} (patience={patience})"
                )
            break
    return (
        TrainResult(best_state_dict=best_state, history=history),
        (tr_idx, va_idx, te_idx),
        (lo, hi),
        te_loader,
    )
