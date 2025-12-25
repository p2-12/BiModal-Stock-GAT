from __future__ import annotations

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


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_time_splits(n, train_frac=0.7, val_frac=0.15):
    n_train=int(n*train_frac)
    n_val=int(n*val_frac)
    idx=np.arange(n)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


def train_node_model(model, dataset, task, batch_size, epochs, lr, weight_decay, patience, label_mode="std", threshold_k=0.5, device=None, splits=None, print_every=1):
    """
    Train a per-node model on graph snapshots.

    - Time splits by default.
    - For classification, thresholds are computed using TRAIN ONLY.
    - Prints train/val loss during training if print_every>0.
    """
    if device is None:
        device=pick_device()

    n=len(dataset)
    if splits is None:
        tr_idx, va_idx, te_idx=make_time_splits(n)
    else:
        tr_idx, va_idx, te_idx=splits

    # thresholds from TRAIN ONLY (flatten across snapshots x stocks)
    lo=hi=None
    if task=="classify":
        train_future=np.stack([dataset[i].y.detach().cpu().numpy() for i in tr_idx], axis=0)
        lo, hi=thresholds_from_train(train_future, mode=label_mode, k=threshold_k)

    def transform_item(data):
        # Always preserve regression target for downstream eval/plots
        if not hasattr(data, "y_reg"):
            data.y_reg=data.y.clone()

        if task=="classify":
            y=data.y_reg.detach().cpu().numpy()
            if y.ndim==2:
                y=y[:, -1]
            data.y=torch.tensor(to_3class_labels(y, lo, hi), dtype=torch.long)

        return data

    tr_ds=[transform_item(dataset[i]) for i in tr_idx]
    va_ds=[transform_item(dataset[i]) for i in va_idx]
    te_ds=[transform_item(dataset[i]) for i in te_idx]

    tr_loader=DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader=DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    te_loader=DataLoader(te_ds, batch_size=batch_size, shuffle=False)

    model=model.to(device)

    if task=="classify":
        criterion=nn.CrossEntropyLoss()
    elif task=="regress":
        criterion=nn.MSELoss()
    else:
        raise ValueError("task must be 'classify' or 'regress'")

    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val=float("inf")
    best_state=None
    bad=0

    history={"train_loss":[], "val_loss":[]}

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss=0.0
        nb=0

        for batch in tr_loader:
            batch=batch.to(device)
            opt.zero_grad()

            out=model(batch.price, batch.text, batch.text_mask, batch.edge_index, batch.edge_attr)

            if task=="classify":
                loss=criterion(out, batch.y.view(-1))
            else:
                loss=criterion(out, batch.y_reg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss+=loss.item()
            nb+=1

        tr_loss/=max(1, nb)

        model.eval()
        va_loss=0.0
        nb=0
        with torch.no_grad():
            for batch in va_loader:
                batch=batch.to(device)
                out=model(batch.price, batch.text, batch.text_mask, batch.edge_index, batch.edge_attr)

                if task=="classify":
                    loss=criterion(out, batch.y.view(-1))
                else:
                    loss=criterion(out, batch.y_reg)

                va_loss+=loss.item()
                nb+=1

        va_loss/=max(1, nb)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if print_every and (epoch%print_every==0 or epoch==1 or epoch==epochs):
            print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

        if va_loss<best_val-1e-5:
            best_val=va_loss
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad=0
        else:
            bad+=1

        if bad>=patience:
            if print_every:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    return TrainResult(best_state_dict=best_state, history=history), (tr_idx, va_idx, te_idx), (lo, hi), te_loader


# ------------------------------------------------------------
# Plotting / evaluation helpers
# ------------------------------------------------------------

def plot_loss_curves(history, title="Training curves"):
    import matplotlib.pyplot as plt
    tr=history.get("train_loss", [])
    va=history.get("val_loss", [])
    plt.figure(figsize=(6,3))
    plt.plot(tr, label="train")
    plt.plot(va, label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _collect_batch_outputs(model, loader, device, task):
    ys=[]
    yregs=[]
    preds=[]
    probs=[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(device)
            out=model(batch.price, batch.text, batch.text_mask, batch.edge_index, batch.edge_attr)

            if hasattr(batch, "y"):
                ys.append(batch.y.view(-1).detach().cpu().numpy())
            if hasattr(batch, "y_reg"):
                yregs.append(batch.y_reg.detach().cpu().numpy())

            if task=="classify":
                p=torch.softmax(out, dim=-1)
                probs.append(p.detach().cpu().numpy())
                preds.append(out.argmax(dim=-1).detach().cpu().numpy())
            else:
                preds.append(out.detach().cpu().numpy())

    ys=np.concatenate(ys) if len(ys) else None
    yregs=np.concatenate(yregs) if len(yregs) else None
    preds=np.concatenate(preds) if len(preds) else None
    probs=np.concatenate(probs) if len(probs) else None
    return ys, yregs, preds, probs


def plot_confusion_matrix(model, loader, device=None, class_names=None, normalize=None, title="Confusion matrix"):
    """
    normalize: None | "true" | "pred" | "all"
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    if device is None:
        device=pick_device()

    y, _, pred, _=_collect_batch_outputs(model, loader, device, task="classify")
    cm=confusion_matrix(y, pred, normalize=normalize)
    disp=ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax=plt.subplots(figsize=(5,4))
    disp.plot(ax=ax, cmap=None, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def mc_dropout_predictions(model, data, device, n_samples=50):
    """
    MC Dropout: run multiple stochastic forward passes with dropout enabled to approximate uncertainty.

    Returns:
      samples: [n_samples, S] if model outputs [S]
               [n_samples, S, H] if model outputs [S,H]
    """
    model=model.to(device)
    model.train()  # enable dropout
    samples=[]

    with torch.no_grad():
        for _ in range(n_samples):
            out=model(data.price, data.text, data.text_mask, data.edge_index, data.edge_attr)
            samples.append(out.detach().cpu().numpy())

    return np.stack(samples, axis=0)


def plot_regression_forecast_with_uncertainty_from_loader(model, loader, batch_i, stock_i, device=None, n_samples=80, title=None):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    if device is None:
        device=pick_device()

    target=None
    for i, batch in enumerate(loader):
        if i==batch_i:
            target=batch
            break
    if target is None:
        raise ValueError(f"batch_i={batch_i} out of range for loader")

    target=target.to(device)

    # Multi-horizon regression: delegate to the curve plotter
    if hasattr(target, "y_reg") and target.y_reg.dim()==2 and target.y_reg.shape[1]>1:
        return plot_multihorizon_forecast_from_loader(model, loader, batch_i, stock_i, device=device, n_samples=n_samples, title=title)

    # Scalar regression target (either y_reg[:,0] or y)
    if hasattr(target, "y_reg"):
        y_true=float(target.y_reg.view(-1)[stock_i].detach().cpu().item())
    else:
        y_true=float(target.y.view(-1)[stock_i].detach().cpu().item())

    model=model.to(device)
    model.train()  # enable dropout

    samples=[]
    with torch.no_grad():
        for _ in range(n_samples):
            out=model(target.price, target.text, target.text_mask, target.edge_index, target.edge_attr)
            out=out.detach().cpu().numpy()
            if out.ndim==2 and out.shape[1]==1:
                out=out[:, 0]
            samples.append(out)

    samples=np.stack(samples, axis=0)  # [K, N_nodes_in_batch]
    mu=samples.mean(axis=0)
    sd=samples.std(axis=0)+1e-12

    pred_mu=float(mu[stock_i])
    pred_lo=float(pred_mu-1.96*sd[stock_i])
    pred_hi=float(pred_mu+1.96*sd[stock_i])

    plt.figure(figsize=(6,3))
    plt.errorbar([0], [pred_mu], yerr=[[pred_mu-pred_lo],[pred_hi-pred_mu]], fmt="o", capsize=6, label="pred mean ± 1.96σ")
    plt.scatter([0], [y_true], marker="x", s=80, label="true")
    plt.xticks([0], [f"test batch {batch_i}, node {stock_i}"])
    plt.ylabel("Future return")
    plt.title(title or "Test-set regression forecast w/ MC-dropout uncertainty")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    model.eval()
    return {"y_true":y_true, "pred_mean":pred_mu, "pred_lo":pred_lo, "pred_hi":pred_hi, "batch_i":batch_i, "stock_i":stock_i}



def plot_multihorizon_forecast_from_loader(model, loader, batch_i, stock_i, device=None, n_samples=80, title=None):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    if device is None:
        device = pick_device()

    target = None
    for i, batch in enumerate(loader):
        if i == batch_i:
            target = batch
            break
    if target is None:
        raise ValueError(f"batch_i={batch_i} out of range")

    target = target.to(device)

    if hasattr(target, "y_reg"):
        y_true = target.y_reg[stock_i].detach().cpu().numpy()  # [H]
    # If this is a multi-horizon regression target, delegate to the curve plotter
    if hasattr(target, "y_reg"):
        y_tmp=target.y_reg[stock_i].detach().cpu().numpy()
        if y_tmp.ndim==1 and y_tmp.shape[0]>1:
            return plot_multihorizon_forecast_from_loader(model, loader, batch_i, stock_i, device=device, n_samples=n_samples, title=title)

    else:
        y_true = target.y[stock_i].detach().cpu().numpy()      # [H]

    model = model.to(device)
    model.train()  # enable dropout

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(target.price, target.text, target.text_mask, target.edge_index, target.edge_attr)
            samples.append(out.detach().cpu().numpy())  # [N, H]

    samples = np.stack(samples, axis=0)  # [K, N, H]
    s = samples[:, stock_i, :]           # [K, H]
    mu = s.mean(axis=0)
    sd = s.std(axis=0) + 1e-12

    lo = mu - 1.96 * sd
    hi = mu + 1.96 * sd

    x = np.arange(1, mu.shape[0] + 1)

    plt.figure(figsize=(7, 3.5))
    plt.plot(x, mu, marker="o", label="pred mean")
    plt.fill_between(x, lo, hi, alpha=0.25, label="±1.96σ")
    plt.plot(x, y_true, marker="x", linestyle="--", label="true")
    plt.xlabel("Days ahead")
    plt.ylabel("Cumulative log return")
    plt.title(title or f"Test-set {mu.shape[0]}-day forecast (batch {batch_i}, node {stock_i})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.eval()
    return {"batch_i": batch_i, "stock_i": stock_i, "pred_mean": mu, "pred_lo": lo, "pred_hi": hi, "y_true": y_true}

def plot_curve_uncertainty_from_test_loader(model, loader, snapshot_i, stock_i, n_samples=30, device="cpu"):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # grab the snapshot_i-th test graph (batch_size=1 loader)
    it = iter(loader)
    batch = None
    for _ in range(snapshot_i + 1):
        batch = next(it)
    g = batch.to(device)

    # true curve: [H]
    y_true = g.y_reg[stock_i].detach().cpu().numpy()

    model = model.to(device)
    model.train()  # MC-dropout ON

    # streaming mean/var (Welford)
    mean = None
    m2 = None
    n = 0

    with torch.no_grad():
        for _ in range(n_samples):
            out = model(g.price, g.text, g.text_mask, g.edge_index, g.edge_attr)  # [S, H]
            x = out[stock_i].detach().cpu().numpy()  # [H]

            n += 1
            if mean is None:
                mean = x.copy()
                m2 = np.zeros_like(x)
            else:
                delta = x - mean
                mean = mean + delta / n
                m2 = m2 + delta * (x - mean)

            if device == "mps":
                torch.mps.empty_cache()

    var = m2 / max(1, (n - 1))
    sd = np.sqrt(var + 1e-12)

    lo = mean - 1.96 * sd
    hi = mean + 1.96 * sd

    x_axis = np.arange(1, len(mean) + 1)

    plt.figure(figsize=(7, 3.5))
    plt.plot(x_axis, mean, marker="o", label="pred mean")
    plt.fill_between(x_axis, lo, hi, alpha=0.25, label="±1.96σ")
    plt.plot(x_axis, y_true, marker="x", linestyle="--", label="true")
    plt.xlabel("Days ahead")
    plt.ylabel("Cumulative log return")
    plt.title(f"Test forecast (snapshot {snapshot_i}, node {stock_i})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.eval()
    return {"pred_mean": mean, "pred_lo": lo, "pred_hi": hi, "y_true": y_true}
