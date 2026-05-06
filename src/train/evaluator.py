"""Evaluation service helpers for collecting model outputs and metrics.

Purpose: centralize test-time forward passes for classification/regression.
Inputs/Outputs: receives model, dataloader, and task; returns numpy arrays.
Assumptions: loaders carry ``price/text/text_mask/edge_*`` tensors.
"""

from __future__ import annotations

import numpy as np
import torch


def collect_batch_outputs(model, loader, device, task):
    ys = []
    yregs = []
    preds = []
    probs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.price, batch.text, batch.text_mask, batch.edge_index, batch.edge_attr)
            if hasattr(batch, "y"):
                ys.append(batch.y.view(-1).detach().cpu().numpy())
            if hasattr(batch, "y_reg"):
                yregs.append(batch.y_reg.detach().cpu().numpy())
            if task == "classify":
                p = torch.softmax(out, dim=-1)
                probs.append(p.detach().cpu().numpy())
                preds.append(out.argmax(dim=-1).detach().cpu().numpy())
            else:
                preds.append(out.detach().cpu().numpy())
    return (
        np.concatenate(ys) if ys else None,
        np.concatenate(yregs) if yregs else None,
        np.concatenate(preds) if preds else None,
        np.concatenate(probs) if probs else None,
    )
