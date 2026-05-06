"""MC-dropout utilities for predictive uncertainty.

Purpose: run stochastic forward passes and summarize confidence bands.
Inputs/Outputs: graph batch tensors in model input format; numpy samples/stats out.
Assumptions: model contains active dropout layers at train-mode inference.
"""

from __future__ import annotations

import numpy as np
import torch


def mc_dropout_predictions(model, data, device, n_samples=50):
    model = model.to(device)
    model.train()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(
                model(data.price, data.text, data.text_mask, data.edge_index, data.edge_attr)
                .detach()
                .cpu()
                .numpy()
            )
    return np.stack(samples, axis=0)
