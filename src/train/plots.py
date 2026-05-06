"""Visualization helpers for training history and diagnostics.

Purpose: keep plotting concerns separate from fit/eval execution.
Inputs/Outputs: matplotlib side effects; optional dict summaries.
Assumptions: classification labels are in {0,1,2} when confusion matrices are used.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_loss_curves(history, title="Training curves"):
    plt.figure(figsize=(6, 3))
    plt.plot(history.get("train_loss", []), label="train")
    plt.plot(history.get("val_loss", []), label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
