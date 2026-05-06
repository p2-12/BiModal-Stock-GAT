"""Compatibility exports for training utilities.

Purpose: maintain existing imports while internals are decomposed.
"""

from .evaluator import collect_batch_outputs as _collect_batch_outputs
from .plots import plot_loss_curves
from .trainer import TrainResult, make_time_splits, pick_device, set_seed, train_node_model
from .uncertainty import mc_dropout_predictions

__all__ = [
    "TrainResult",
    "set_seed",
    "pick_device",
    "make_time_splits",
    "train_node_model",
    "_collect_batch_outputs",
    "plot_loss_curves",
    "mc_dropout_predictions",
]
