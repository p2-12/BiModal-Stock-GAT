"""Structured logging helpers.

Purpose: standardize run IDs, dataset versions, and prediction trace IDs.
"""

from __future__ import annotations

import logging
import uuid


def get_logger(name: str = "bimodal"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return logging.getLogger(name)


def new_trace_id() -> str:
    return str(uuid.uuid4())
