#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path helpers for DCT's packaged data and metric compatibility runtime."""

import os
import sys


COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(COMMON_DIR))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, "survot_rank", "research")


def _find_compat_runtime_dir():
    """Find the local dataset/loss compatibility runtime required by DCT."""
    env = os.environ.get("SURVOT_RUNTIME_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)

    candidate = os.path.join(RESEARCH_DIR, "legacy", "slotspe_runtime")
    has_dataset = os.path.isfile(os.path.join(candidate, "dataset", "dataset_survival.py"))
    has_losses = os.path.isfile(os.path.join(candidate, "utils", "loss_func.py"))
    return os.path.abspath(candidate) if has_dataset and has_losses else None


COMPAT_RUNTIME_DIR = _find_compat_runtime_dir()


def ensure_compat_runtime_in_path():
    """Add the packaged DCT compatibility runtime to ``sys.path``."""
    if COMPAT_RUNTIME_DIR is None:
        raise FileNotFoundError(
            "Could not find the packaged DCT compatibility runtime. Set "
            "SURVOT_RUNTIME_DIR or restore its dataset and loss modules."
        )
    if COMPAT_RUNTIME_DIR not in sys.path:
        sys.path.insert(0, COMPAT_RUNTIME_DIR)


def get_compat_runtime_dir():
    """Return the packaged DCT compatibility runtime directory, if present."""
    return COMPAT_RUNTIME_DIR
