#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import inspect
import logging
log = logging.getLogger(__name__)


def save_simulation_npz(path, **data):
    np.savez(path, **data)
    log.info("Saving simulation run. FileName = %s", path) 


def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def fmt_attr(value):
    if hasattr(value, "shape"):
        return f"shape={value.shape}"
    return str(value)


def log_attrs(logger, obj, names, prefix=""):
    msg = " | ".join(f"{name}={fmt_attr(getattr(obj, name))}" for name in names)
    logger.info("%s%s", prefix, msg)
