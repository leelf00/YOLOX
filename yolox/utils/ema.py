#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import math
from copy import deepcopy

import torch
import torch.nn as nn

__all__ = ["ModelEMA", "is_parallel"]


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        # cache (ema, model) tensor pairs once so update() skips state_dict()
        ema_mod = self.ema.module if is_parallel(self.ema) else self.ema
        mod = model.module if is_parallel(model) else model
        ema_sd = ema_mod.state_dict()
        mod_sd = mod.state_dict()
        self._pairs = [
            (ema_sd[k], mod_sd[k])
            for k in ema_sd
            if ema_sd[k].dtype.is_floating_point
        ]

    def update(self, model=None):
        # Update EMA parameters in-place using cached tensor references
        # (equivalent to the old state_dict() loop, but without rebuilding
        # the state_dict every iteration). `model` is accepted for backward
        # compatibility; the references captured at init are used instead.
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            om = 1.0 - d
            for ev, mv in self._pairs:
                ev.mul_(d).add_(mv, alpha=om)
