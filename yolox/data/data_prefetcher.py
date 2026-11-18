#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from collections import deque

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.

    ``prefetch_depth`` controls how many batches are kept in flight ahead of the
    one being consumed. depth=1 (default) reproduces the original single-batch
    behaviour. depth=2 or 3 keeps more batches in flight so an occasional slow
    worker cannot stall the GPU (hides data_time spikes).
    """

    def __init__(self, loader, prefetch_depth=1):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.depth = max(1, int(prefetch_depth))
        self._queue = deque()
        for _ in range(self.depth):
            self._preload_one()

    def _preload_one(self):
        try:
            next_input, next_target, _, _ = next(self.loader)
        except StopIteration:
            self._queue.append((None, None))
            return
        with torch.cuda.stream(self.stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
        self._queue.append((next_input, next_target))

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input, target = self._queue.popleft()
        self._preload_one()
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        return input, target
