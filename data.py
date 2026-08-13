from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DoQDataset(Dataset):
    """Load one preprocessed DMWF split from an NPZ file."""

    def __init__(self, path: str | Path):
        path = Path(path)
        with np.load(path) as bundle:
            missing = {"X", "y", "c"}.difference(bundle.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            x = np.asarray(bundle["X"], dtype=np.float32)
            y = np.asarray(bundle["y"], dtype=np.float32)
            c = np.asarray(bundle["c"], dtype=np.int64)
        if x.ndim != 3 or x.shape[1] != 3:
            raise ValueError("X must have shape [N, 3, T]")
        if y.ndim != 2 or len(y) != len(x):
            raise ValueError("y must have shape [N, C]")
        if c.ndim != 1 or len(c) != len(x):
            raise ValueError("c must have shape [N]")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain finite values")
        if np.any(c < 0) or not np.array_equal(c, y.sum(axis=1).astype(np.int64)):
            raise ValueError("c must equal the number of positive labels in each row")
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.c = torch.from_numpy(c)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.c[index]

    @property
    def num_classes(self) -> int:
        return int(self.y.shape[1])

    @property
    def sequence_length(self) -> int:
        return int(self.x.shape[2])

