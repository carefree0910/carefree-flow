import numpy as np

from typing import Optional
from oneflow import Tensor
from oneflow.utils.data import Dataset

from ....types import arr_type
from ....types import arrays_type
from ....misc.toolkit import to_flow


class TensorDataset(Dataset):
    def __init__(
        self,
        x: arr_type,
        y: Optional[arr_type],
        others: Optional[arrays_type] = None,
    ):
        if isinstance(x, np.ndarray):
            x = to_flow(x)
        if isinstance(y, np.ndarray):
            y = to_flow(y)
        if others is not None:
            others = [v if isinstance(v, Tensor) else to_flow(v) for v in others]
        self.x = x
        self.y = y
        self.others = others

    def __getitem__(self, index: int) -> arrays_type:
        label = 0 if self.y is None else self.y[index]
        item = [self.x[index], label]
        if self.others is not None:
            item.extend(v[index] for v in self.others)
        return tuple(item)

    def __len__(self) -> int:
        return self.x.shape[0]
