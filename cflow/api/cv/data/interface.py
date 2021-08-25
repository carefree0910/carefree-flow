from typing import Any
from typing import Dict
from typing import Tuple
from typing import Optional

from .core import TensorDataset
from ....types import arr_type
from ....types import arrays_type
from ....types import sample_weights_type
from ....misc.internal_ import CVDataset
from ....misc.internal_ import CVLoader
from ....misc.internal_ import DataLoader
from ....misc.internal_ import DLDataModule


@DLDataModule.register("tensor")
class TensorData(DLDataModule):
    def __init__(
        self,
        x_train: arr_type,
        y_train: Optional[arr_type] = None,
        x_valid: Optional[arr_type] = None,
        y_valid: Optional[arr_type] = None,
        train_others: Optional[arrays_type] = None,
        valid_others: Optional[arrays_type] = None,
        *,
        shuffle: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.train_others = train_others
        self.valid_others = valid_others
        self.kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    @property
    def info(self) -> Dict[str, Any]:
        return self.kw

    # TODO : support sample weights
    def prepare(self, sample_weights: sample_weights_type) -> None:
        def _get_data(x: Any, y: Any, others: Any) -> CVDataset:
            return CVDataset(TensorDataset(x, y, others))

        self.train_data = _get_data(self.x_train, self.y_train, self.train_others)
        if self.x_valid is None:
            self.valid_data = None
        else:
            self.valid_data = _get_data(self.x_valid, self.y_valid, self.valid_others)

    def initialize(self) -> Tuple[CVLoader, Optional[CVLoader]]:
        train_loader = CVLoader(DataLoader(self.train_data, **self.kw))  # type: ignore
        if self.valid_data is None:
            valid_loader = None
        else:
            valid_loader = CVLoader(DataLoader(self.valid_data, **self.kw))  # type: ignore
        return train_loader, valid_loader


__all__ = [
    "TensorData",
]
