import numpy as np
import oneflow as flow

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Optional


arr_type = Union[np.ndarray, flow.Tensor]
data_type = Optional[Union[np.ndarray, str]]
param_type = Union[flow.Tensor, flow.nn.Parameter]
general_config_type = Optional[Union[str, Dict[str, Any]]]
np_arrays_type = Tuple[Union[np.ndarray, Any], ...]
item_type = Optional[Union[flow.Tensor, Any]]
arrays_type = Union[flow.Tensor, Tuple[item_type, ...]]
states_callback_type = Optional[Callable[[Any, Dict[str, Any]], Dict[str, Any]]]
sample_weights_type = Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]


__all__ = [
    "arr_type",
    "data_type",
    "param_type",
    "general_config_type",
    "np_arrays_type",
    "item_type",
    "arrays_type",
    "states_callback_type",
    "sample_weights_type",
]
