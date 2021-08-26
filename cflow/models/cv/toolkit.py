import math

import oneflow as flow

from typing import Union
from typing import Optional


def get_latent_resolution(img_size: int, num_layer: int) -> int:
    return int(round(img_size / 2 ** num_layer))


def auto_num_layers(img_size: int, min_size: int = 4, target_layers: int = 4) -> int:
    max_layers = int(round(math.log2(img_size / min_size)))
    return max(2, min(target_layers, max_layers))


def slerp(
    x1: flow.Tensor,
    x2: flow.Tensor,
    r1: Union[float, flow.Tensor],
    r2: Optional[Union[float, flow.Tensor]] = None,
) -> flow.Tensor:
    low_norm = x1 / flow.norm(x1, dim=1, keepdim=True)
    high_norm = x2 / flow.norm(x2, dim=1, keepdim=True)
    omega = flow.acos((low_norm * high_norm).sum(1))
    so = flow.sin(omega)
    if r2 is None:
        r2 = 1.0 - r1
    x1_part = (flow.sin(r1 * omega) / so).unsqueeze(1) * x1
    x2_part = (flow.sin(r2 * omega) / so).unsqueeze(1) * x2
    return x1_part + x2_part
