import oneflow as flow
import oneflow.nn as nn

from typing import Any
from typing import Optional

from .protocol import EncoderBase
from .protocol import Encoder1DBase
from ....types import item_type
from ....types import arrays_type
from ....protocol import TrainerState


@EncoderBase.register("lenet")
class LeNetEncoder(EncoderBase):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        latent_channels: int = 128,
    ):
        super().__init__(img_size, in_channels, 2, latent_channels)
        if img_size % 4 != 0:
            raise ValueError("`img_size` should be divisible by 4")
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=latent_channels,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=5,
                padding=2,
                stride=1,
            ),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        batch_idx: int,
        net: flow.Tensor,
        *inputs: item_type,
    ) -> arrays_type:
        return self.net(net)


@Encoder1DBase.register("lenet")
class LeNetEncoder1D(Encoder1DBase):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        latent_dim: int = 128,
    ):
        super().__init__(img_size, in_channels, latent_dim)
        self.net = LeNetEncoder(img_size, in_channels, latent_dim)
        self.pool = nn.AvgPool2d(kernel_size=img_size // 4)

    def forward(
        self,
        batch_idx: int,
        net: flow.Tensor,
        *inputs: item_type,
    ) -> arrays_type:
        net = self.net(batch_idx, net, *inputs)
        net = self.pool(net)
        return net.squeeze()


__all__ = [
    "LeNetEncoder",
    "LeNetEncoder1D",
]
