import oneflow as flow
import oneflow.nn as nn

from typing import Any
from typing import Dict
from typing import Optional

from ..encoder import Encoder1DBase
from ....types import item_type
from ....types import arrays_type
from ....protocol import ModelProtocol


@ModelProtocol.register("clf")
class VanillaClassifier(ModelProtocol):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_size: Optional[int] = None,
        latent_dim: int = 128,
        *,
        encoder1d: str = "lenet",
        encoder1d_configs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        # encoder1d
        if encoder1d_configs is None:
            encoder1d_configs = {}
        encoder1d_configs["img_size"] = img_size
        encoder1d_configs["in_channels"] = in_channels
        encoder1d_configs["latent_dim"] = latent_dim
        self.encoder1d = Encoder1DBase.make(encoder1d, config=encoder1d_configs)
        # head
        self.head = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=latent_dim, out_features=num_classes),
        )

    def forward(
        self,
        batch_idx: int,
        net: flow.Tensor,
        *inputs: item_type,
    ) -> arrays_type:
        encoding = self.encoder1d(batch_idx, net, *inputs)
        return self.head(encoding)

    def classify(self, net: flow.Tensor) -> flow.Tensor:
        return self.forward(0, net)


__all__ = ["VanillaClassifier"]
