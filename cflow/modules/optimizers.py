import oneflow as flow

from typing import Dict
from typing import Type
from typing import Callable
from oneflow.optim import Optimizer


optimizer_dict: Dict[str, Type[Optimizer]] = {}


def register_optimizer(name: str) -> Callable[[Type], Type]:
    def _register(cls_: Type) -> Type:
        global optimizer_dict
        optimizer_dict[name] = cls_
        return cls_

    return _register


register_optimizer("sgd")(flow.optim.SGD)
register_optimizer("adam")(flow.optim.Adam)
register_optimizer("adamw")(flow.optim.AdamW)
register_optimizer("rmsprop")(flow.optim.RMSprop)


__all__ = ["optimizer_dict", "register_optimizer"]
