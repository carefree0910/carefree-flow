from typing import Type
from typing import Callable
from oneflow.optim.lr_scheduler import StepLR
from oneflow.optim.lr_scheduler import CosineAnnealingLR


scheduler_dict = {}


def register_scheduler(name: str) -> Callable[[Type], Type]:
    def _register(cls_: Type) -> Type:
        global scheduler_dict
        scheduler_dict[name] = cls_
        return cls_

    return _register


register_scheduler("step")(StepLR)
register_scheduler("cosine")(CosineAnnealingLR)


__all__ = ["scheduler_dict", "register_scheduler"]
