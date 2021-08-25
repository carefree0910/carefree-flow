import numpy as np
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

from abc import ABCMeta
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union
from typing import Callable
from typing import Optional

from ...types import arrays_type
from ...protocol import LossProtocol
from ...protocol import TrainerState
from ...misc.toolkit import iou
from ...misc.toolkit import to_flow


@LossProtocol.register("iou")
class IOULoss(LossProtocol):
    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        logits = forward_results[0]
        labels = batch[1]
        return 1.0 - iou(logits, labels)


@LossProtocol.register("bce")
class BCELoss(LossProtocol):
    def _init_config(self) -> None:
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        predictions = forward_results[0]
        labels = batch[1]
        losses = self.bce(predictions, labels)
        return losses.mean(tuple(range(1, len(losses.shape))))


@LossProtocol.register("mae")
class MAELoss(LossProtocol):
    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        predictions = forward_results[0]
        labels = batch[1]
        return F.l1_loss(predictions, labels, reduction="none")


@LossProtocol.register("sigmoid_mae")
class SigmoidMAELoss(LossProtocol):
    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        predictions = forward_results[0]
        labels = batch[1]
        losses = F.l1_loss(flow.sigmoid(predictions), labels, reduction="none")
        return losses.mean((1, 2, 3))


@LossProtocol.register("mse")
class MSELoss(LossProtocol):
    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        predictions = forward_results[0]
        labels = batch[1]
        return F.mse_loss(predictions, labels, reduction="none")


@LossProtocol.register("quantile")
class QuantileLoss(LossProtocol):
    def _init_config(self) -> None:
        q = self.config.get("q")
        if q is None:
            raise ValueError("'q' should be provided in Quantile loss")
        if isinstance(q, float):
            self.register_buffer("q", flow.tensor([q], flow.float32))
        else:
            q = np.asarray(q, np.float32).reshape([1, -1])
            self.register_buffer("q", to_flow(q))

    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        quantile_error = batch[1] - forward_results[0]  # type: ignore
        neg_errors = self.q * quantile_error  # type: ignore
        pos_errors = (self.q - 1.0) * quantile_error  # type: ignore
        quantile_losses = flow.max(neg_errors, pos_errors)
        return quantile_losses.mean(1, keepdim=True)


@LossProtocol.register("cross_entropy")
class CrossEntropyLoss(LossProtocol):
    def _init_config(self) -> None:
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        predictions = forward_results[0]
        labels = batch[1].squeeze()  # type: ignore
        return self.ce(predictions, labels)


@LossProtocol.register("focal")
class FocalLoss(LossProtocol):
    def _init_config(self) -> None:
        self._input_logits = self.config.setdefault("input_logits", True)
        self._eps = self.config.setdefault("eps", 1e-6)
        self._gamma = self.config.setdefault("gamma", 2.0)
        alpha = self.config.setdefault("alpha", None)
        if isinstance(alpha, (int, float)):
            alpha = [alpha, 1 - alpha]
        elif isinstance(alpha, (list, tuple)):
            alpha = list(alpha)
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", to_flow(np.array(alpha, np.float32)))

    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        predictions = forward_results[0]
        labels = batch[1]
        if not self._input_logits:
            prob_mat = predictions.view(-1, predictions.shape[-1]) + self._eps  # type: ignore
        else:
            logits_mat = predictions.view(-1, predictions.shape[-1])  # type: ignore
            prob_mat = F.softmax(logits_mat, dim=1) + self._eps
        gathered_prob_flat = prob_mat.gather(dim=1, index=labels).view(-1)
        gathered_log_prob_flat = gathered_prob_flat.log()
        if self.alpha is not None:
            alpha_target = self.alpha.gather(dim=0, index=labels.view(-1))
            gathered_log_prob_flat = gathered_log_prob_flat * alpha_target
        return -gathered_log_prob_flat * (1 - gathered_prob_flat) ** self._gamma


multi_prefix_mapping: Dict[str, Type["MultiLoss"]] = {}


class MultiLoss(LossProtocol, metaclass=ABCMeta):
    prefix: str

    names: Union[str, List[str]]
    configs: Dict[str, Any]
    base_losses: nn.ModuleList

    def _init_config(self) -> None:
        if isinstance(self.names, str):
            base_losses = [LossProtocol.make(self.names, self.configs)]
        else:
            base_losses = [
                LossProtocol.make(name, self.configs.get(name, {}))
                for name in self.names
            ]
        self.base_losses = nn.ModuleList(base_losses)

    @classmethod
    def register_(
        cls,
        base_loss_names: Union[str, List[str]],
        base_configs: Optional[Dict[str, Any]] = None,
        *,
        tag: Optional[str] = None,
    ) -> None:
        if tag is None:
            if isinstance(base_loss_names, str):
                tag = f"{cls.prefix}_{base_loss_names}"
            else:
                tag = f"{cls.prefix}_{'_'.join(base_loss_names)}"
        if tag in cls.d:
            return None

        @cls.register(tag)
        class _(cls):  # type: ignore
            names = base_loss_names
            configs = base_configs or {}

    @classmethod
    def record_prefix(cls) -> Callable[[Type["MultiLoss"]], Type["MultiLoss"]]:
        def _(cls_: Type[MultiLoss]) -> Type[MultiLoss]:
            global multi_prefix_mapping
            multi_prefix_mapping[cls_.prefix] = cls_
            return cls_

        return _


@MultiLoss.record_prefix()
class MultiTaskLoss(MultiLoss):
    prefix = "multi_task"

    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        losses = []
        for loss_ins in self.base_losses:
            loss = loss_ins._core(forward_results, batch, state, **kwargs)
            # TODO : expose these keys
            # losses[f"{loss_ins.__identifier__}"] = loss
            losses.append(loss)
        losses.insert(0, sum(losses))
        return losses


@MultiLoss.record_prefix()
class MultiStageLoss(MultiLoss):
    prefix = "multi_stage"

    def _core(
        self,
        forward_results: arrays_type,
        batch: arrays_type,
        state: Optional[TrainerState] = None,
        **kwargs: Any,
    ) -> arrays_type:
        fr_list = list(forward_results)
        predictions = fr_list[0]
        losses = []
        for i, pred in enumerate(predictions):
            fr_list[0] = pred
            for loss_ins in self.base_losses:
                loss = loss_ins._core(fr_list, batch, state, **kwargs)
                # TODO : expose these keys
                # losses[f"{loss_ins.__identifier__}{i}"] = loss
                losses.append(loss)
        losses.insert(0, sum(losses))
        return losses


__all__ = [
    "IOULoss",
    "MAELoss",
    "MSELoss",
    "QuantileLoss",
    "SigmoidMAELoss",
    "CrossEntropyLoss",
    "FocalLoss",
    "MultiStageLoss",
]
