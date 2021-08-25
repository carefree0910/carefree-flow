import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from sklearn import metrics
from scipy import stats as ss

from ...types import np_arrays_type
from ...protocol import MetricsOutputs
from ...protocol import MetricProtocol
from ...protocol import DataLoaderProtocol
from ...misc.toolkit import iou
from ...misc.toolkit import softmax


@MetricProtocol.register("acc")
class Accuracy(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        logits = np_outputs[0]
        predictions = logits.argmax(1)
        labels = np_batch[1].reshape(predictions.shape)  # type: ignore
        return (predictions == labels).mean().item()


@MetricProtocol.register("quantile")
class Quantile(MetricProtocol):
    def __init__(self, q: Any):
        super().__init__()
        if not isinstance(q, float):
            q = np.asarray(q, np.float32).reshape([1, -1])
        self.q = q

    @property
    def is_positive(self) -> bool:
        return False

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        diff = np_batch[1] - np_outputs[0]  # type: ignore
        return np.maximum(self.q * diff, (self.q - 1.0) * diff).mean(0).sum().item()


@MetricProtocol.register("f1")
class F1Score(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        labels = np_batch[1].ravel()  # type: ignore
        predictions = np_outputs[0].ravel()
        return metrics.f1_score(labels, predictions)


@MetricProtocol.register("r2")
class R2Score(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        labels = np_batch[1].ravel()  # type: ignore
        predictions = np_outputs[0].ravel()
        return metrics.r2_score(labels, predictions)


@MetricProtocol.register("auc")
class AUC(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    @property
    def requires_all(self) -> bool:
        return True

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        logits = np_outputs[0]
        num_classes = logits.shape[1]
        probabilities = softmax(logits)
        labels = np_batch[1].ravel()  # type: ignore
        if num_classes == 2:
            return metrics.roc_auc_score(labels, probabilities[..., 1])
        return metrics.roc_auc_score(labels, probabilities, multi_class="ovr")


@MetricProtocol.register("mae")
class MAE(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return False

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        predictions = np_outputs[0]
        return np.mean(np.abs(np_batch[1] - predictions)).item()  # type: ignore


@MetricProtocol.register("mse")
class MSE(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return False

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        predictions = np_outputs[0]
        return np.mean(np.square(np_batch[1] - predictions)).item()  # type: ignore


@MetricProtocol.register("ber")
class BER(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return False

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        labels = np_batch[1].ravel()  # type: ignore
        predictions = np_outputs[0].ravel()
        mat = metrics.confusion_matrix(labels, predictions)
        tp = np.diag(mat)
        fp = mat.sum(axis=0) - tp
        fn = mat.sum(axis=1) - tp
        tn = mat.sum() - (tp + fp + fn)
        return (0.5 * np.mean((fn / (tp + fn) + fp / (tn + fp)))).item()


@MetricProtocol.register("corr")
class Correlation(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        labels = np_batch[1].ravel()  # type: ignore
        predictions = np_outputs[0].ravel()
        return float(ss.pearsonr(labels, predictions)[0])


@MetricProtocol.register("iou")
class IOU(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        return True

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        logits = np_outputs[0]
        labels = np_batch[0]
        return iou(logits, labels).mean().item()


class MultipleMetrics(MetricProtocol):
    @property
    def is_positive(self) -> bool:
        raise NotImplementedError

    @property
    def requires_all(self) -> bool:
        return any(metric.requires_all for metric in self.metrics)

    def _core(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol],
    ) -> float:
        raise NotImplementedError

    def __init__(
        self,
        metric_list: List[MetricProtocol],
        *,
        weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.metrics = metric_list
        self.weights = weights or {}

    def evaluate(
        self,
        np_batch: np_arrays_type,
        np_outputs: np_arrays_type,
        loader: Optional[DataLoaderProtocol] = None,
    ) -> MetricsOutputs:
        scores: List[float] = []
        weights: List[float] = []
        metrics_values: Dict[str, float] = {}
        for metric in self.metrics:
            metric_outputs = metric.evaluate(np_batch, np_outputs, loader)
            w = self.weights.get(metric.__identifier__, 1.0)
            weights.append(w)
            scores.append(metric_outputs.final_score * w)
            metrics_values.update(metric_outputs.metric_values)
        return MetricsOutputs(sum(scores) / sum(weights), metrics_values)


__all__ = [
    "AUC",
    "BER",
    "IOU",
    "MAE",
    "MSE",
    "F1Score",
    "R2Score",
    "Accuracy",
    "Quantile",
    "Correlation",
    "MultipleMetrics",
]
