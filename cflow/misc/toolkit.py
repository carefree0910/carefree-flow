import os
import json
import math
import time
import shutil
import inspect
import logging
import urllib.request

import numpy as np
import oneflow as flow
import oneflow.nn as nn
import matplotlib.pyplot as plt
import oneflow.nn.functional as F

from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import ContextManager
from zipfile import ZipFile
from argparse import Namespace
from datetime import datetime
from datetime import timedelta
from collections import OrderedDict
from cftool.misc import register_core
from cftool.misc import show_or_save
from cftool.misc import shallow_copy_dict
from cftool.misc import context_error_handler
from cftool.misc import LoggingMixin

from ..types import arr_type
from ..types import arrays_type
from ..types import general_config_type
from ..types import sample_weights_type
from ..constants import TIME_FORMAT
from ..constants import WARNING_PREFIX


# general


def _parse_config(config: general_config_type) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, str):
        with open(config, "r") as f:
            return json.load(f)
    return shallow_copy_dict(config)


def prepare_workplace_from(workplace: str, timeout: timedelta = timedelta(7)) -> str:
    current_time = datetime.now()
    if os.path.isdir(workplace):
        for stuff in os.listdir(workplace):
            if not os.path.isdir(os.path.join(workplace, stuff)):
                continue
            try:
                stuff_time = datetime.strptime(stuff, TIME_FORMAT)
                stuff_delta = current_time - stuff_time
                if stuff_delta > timeout:
                    print(
                        f"{WARNING_PREFIX}{stuff} will be removed "
                        f"(already {stuff_delta} ago)"
                    )
                    shutil.rmtree(os.path.join(workplace, stuff))
            except:
                pass
    workplace = os.path.join(workplace, current_time.strftime(TIME_FORMAT))
    os.makedirs(workplace)
    return workplace


def get_latest_workplace(root: str) -> Optional[str]:
    all_workplaces = []
    for stuff in os.listdir(root):
        if not os.path.isdir(os.path.join(root, stuff)):
            continue
        try:
            datetime.strptime(stuff, TIME_FORMAT)
            all_workplaces.append(stuff)
        except:
            pass
    if not all_workplaces:
        return None
    return os.path.join(root, sorted(all_workplaces)[-1])


def sort_dict_by_value(d: Dict[Any, Any], *, reverse: bool = False) -> OrderedDict:
    sorted_items = sorted([(v, k) for k, v in d.items()], reverse=reverse)
    return OrderedDict({item[1]: item[0] for item in sorted_items})


def to_standard(arr: np.ndarray) -> np.ndarray:
    if is_int(arr):
        arr = arr.astype(np.int64)
    elif is_float(arr):
        arr = arr.astype(np.float32)
    return arr


def parse_args(args: Any) -> Namespace:
    return Namespace(**{k: None if not v else v for k, v in args.__dict__.items()})


def parse_path(path: Optional[str], root_dir: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if root_dir is None:
        return path
    return os.path.abspath(os.path.join(root_dir, path))


def get_arguments(*, pop_class_attributes: bool = True) -> Dict[str, Any]:
    frame = inspect.currentframe().f_back  # type: ignore
    if frame is None:
        raise ValueError("`get_arguments` should be called inside a frame")
    arguments = inspect.getargvalues(frame)[-1]
    if pop_class_attributes:
        arguments.pop("self", None)
        arguments.pop("__class__", None)
    return arguments


def download_dataset(
    name: str,
    *,
    root: str = os.getcwd(),
    remove_zip: Optional[bool] = None,
    extract_zip: bool = True,
    prefix: str = "https://github.com/carefree0910/datasets/releases/download/latest/",
) -> None:
    os.makedirs(root, exist_ok=True)
    file = f"{name}.zip"
    tgt_zip_path = os.path.join(root, file)
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=name) as t:
        urllib.request.urlretrieve(
            f"{prefix}{file}",
            filename=tgt_zip_path,
            reporthook=t.update_to,
        )
    if extract_zip:
        with ZipFile(tgt_zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(root, name))
    if remove_zip is None:
        remove_zip = extract_zip
    if remove_zip:
        if extract_zip:
            os.remove(tgt_zip_path)
        else:
            print(f"{WARNING_PREFIX}zip file is not extracted, so we'll not remove it!")


def _rmtree(folder: str, patience: float = 10.0) -> None:
    if not os.path.isdir(folder):
        return None
    t = time.time()
    while True:
        try:
            if time.time() - t >= patience:
                prefix = LoggingMixin.warning_prefix
                print(f"\n{prefix}failed to rmtree: {folder}")
                break
            shutil.rmtree(folder)
            break
        except:
            print("", end=".", flush=True)
            time.sleep(1)


T = TypeVar("T")


class WithRegister(Generic[T]):
    d: Dict[str, Type[T]]
    __identifier__: str

    @classmethod
    def get(cls, name: str) -> Type[T]:
        return cls.d[name]

    @classmethod
    def make(cls, name: str, config: Dict[str, Any]) -> T:
        return cls.get(name)(**config)  # type: ignore

    @classmethod
    def make_multiple(
        cls,
        names: Union[str, List[str]],
        configs: Optional[Dict[str, Any]] = None,
    ) -> Union[T, List[T]]:
        if configs is None:
            configs = {}
        if isinstance(names, str):
            return cls.make(names, configs)  # type: ignore
        return [
            cls.make(name, shallow_copy_dict(configs.get(name, {})))  # type: ignore
            for name in names
        ]

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        def before(cls_: Type) -> None:
            cls_.__identifier__ = name

        return register_core(name, cls.d, before_register=before)


class WeightsStrategy:
    def __init__(self, strategy: Optional[str]):
        self.strategy = strategy

    def __call__(self, num_train: int, num_valid: int) -> sample_weights_type:
        if self.strategy is None:
            return None
        return getattr(self, self.strategy)(num_train, num_valid)

    def linear_decay(self, num_train: int, num_valid: int) -> sample_weights_type:
        return np.linspace(0, 1, num_train + 1)[1:]

    def radius_decay(self, num_train: int, num_valid: int) -> sample_weights_type:
        return np.sin(np.arccos(1.0 - np.linspace(0, 1, num_train + 1)[1:]))

    def log_decay(self, num_train: int, num_valid: int) -> sample_weights_type:
        return np.log(np.arange(num_train) + np.e)

    def sigmoid_decay(self, num_train: int, num_valid: int) -> sample_weights_type:
        x = np.linspace(-5.0, 5.0, num_train)
        return 1.0 / (1.0 + np.exp(-x))

    def visualize(self, export_path: str = "weights_strategy.png") -> None:
        n = 1000
        x = np.linspace(0, 1, n)
        y = self(n, 0)
        if isinstance(y, tuple):
            y = y[0]
        plt.figure()
        plt.plot(x, y)
        show_or_save(export_path)


class LoggingMixinWithRank(LoggingMixin):
    is_rank_0: bool = True

    def set_rank_0(self, value: bool) -> None:
        self.is_rank_0 = value
        for v in self.__dict__.values():
            if isinstance(v, LoggingMixinWithRank):
                v.set_rank_0(value)

    def _init_logging(
        self,
        verbose_level: Optional[int] = 2,
        trigger: bool = True,
    ) -> None:
        if not self.is_rank_0:
            return None
        super()._init_logging(verbose_level, trigger)

    def log_msg(
        self,
        body: str,
        prefix: str = "",
        verbose_level: Optional[int] = 1,
        msg_level: int = logging.INFO,
        frame: Any = None,
    ) -> None:
        if not self.is_rank_0:
            return None
        super().log_msg(body, prefix, verbose_level, msg_level, frame)

    def log_block_msg(
        self,
        body: str,
        prefix: str = "",
        title: str = "",
        verbose_level: Optional[int] = 1,
        msg_level: int = logging.INFO,
        frame: Any = None,
    ) -> None:
        if not self.is_rank_0:
            return None
        super().log_block_msg(body, prefix, title, verbose_level, msg_level, frame)

    def log_timing(self) -> None:
        if not self.is_rank_0:
            return None
        return super().log_timing()


class DownloadProgressBar(tqdm):
    def update_to(
        self,
        b: int = 1,
        bsize: int = 1,
        tsize: Optional[int] = None,
    ) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# dl


def to_flow(arr: np.ndarray) -> flow.Tensor:
    dtype = flow.int32 if is_int(arr) else flow.float32
    return flow.tensor(arr, dtype=dtype)


def to_numpy(tensor: flow.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def to_device(batch: arrays_type, device: flow.device) -> arrays_type:
    return tuple(
        v.to(device)
        if isinstance(v, flow.Tensor)
        else [vv.to(device) if isinstance(vv, flow.Tensor) else vv for vv in v]
        if isinstance(v, list)
        else v
        for v in batch
    )


def squeeze(arr: arr_type) -> arr_type:
    n = arr.shape[0]
    arr = arr.squeeze()
    if n == 1:
        arr = arr[None, ...]
    return arr


def softmax(arr: arr_type) -> arr_type:
    if isinstance(arr, flow.Tensor):
        return F.softmax(arr, dim=1)
    logits = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(1, keepdims=True)


class mode_context(context_error_handler):
    def __init__(
        self,
        module: nn.Module,
        *,
        to_train: Optional[bool],
        use_grad: Optional[bool],
    ):
        self._to_train = to_train
        self._module, self._training = module, module.training
        self._cache = {p: p.requires_grad for p in module.parameters()}
        if use_grad is not None:
            for p in module.parameters():
                p.requires_grad_(use_grad)
        if use_grad is None:
            self._grad_context: Optional[ContextManager] = None
        else:
            self._grad_context = flow.grad_enable() if use_grad else flow.no_grad()

    def __enter__(self) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._to_train)
        if self._grad_context is not None:
            self._grad_context.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._to_train is not None:
            self._module.train(mode=self._training)
        if self._grad_context is not None:
            self._grad_context.__exit__(exc_type, exc_val, exc_tb)
        for p, v in self._cache.items():
            p.requires_grad_(v)


class train_context(mode_context):
    def __init__(self, module: nn.Module, *, use_grad: bool = True):
        super().__init__(module, to_train=True, use_grad=use_grad)


class eval_context(mode_context):
    def __init__(self, module: nn.Module, *, use_grad: Optional[bool] = False):
        super().__init__(
            module,
            to_train=False,
            use_grad=use_grad,
        )


# ml


def is_int(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.integer)


def is_float(arr: np.ndarray) -> bool:
    return np.issubdtype(arr.dtype, np.floating)


# cv


@flow.no_grad()
def make_grid(
    tensor: Union[flow.Tensor, List[flow.Tensor]],
    num_rows: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
) -> flow.Tensor:
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = flow.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = flow.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = flow.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place

        def norm_ip(img: flow.Tensor, low: Any, high: Any) -> flow.Tensor:
            img = img.clamp(min=low, max=high)
            img = (img - low) / max(high - low, 1.0e-5)
            return img

        def norm_range(t: flow.Tensor, value_range: Any) -> flow.Tensor:
            if value_range is not None:
                t = norm_ip(t, value_range[0], value_range[1])
            else:
                t = norm_ip(t, to_numpy(t.min()).item(), to_numpy(t.max()).item())
            return t

        if scale_each is True:
            for i, t in enumerate(tensor):
                tensor[i] = norm_range(t, value_range)
        else:
            tensor = norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(num_rows, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = flow.zeros(num_channels, height * ymaps + padding, width * xmaps + padding)
    grid += pad_value
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h_start, w_start = y * height + padding, x * width + padding
            h_end, w_end = h_start + height - padding, w_start + width - padding
            grid[:, h_start:h_end, w_start:w_end] = tensor[k]
            k = k + 1
    return grid


def save_images(arr: arr_type, path: str, num_rows: Optional[int] = None) -> None:
    if isinstance(arr, np.ndarray):
        arr = to_flow(arr)
    if num_rows is None:
        num_rows = math.ceil(math.sqrt(arr.shape[0]))
    grid = make_grid(arr, num_rows=num_rows, normalize=True)
    grid = grid.mul(255).add_(0.5).clamp(0, 255).permute(1, 2, 0)
    grid = grid.to("cpu", flow.uint8).numpy()
    Image.fromarray(grid).save(path)


def iou(logits: arr_type, labels: arr_type) -> arr_type:
    is_torch = isinstance(logits, flow.Tensor)
    num_classes = logits.shape[1]
    if num_classes == 1:
        if is_torch:
            heat_map = flow.sigmoid(logits)
        else:
            heat_map = 1.0 / (1.0 + np.exp(-logits))
    elif num_classes == 2:
        heat_map = softmax(logits)[:, [1]]
    else:
        raise ValueError("`IOU` only supports binary situations")
    intersect = heat_map * labels
    union = heat_map + labels - intersect
    kwargs = {"dim" if is_torch else "axis": tuple(range(1, len(intersect.shape)))}
    return intersect.sum(**kwargs) / union.sum(**kwargs)


def make_indices_visualization_map(indices: flow.Tensor) -> flow.Tensor:
    images = []
    for idx in indices.view(-1).tolist():
        img = Image.new("RGB", (28, 28), (250, 250, 250))
        draw = ImageDraw.Draw(img)
        draw.text((12, 9), str(idx), (0, 0, 0))
        images.append(to_flow(np.array(img).transpose([2, 0, 1])))
    return flow.stack(images).float()
