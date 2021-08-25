import os
import json
import time
import mlflow
import shutil
import getpass
import platform

from typing import Any
from typing import Dict
from typing import Optional
from cftool.misc import lock_manager
from cftool.misc import fix_float_to_length
from mlflow.exceptions import MlflowException
from mlflow.utils.mlflow_tags import MLFLOW_USER
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from mlflow.tracking.fluent import _RUN_ID_ENV_VAR

from ....trainer import Trainer
from ....trainer import TrainerCallback
from ....protocol import StepOutputs
from ....protocol import TrainerState
from ....protocol import MetricsOutputs
from ....constants import CKPT_PREFIX
from ....constants import SCORES_FILE
from ....constants import WARNING_PREFIX


def parse_mlflow_uri(path: str) -> str:
    delim = "/" if platform.system() == "Windows" else ""
    return f"file://{delim}{path}"


@TrainerCallback.register("_log_metrics_msg")
class _LogMetricsMsgCallback(TrainerCallback):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.timer = time.time()

    def log_metrics_msg(
        self,
        metrics_outputs: MetricsOutputs,
        metrics_log_path: str,
        state: TrainerState,
    ) -> None:
        if not self.is_rank_0:
            return None
        final_score = metrics_outputs.final_score
        metric_values = metrics_outputs.metric_values
        core = " | ".join(
            [
                f"{k} : {fix_float_to_length(metric_values[k], 8)}"
                for k in sorted(metric_values)
            ]
        )
        total_step = state.num_step_per_epoch
        current_step = state.step % total_step
        if current_step == 0:
            current_step = total_step if state.step > 0 else 0
        step_ratio = f"[{current_step} / {total_step}]"
        timer_str = f"[{time.time() - self.timer:.3f}s]"
        msg = (
            f"(epoch {state.epoch:^4d} {step_ratio} {timer_str} | {core} | "
            f"score : {fix_float_to_length(final_score, 8)} |"
        )
        if self.verbose:
            print(msg)
        with open(metrics_log_path, "a") as f:
            f.write(f"{msg}\n")
        self.timer = time.time()


@TrainerCallback.register("_inject_loader_name")
class _InjectLoaderName(TrainerCallback):
    def mutate_train_forward_kwargs(
        self,
        kwargs: Dict[str, Any],
        trainer: "Trainer",
    ) -> None:
        kwargs["loader_name"] = trainer.train_loader.name  # type: ignore


@TrainerCallback.register("mlflow")
class MLFlowCallback(TrainerCallback):
    def __init__(
        self,
        experiment_name: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_name_prefix: Optional[str] = None,
        run_tags: Optional[Dict[str, Any]] = None,
        tracking_folder: str = os.getcwd(),
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.params = params
        self.run_name = run_name
        self.run_name_prefix = run_name_prefix
        self.run_tags = run_tags
        self.tracking_folder = tracking_folder

    def initialize(self) -> None:
        if not self.is_rank_0:
            return None
        tracking_folder = os.path.abspath(self.tracking_folder)
        tracking_dir = os.path.join(tracking_folder, "mlruns")
        with lock_manager(tracking_folder, []) as lock:
            os.makedirs(tracking_dir, exist_ok=True)
            tracking_uri = parse_mlflow_uri(tracking_dir)
            self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
            name = self.experiment_name
            experiment = self.mlflow_client.get_experiment_by_name(name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
            else:
                experiment_id = self.mlflow_client.create_experiment(name)
            lock._stuffs = [os.path.join(tracking_dir, experiment_id)]

        run = None
        from_external = False
        if _RUN_ID_ENV_VAR in os.environ:
            existing_run_id = os.environ[_RUN_ID_ENV_VAR]
            del os.environ[_RUN_ID_ENV_VAR]
            try:
                run = self.mlflow_client.get_run(existing_run_id)
                from_external = True
            except MlflowException:
                print(
                    f"{WARNING_PREFIX}`run_id` is found in environment but "
                    "corresponding mlflow run does not exist. This might cause by "
                    "external calls."
                )

        if run is None:
            if self.run_tags is None:
                self.run_tags = {}
            self.run_tags.setdefault(MLFLOW_USER, getpass.getuser())
            if self.run_name is not None:
                if self.run_name_prefix is not None:
                    self.run_name = f"{self.run_name_prefix}_{self.run_name}"
                self.run_tags.setdefault(MLFLOW_RUN_NAME, self.run_name)
            run = self.mlflow_client.create_run(experiment_id, tags=self.run_tags)
        self.run_id = run.info.run_id

        if not from_external:
            for key, value in (self.params or {}).items():
                self.mlflow_client.log_param(self.run_id, key, value)

    def log_lr(self, key: str, lr: float, state: TrainerState) -> None:
        if not self.is_rank_0:
            return None
        self.mlflow_client.log_metric(self.run_id, key, lr, step=state.step)

    def log_metrics(self, metric_outputs: MetricsOutputs, state: TrainerState) -> None:
        if not self.is_rank_0:
            return None
        for key, value in metric_outputs.metric_values.items():
            self.mlflow_client.log_metric(self.run_id, key, value, step=state.step)
        score = metric_outputs.final_score
        self.mlflow_client.log_metric(self.run_id, "score", score, step=state.step)

    def log_artifacts(self, trainer: Trainer) -> None:
        if not self.is_rank_0:
            return None
        self.mlflow_client.log_artifacts(self.run_id, trainer.workplace)

    def after_step(self, step_outputs: StepOutputs, state: TrainerState) -> None:
        if not self.is_rank_0:
            return None
        if state.should_log_losses:
            for i, tensor in enumerate(step_outputs.losses):
                key = f"tr_loss_{i}"
                value = tensor.item()
                self.mlflow_client.log_metric(self.run_id, key, value, step=state.step)

    def finalize(self, trainer: Trainer) -> None:
        if not self.is_rank_0:
            return None
        self.log_artifacts(trainer)
        self.mlflow_client.set_terminated(self.run_id)


class ArtifactCallback(TrainerCallback):
    key: str

    def __init__(self, num_keep: int = 25):
        super().__init__()
        self.num_keep = num_keep

    def _prepare_folder(self, trainer: Trainer, *, check_num_keep: bool = True) -> str:
        state = trainer.state
        sub_folder = os.path.join(trainer.workplace, self.key)
        os.makedirs(sub_folder, exist_ok=True)
        if not check_num_keep:
            sub_folder = os.path.join(sub_folder, str(state.step))
            os.makedirs(sub_folder, exist_ok=True)
            return sub_folder
        current_steps = sorted(map(int, os.listdir(sub_folder)))
        if len(current_steps) >= self.num_keep:
            must_keep = set()
            checkpoint_folder = trainer.checkpoint_folder
            if checkpoint_folder is not None:
                score_path = os.path.join(checkpoint_folder, SCORES_FILE)
                with open(score_path, "r") as f:
                    for key in json.load(f):
                        name = os.path.splitext(key)[0]
                        must_keep.add(int(name[len(CKPT_PREFIX) :]))
            num_left = len(current_steps)
            for step in current_steps:
                if step in must_keep:
                    continue
                shutil.rmtree(os.path.join(sub_folder, str(step)))
                num_left -= 1
                if num_left < self.num_keep:
                    break
        sub_folder = os.path.join(sub_folder, str(state.step))
        os.makedirs(sub_folder, exist_ok=True)
        return sub_folder


__all__ = [
    "MLFlowCallback",
    "ArtifactCallback",
]
