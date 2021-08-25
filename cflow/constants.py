from cftool.misc import LoggingMixin

INFO_PREFIX = LoggingMixin.info_prefix
ERROR_PREFIX = LoggingMixin.error_prefix
WARNING_PREFIX = LoggingMixin.warning_prefix

TIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

CKPT_PREFIX = "model_"
SCORES_FILE = "scores.json"
CHECKPOINTS_FOLDER = "checkpoints"

META_CONFIG_NAME = "__meta__"
DATA_CONFIG_FILE = "__data__.json"
ML_PIPELINE_SAVE_NAME = "ml_pipeline"
