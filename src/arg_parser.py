"""Replacement for old argument parser which uses Hydra for better config management"""

from typing import Optional, Dict, List, Any, Tuple, Union
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from loguru import logger


@dataclass
class LoaderConfig:
    """common parameters for train/val pipelines"""

    # image size. could be different for train/val loaders!
    image_size: int = 224
    batch_size: int = 256
    # number of dataloading workers
    workers: int = 6  # enough to fully utilize GPU
    # number of classes in dataset
    num_classes: int = 1000
    _is_train: bool = False
    root_data_dir: str = "${env:IMAGENET_DIR}"
    use_tfrecords: bool = False


@dataclass
class TrainLoaderConfig(LoaderConfig):
    """train pipeline augmentations"""

    _is_train: bool = True
    # min sampled area for image during training
    min_area: float = 0.08
    # probability of applying Gaussian Blur
    blur_prob: float = 0
    # probability of turning image into grayscale
    gray_prob: float = 0
    # probability of applying brightness-contrast-hue-saturation augmentation
    color_twist_prob: float = 0
    # 3 arg below are for color twist augmentation. the have no effect if `color_twist_prob` is 0
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    brightness_range: Tuple[float, float] = (0.7, 1.3)

    # if True randomly use triangular / cubic interpolations for images. Works as a strong augmentation and makes model
    # very robust to interpolation method during inference
    random_interpolation: bool = False
    # random erase probability. removes rectangle patches from image and feel them with mean value
    # Ref: https://arxiv.org/pdf/1708.04896.pdf
    re_prob: float = 0
    re_count: int = 3


@dataclass
class ValLoaderConfig(LoaderConfig):
    """validation pipeline"""

    # 50.000 should be dividable by batch_size * num_gpu
    # otherwise reduced accuracy differs from accuracy on 1 gpu
    batch_size: int = 250
    full_crop: bool = False


@dataclass
class DataStage:
    # start and end epoch for current stage
    start: int = 0
    end: int = 90
    lr: Optional[Tuple[float, float]] = None
    lr_mode: Optional[str] = "linear"
    extra_args: Optional[Dict] = None


@dataclass
class RunnerConfig:
    stages: List = field(default_factory=lambda: [DataStage(lr=(0.1, 0))])

    # path to checkpoint to load from
    resume: Optional[str] = None
    # sometimes we want to resume training like nothing happened, but sometimes want to start from scratch
    load_start_epoch: bool = True
    start_epoch: int = 0

    # usefull to emulate larger batch size. take care to scale LR accordingly!
    accumulate_steps: int = 1
    # Exponential moving average decay. i usually choose decay as 0.3 ^ (-step_per_epoch). typically should be 0.999 for 4 GPUs and 0.9997 for 1 GPU
    ema_decay: float = 0
    # flag to use mixed precision. on by default
    fp16: bool = True

    extra_callbacks: List = field(
        default_factory=lambda: (
            dict(_target_="pytorch_tools.fit_wrapper.callbacks.Callback"),
            dict(_target_="pytorch_tools.fit_wrapper.callbacks.Callback"),
        )
    )
    # if True, skip training and only evaluate model. should be combined with passing `resume` otherwise this is useless
    evaluate: bool = False


@dataclass
class LoggerConfig:
    exp_name: str = "test_run"
    dir: str = "logs"
    # print model before training. usefull to validate that everything is correct
    print_model: bool = False
    # add histogram of weights to TB each epoch
    histogram: bool = False
    # Flag to also save optimizer into save dict. makes it 2x times larger
    save_optim: bool = False


# this 4 lines are an example of how to make some attributes a group. maybe usefull at some stage
# defaults = [{"data": "default_data"}]
# defaults: List[Any] = field(default_factory=lambda: defaults)
# data: Any = MISSING
# cs.store(group="data", name="default_data", node=DataConfig)


@dataclass
class StrictConfig:
    loader: TrainLoaderConfig = TrainLoaderConfig()
    val_loader: ValLoaderConfig = ValLoaderConfig()

    model: Dict[str, Any] = field(default_factory=lambda: dict(_target_="pytorch_tools.models.resnet18"))
    # flag to convert all convs to WS convs
    weight_standardization: bool = False

    # flag to filter BN from wd. makes it much easier for model to overfit
    filter_bn_wd: bool = False
    bn_momentum: float = 0.1
    init_gamma: Optional[float] = 1.72  # for swish

    # by default using fused version of SGD because it's slightly faster
    optim: Dict[str, Any] = field(
        default_factory=lambda: dict(_target_="torch.optim._multi_tensor.SGD", lr=0, weight_decay=1e-4)
    )
    # default loss is CCE
    criterion: Dict[str, Any] = field(
        default_factory=lambda: dict(_target_="pytorch_tools.losses.smooth.CrossEntropyLoss")
    )
    run: RunnerConfig = RunnerConfig()
    log: LoggerConfig = LoggerConfig()
    # if True, would only train for 10 steps each epoch
    debug: bool = False
    # if given would make run reproducible
    random_seed: Optional[int] = None

    # this arg should be set in your shell
    world_size: int = "${env:WORLD_SIZE}"
    local_rank: int = "${env:LOCAL_RANK}"

    # this would be filled later in code
    distributed: bool = False
    is_master: bool = True


cs = ConfigStore.instance()
cs.store(name="strict_config", node=StrictConfig)


@hydra.main(config_path="../configs", config_name="base")
def test_app(cfg: StrictConfig) -> None:
    # test that parsing works as expected
    print(OmegaConf.to_yaml(cfg))

    # import os
    # print("Working directory : {}".format(os.getcwd()))

    # import subprocess
    # kwargs = {"universal_newlines": True, "stdout": subprocess.PIPE}
    # with open("commit_hash.txt", "w") as f:
    #     f.write(subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs).stdout)
    # with open("diff.txt", "w") as f:
    #     f.write(subprocess.run(["git", "diff"], **kwargs).stdout)

    # model = hydra.utils.call(cfg.model)
    # optim = hydra.utils.call(cfg.optim, model.parameters())
    # print(optim)
    # crit = hydra.utils.call(cfg.criterion)
    # print(crit)
    # logger.info("Log inside argparse")
    # extra_clb = [hydra.utils.call(clb_cfg) for clb_cfg in cfg.runner.extra_callbacks]
    # print(extra_clb)
    return 100


if __name__ == "__main__":
    test_app()
