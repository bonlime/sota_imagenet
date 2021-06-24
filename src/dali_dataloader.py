"""
data loader using NVIDIA DALI v1.2

there are still two things missing comared to previous version of loader
TODO: (emil 19.06.21) Rectangular Validation
TODO: (emil 19.06.21) mixmatch pipeline 
"""
import math
import torch
from pathlib import Path
from copy import deepcopy
from loguru import logger
from omegaconf import OmegaConf

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import nvidia.dali.tfrecord as tfrec

from pytorch_tools.utils.misc import env_rank, env_world_size, listify

from src.arg_parser import LoaderConfig, StrictConfig, TrainLoaderConfig, ValLoaderConfig

# values used for normalization. there is no reason to use Imagenet mean/std so i'm normalizing to [-5, 5]
DATA_MEAN = (0.5 * 255, 0.5 * 255, 0.5 * 255)
DATA_STD = (0.2 * 255, 0.2 * 255, 0.2 * 255)

# DATA_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
# DATA_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def mix(condition, true_case, false_case):
    """multiplex between two images. needed because DALI doesn't support conditional execution"""
    neg_condition = condition ^ True
    return condition * true_case + neg_condition * false_case


@pipeline_def
def train_pipeline(cfg: TrainLoaderConfig):
    root_dir = Path(cfg.root_data_dir)
    common_input_kwargs = dict(random_shuffle=True, shard_id=env_rank(), num_shards=env_world_size(), name="Reader")
    if cfg.use_tfrecords:
        records = [str(i) for i in sorted((root_dir / "train_records").iterdir())]
        indexes = [str(i) for i in sorted((root_dir / "train_indexes").iterdir())]
        features = {
            "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
        }
        inputs = fn.readers.tfrecord(
            path=records, index_path=indexes, features=features, read_ahead=True, **common_input_kwargs,
        )
        jpeg, label = inputs["image/encoded"], inputs["image/class/label"]
    else:
        jpeg, label = fn.readers.file(file_root=root_dir / "train", **common_input_kwargs)
    image = fn.decoders.image_random_crop(
        jpeg,
        device="mixed",
        random_aspect_ratio=[0.75, 1.25],
        random_area=[cfg.min_area, 1.0],
        num_attempts=100,
        output_type=types.RGB,
    )

    image_tr = fn.resize(image, device="gpu", size=cfg.image_size, interp_type=types.INTERP_TRIANGULAR)
    if cfg.random_interpolation:
        image_cub = fn.resize(image, device="gpu", size=cfg.image_size, interp_type=types.INTERP_CUBIC)
        image = mix(fn.random.coin_flip(probability=0.5), image_cub, image_tr)
    else:
        image = image_tr

    if cfg.blur_prob > 0:
        blur_image = fn.gaussian_blur(image, device="gpu", window_size=11, sigma=fn.random.uniform(range=[0.5, 1.1]))
        image = mix(fn.random.coin_flip(probability=cfg.blur_prob, dtype=types.BOOL), blur_image, image)

    if cfg.color_twist_prob > 0:
        image_ct = fn.color_twist(
            image,
            device="gpu",
            contrast=fn.random.uniform(range=[0.7, 1.3]),
            brightness=fn.random.uniform(range=[0.7, 1.3]),
            hue=fn.random.uniform(range=[-20, 20]),  # in degrees
            saturation=fn.random.uniform(range=[0.7, 1.3]),
        )
        image = mix(fn.random.coin_flip(probability=cfg.color_twist_prob, dtype=types.BOOL), image_ct, image)

    if cfg.gray_prob > 0:
        grayscale_coin = fn.cast(fn.random.coin_flip(probability=cfg.gray_prob), dtype=types.FLOAT)
        image = fn.hsv(image, device="gpu", saturation=grayscale_coin)

    if cfg.re_prob:  # random erasing
        image_re = fn.erase(
            image,
            device="gpu",
            anchor=fn.random.uniform(range=(0.0, 1), shape=cfg.re_count * 2),
            shape=fn.random.uniform(range=(0.05, 0.25), shape=cfg.re_count * 2),
            axis_names="HW",
            fill_value=DATA_MEAN,
            normalized_anchor=True,
            normalized_shape=True,
        )
        image = mix(fn.random.coin_flip(probability=cfg.re_prob, dtype=types.BOOL), image_re, image)

    image = fn.crop_mirror_normalize(
        image,
        device="gpu",
        crop=(cfg.image_size, cfg.image_size),
        mirror=fn.random.coin_flip(probability=0.5),
        mean=DATA_MEAN,
        std=DATA_STD,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
    )
    label = fn.one_hot(label, num_classes=cfg.num_classes).gpu()
    return image, label


@pipeline_def
def val_pipeline(cfg: ValLoaderConfig):
    root_dir = Path(cfg.root_data_dir)
    common_input_kwargs = dict(shard_id=env_rank(), num_shards=env_world_size(), name="Reader")
    if cfg.use_tfrecords:
        records = [str(i) for i in sorted((root_dir / "val_records").iterdir())]
        indexes = [str(i) for i in sorted((root_dir / "val_indexes").iterdir())]
        features = {
            "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
        }
        inputs = fn.readers.tfrecord(
            path=records, index_path=indexes, features=features, read_ahead=True, **common_input_kwargs
        )
        jpeg, label = inputs["image/encoded"], inputs["image/class/label"]
    else:
        jpeg, label = fn.readers.file(file_root=str(root_dir / "val"), **common_input_kwargs)

    image = fn.decoders.image(jpeg, device="mixed", output_type=types.RGB)

    crop_size = cfg.image_size if cfg.full_crop else math.ceil((cfg.image_size * 1.14 + 8) // 16 * 16)
    image = fn.resize(image, device="gpu", interp_type=types.INTERP_TRIANGULAR, resize_shorter=crop_size)

    image = fn.crop_mirror_normalize(
        image,
        device="gpu",
        crop=(cfg.image_size, cfg.image_size),
        mean=DATA_MEAN,
        std=DATA_STD,
        dtype=types.FLOAT,
        output_layout=types.NCHW,
    )
    label = fn.one_hot(label, num_classes=cfg.num_classes).gpu()
    return image, label


class DaliLoader:
    """Wrap dali to look like torch dataloader"""

    def __init__(self, cfg: LoaderConfig):
        """Returns train or val iterator over Imagenet data"""
        pipeline = train_pipeline if cfg._is_train else val_pipeline
        pipe = pipeline(batch_size=cfg.batch_size, num_threads=cfg.workers, device_id=env_rank(), cfg=cfg)
        pipe.build()
        self.loader = DALIClassificationIterator(
            pipe, reader_name="Reader", auto_reset=True, last_batch_policy=LastBatchPolicy.DROP,
        )

    def __len__(self):
        return math.ceil(self.loader._size / self.loader.batch_size)

    def __iter__(self):
        return ((batch[0]["data"], batch[0]["label"]) for batch in self.loader)


class DaliDataManager:
    """Wrapper to reinitialize loaders durring training. Allows progressive image resize, augmentaiton increase/decrease and etc."""

    def __init__(self, cfg: StrictConfig):
        self.cfg = cfg
        self.stages = cfg.run.stages
        self.tot_epochs = max([stage.end for stage in self.stages])
        self._validate_stages()

        self.loader = None
        self.val_loader = None
        self.start_epoch = None
        self.end_epoch = None

    def __len__(self):
        return len(self.stages)

    def _validate_stages(self):
        end = 0
        for stage in self.stages:
            assert stage.start == end, "error in data stages. start != end"
            assert stage.end > stage.start, "error in data stages, end <= start"
            end = stage.end

    def set_stage(self, idx: int) -> None:
        self.start_epoch = self.stages[idx].start
        self.end_epoch = self.stages[idx].end

        if self.stages[idx].extra_args is None and self.loader is not None:
            return  # only learning rate changed, no need to create loader

        train_cfg = deepcopy(self.cfg.loader)
        val_cfg = deepcopy(self.cfg.val_loader)

        if self.stages[idx].extra_args is not None:
            for key, value in self.stages[idx].extra_args.items():
                setattr(train_cfg, key, value)

        # for now only image size changes in val loader
        val_cfg.image_size = train_cfg.image_size

        logger.info(f"Loader changed. New data config:\n{OmegaConf.to_yaml(train_cfg)}")

        # remove previous loader to prevent DALI errors
        if self.loader is not None:
            del self.loader
            del self.val_loader
            torch.cuda.empty_cache()

        self.loader = DaliLoader(train_cfg)
        self.val_loader = DaliLoader(val_cfg)
