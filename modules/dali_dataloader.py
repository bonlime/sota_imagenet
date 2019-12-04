"""Dali dataloader for imagenet"""
from nvidia import dali
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import modules.config as cfg
import math

DATA_DIR = "data/"


class HybridPipe(dali.pipeline.Pipeline):
    def __init__(self, train, dali_cpu=False):
        super(HybridPipe, self).__init__(cfg.FLAGS.bs, cfg.FLAGS.workers, cfg.FLAGS.local_rank)
        sz = cfg.FLAGS.sz
        data_dir = DATA_DIR + "320/" if sz < 224 and train else DATA_DIR + "raw-data/"
        data_dir += "train/" if train else "validation/"
        # only shuffle train data
        self.input = dali.ops.FileReader(
            file_root=data_dir,
            random_shuffle=train,
            shard_id=cfg.FLAGS.local_rank,
            num_shards=cfg.FLAGS.world_size,
            read_ahead=True,
        )

        if train:
            self.decode = dali.ops.ImageDecoderRandomCrop(
                device="cpu" if dali_cpu else "mixed",
                output_type=dali.types.RGB,
                random_aspect_ratio=[0.75, 1.25],
                random_area=[cfg.FLAGS.min_area, 1.0],
                num_attempts=100,
            )
            # resize doesn't preserve aspect ratio on purpose
            # works much better with INTERP_TRIANGULAR
            self.resize = dali.ops.Resize(
                device="cpu" if dali_cpu else "gpu",
                interp_type=dali.types.INTERP_TRIANGULAR,
                resize_x=sz,
                resize_y=sz,
            )
        else:
            self.decode = dali.ops.ImageDecoder(device="mixed", output_type=dali.types.RGB)
            # 14% bigger and dividable by 16 then center crop
            crop_size = math.ceil((sz * 1.14 + 8) // 16 * 16)
            self.resize = dali.ops.Resize(
                device="gpu",
                interp_type=dali.types.INTERP_TRIANGULAR,
                resize_shorter=crop_size,
            )

        self.ctwist = dali.ops.ColorTwist(device="gpu")
        self.jitter = dali.ops.Jitter(device="gpu")
        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dali.types.FLOAT,
            crop=(sz, sz),
            image_type=dali.types.RGB,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            output_layout=dali.types.NCHW,
        )
        self.coin = dali.ops.CoinFlip()
        self.rng1 = dali.ops.Uniform(range=[0, 1])
        self.rng2 = dali.ops.Uniform(range=[0.85, 1.15])
        self.rng3 = dali.ops.Uniform(range=[-15, 15])
        self.train = train
        self.dali_cpu = dali_cpu

    def define_graph(self):
        # Read images and labels
        images, labels = self.input(name="Reader")

        # Decode and augmentation
        images = self.decode(images)
        images = self.resize(images)
        if self.dali_cpu:
            images = images.gpu()
        if self.train:
            if cfg.FLAGS.ctwist:
                # always improves quiality slightly
                images = self.ctwist(
                    images,
                    saturation=self.rng2(),
                    contrast=self.rng2(),
                    brightness=self.rng2(),
                    hue=self.rng3(),
                )
            # images = self.jitter(images, mask=self.coin())
            images = self.normalize(
                images, mirror=self.coin(), crop_pos_x=self.rng1(), crop_pos_y=self.rng1()
            )
        else:
            images = self.normalize(images)

        return images, labels.gpu()


class DALIWrapper:
    """Wrap dali to look like torch dataloader"""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return self.loader._size // self.loader.batch_size

    def __iter__(self):
        return (
            (batch[0]["data"], batch[0]["label"].squeeze().long()) for batch in self.loader
        )


def get_loader(train):
    """Returns train or val iterator over Imagenet data"""
    pipe = HybridPipe(train=train)
    pipe.build()
    loader = DALIClassificationIterator(
        pipe,
        size=pipe.epoch_size("Reader") / cfg.FLAGS.world_size,
        auto_reset=True,
        fill_last_batch=train,  # want real accuracy on validiation
        last_batch_padded=True,  # want epochs to have the same length
    )
    return DALIWrapper(loader)
