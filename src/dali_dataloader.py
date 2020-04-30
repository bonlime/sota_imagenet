"""Dali dataloader for imagenet"""
import math
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from pytorch_tools.utils.misc import env_rank, env_world_size

DATA_DIR = "data/"

class HybridPipe(Pipeline):
    def __init__(self, train=False, bs=32, workers=4, sz=224, ctwist=True, dali_cpu=False, min_area=0.08):

        local_rank, world_size = env_rank(), env_world_size()
        super(HybridPipe, self).__init__(bs, workers, local_rank, seed=42)
        data_dir = DATA_DIR + "320/" if sz < 224 and train else DATA_DIR + "raw-data/"
        data_dir += "train/" if train else "val/"
        # only shuffle train data
        self.input = ops.FileReader(
            file_root=data_dir,
            random_shuffle=train,
            shard_id=local_rank,
            num_shards=world_size,
            # read_ahead=True,
        )

        if train:
            self.decode = ops.ImageDecoderRandomCrop(
                output_type=types.RGB,
                device="cpu" if dali_cpu else "mixed",
                random_aspect_ratio=[0.75, 1.25],
                random_area=[min_area, 1.0],
                num_attempts=100,
            )
            # resize doesn't preserve aspect ratio on purpose
            # works much better with INTERP_TRIANGULAR
            self.resize = ops.Resize(
                device="cpu" if dali_cpu else "gpu",
                interp_type=types.INTERP_TRIANGULAR,
                resize_x=sz,
                resize_y=sz,
            )
        else:
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            # 14% bigger and dividable by 16 then center crop
            crop_size = math.ceil((sz * 1.14 + 8) // 16 * 16)
            self.resize = ops.Resize(
                device="gpu", interp_type=types.INTERP_TRIANGULAR, resize_shorter=crop_size,
            )
        # color augs
        self.contrast = ops.BrightnessContrast(device="gpu")
        self.hsv = ops.Hsv(device="gpu")
        self.rgb2bgr = ops.ColorSpaceConversion(image_type=types.RGB, output_type=types.BGR, device="gpu")
        self.rgb2gray = ops.ColorSpaceConversion(image_type=types.RGB, output_type=types.GRAY, device="gpu")
        self.gray2rgb = ops.ColorSpaceConversion(image_type=types.GRAY, output_type=types.RGB, device="gpu")

        self.jitter = ops.Jitter(device="gpu")
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            crop=(sz, sz),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            image_type=types.RGB,
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
        )
        self.coin = ops.CoinFlip()
        self.coin03 = ops.CoinFlip(probability=0.3)

        self.rng1 = ops.Uniform(range=[0, 1])
        self.rng2 = ops.Uniform(range=[0.85, 1.15])
        self.rng3 = ops.Uniform(range=[-15, 15])
        self.train = train
        self.dali_cpu = dali_cpu
        self.ctwist = ctwist

    def define_graph(self):
        # Read images and labels
        images, labels = self.input(name="Reader")

        # Decode and augmentation
        images = self.decode(images)
        images = self.resize(images)
        if self.dali_cpu:
            images = images.gpu()
        if self.train:
            if self.ctwist:
                if self.coin03():
                    images = self.rgb2bgr(images)
                if self.coin03():
                    images = self.rgb2gray(images)
                    images = self.gray2rgb(images)
                images = self.contrast(
                    images, contrast=self.rng2(), brightness=self.rng2()
                )
                images = self.hsv(
                    images, hue=self.rng3(), saturation=self.rng2(), value=self.rng2(),
                )
            # images = self.jitter(images, mask=self.coin())
            images = self.normalize(
                images, mirror=self.coin(), crop_pos_x=self.rng1(), crop_pos_y=self.rng1()
            )
        else:
            images = self.normalize(images)

        return images, labels.gpu()


class DaliLoader:
    """Wrap dali to look like torch dataloader"""

    def __init__(self, train=False, bs=32, workers=4, sz=224, ctwist=True, min_area=0.08):
        """Returns train or val iterator over Imagenet data"""
        pipe = HybridPipe(train=train, bs=bs, workers=workers, sz=sz, ctwist=ctwist, min_area=min_area)
        pipe.build()
        self.loader = DALIClassificationIterator(
            pipe,
            size=pipe.epoch_size("Reader") / env_world_size(),
            auto_reset=True,
            fill_last_batch=train,  # want real accuracy on validiation
            last_batch_padded=True,  # want epochs to have the same length
        )

    def __len__(self):
        return math.ceil(self.loader._size / self.loader.batch_size)

    def __iter__(self):
        return ((batch[0]["data"], batch[0]["label"].squeeze().long()) for batch in self.loader)
