"""Dali dataloader for imagenet"""
import math
import nvidia.dali as dali
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from pytorch_tools.utils.misc import env_rank, env_world_size


class HybridPipe(Pipeline):
    def __init__(
        self,
        train=False,
        bs=32,
        workers=4,
        sz=224,
        ctwist=True,
        min_area=0.08,
        resize_method="linear",
        crop_method="",
        data_dir="data/",
        use_tfrecords=False,
    ):

        local_rank, world_size = env_rank(), env_world_size()
        super(HybridPipe, self).__init__(bs, workers, local_rank, seed=42)
        
        # only shuffle train data
        if use_tfrecords:
            records_files_f = data_dir + "/tfrecords/"
            records_files_f += "train-{:05d}-of-01024" if train else "validation-{:05d}-of-00128"
            records_files = [records_files_f.format(i) for i in range(1024 if train else 128)]
            idx_files_f = data_dir + "/record_idxs/"
            idx_files_f += "train-{:05d}-of-01024" if train else "validation-{:05d}-of-00128"
            idx_files = [idx_files_f.format(i) for i in range(1024 if train else 128)]
            self.input = dali.ops.TFRecordReader(
                path=records_files,
                index_path=idx_files,
                random_shuffle=train,
                initial_fill=10000,  # generate a lot of random numbers in advance	
                features={
                    'image/class/label': dali.tfrecord.FixedLenFeature([], dali.tfrecord.int64,  -1),
                    'image/filename': dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                    'image/encoded': dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                }
            )	
        else: 
            data_dir += "320/" if sz < 224 and train else "raw-data/"
            data_dir += "train/" if train else "val/"
            self.input = ops.FileReader(
                file_root=data_dir,
                random_shuffle=train,
                shard_id=local_rank,
                num_shards=world_size,
            )
        interp_type = types.INTERP_TRIANGULAR if resize_method == "linear" else types.INTERP_CUBIC
        if train:
            self.decode = ops.ImageDecoderRandomCrop(
                output_type=types.RGB,
                device="mixed",
                random_aspect_ratio=[0.75, 1.25],
                random_area=[min_area, 1.0],
                num_attempts=100,
            )
            # resize doesn't preserve aspect ratio on purpose
            # works much better with INTERP_TRIANGULAR
            self.resize = ops.Resize(device="gpu", interp_type=interp_type, resize_x=sz, resize_y=sz,)
        else:
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
            if crop_method == "full":
                crop_size = sz
            else:
                # 14% bigger and dividable by 16 then center crop
                crop_size = math.ceil((sz * 1.14 + 8) // 16 * 16)
            self.resize = ops.Resize(device="gpu", interp_type=interp_type, resize_shorter=crop_size,)
        # color augs
        self.contrast = ops.BrightnessContrast(device="gpu")
        self.hsv = ops.Hsv(device="gpu")

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
        # jitter is a very strong aug want to have it rarely
        self.coin_jitter = ops.CoinFlip(probability=0.1)

        self.rng1 = ops.Uniform(range=[0, 1])
        self.rng2 = ops.Uniform(range=[0.85, 1.15])
        self.rng3 = ops.Uniform(range=[-15, 15])
        self.train = train
        self.use_tfrecords = use_tfrecords
        self.ctwist = ctwist

    def define_graph(self):
        # Read images and labels
        if self.use_tfrecords:
            inputs = self.input(name="Reader")
            images = inputs["image/encoded"]
            labels = inputs["image/class/label"]
        else:
            images, labels = self.input(name="Reader")

        # Decode and augmentation
        images = self.decode(images)
        # remove jitter for now. turn on for longer schedules for stronger regularization
        # if self.train and self.ctwist:
        #    # want to jitter before resize so that following op smoothes the jitter
        #    images = self.jitter(images, mask=self.coin_jitter())
        images = self.resize(images)

        if self.train:
            if self.ctwist:
                images = self.contrast(images, contrast=self.rng2(), brightness=self.rng2())
                images = self.hsv(images, hue=self.rng3(), saturation=self.rng2(), value=self.rng2())
            images = self.normalize(
                images, mirror=self.coin(), crop_pos_x=self.rng1(), crop_pos_y=self.rng1()
            )
        else:
            images = self.normalize(images)

        return images, labels.gpu()


class DaliLoader:
    """Wrap dali to look like torch dataloader"""

    def __init__(
        self,
        train=False,
        bs=32,
        workers=4,
        sz=224,
        ctwist=True,
        min_area=0.08,
        resize_method="linear",
        crop_method="",
        classes_divisor=1,  # reduce number of classes by // cls_div
        data_dir="data/",
        use_tfrecords=False,
    ):
        """Returns train or val iterator over Imagenet data"""
        pipe = HybridPipe(
            train=train,
            bs=bs,
            workers=workers,
            sz=sz,
            ctwist=ctwist,
            min_area=min_area,
            resize_method=resize_method,
            crop_method=crop_method,
            data_dir=data_dir,
            use_tfrecords=use_tfrecords,
        )
        pipe.build()
        self.loader = DALIClassificationIterator(
            pipe,
            size=pipe.epoch_size("Reader") / env_world_size(),
            auto_reset=True,
            fill_last_batch=train,  # want real accuracy on validiation
            last_batch_padded=True,  # want epochs to have the same length
        )
        self.cls_div = classes_divisor

    def __len__(self):
        return math.ceil(self.loader._size / self.loader.batch_size)

    def __iter__(self):
        # fmt: off
        return ((batch[0]["data"], batch[0]["label"].squeeze().long() // self.cls_div) for batch in self.loader)
        # fmt: on
