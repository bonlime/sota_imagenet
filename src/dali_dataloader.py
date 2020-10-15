"""Dali dataloader for imagenet"""
import math
import numpy as np
from PIL import Image
from pathlib import Path
import nvidia.dali as dali
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from pytorch_tools.utils.misc import env_rank, env_world_size

ROOT_DATA_DIR = "data/"  # images should be mounted or linked to data/ folder inside this repo


class ValRectExternalInputIterator:
    """EII for making rectangle crops using Dali
    
    All images are first sorted by aspect ratio and then are window averaged with
    window of size BS so that all images in the batch have the same aspect ratio
    
    """

    def __init__(self, batch_size=32, size=224):
        data_dir = Path(ROOT_DATA_DIR) / "raw-data" / "val"
        # according to the docs DALI File Reader does the same
        folder_names = sorted([i.name for i in data_dir.iterdir() if i.is_dir()])
        FOLDER_TO_CLASS = {f: i for i, f in enumerate(folder_names)}
        # get AR for all validation images
        all_images = np.array(list(data_dir.glob("*/*.JPEG")))
        images_size = np.array([Image.open(i).size for i in all_images])
        images_ar = images_size[:, 0] / images_size[:, 1]  # aspect ratios

        # sort by AR and take moving average with window of size BS
        self.sorted_ar = images_ar[np.argsort(images_ar)].round(2)  # round to have nicer prints
        sorted_images = all_images[np.argsort(images_ar)]
        self.sorted_labels = np.array([FOLDER_TO_CLASS[i.parent.name] for i in sorted_images])
        for i in range(0, len(self.sorted_ar), batch_size):
            self.sorted_ar[i : i + batch_size] = self.sorted_ar[i : i + batch_size].mean()
        self.sorted_images = list(map(str, sorted_images))  # turn to str to avoid doing it later

        # account for world size and rank
        local_rank = env_rank()
        per_shard = len(self.sorted_images) // env_world_size()  # number of images in this process
        self.sorted_images = self.sorted_images[local_rank * per_shard : (local_rank + 1) * per_shard]
        self.sorted_labels = self.sorted_labels[local_rank * per_shard : (local_rank + 1) * per_shard]
        self.sorted_ar = self.sorted_ar[local_rank * per_shard : (local_rank + 1) * per_shard]

        self.batch_size = batch_size
        self.size = size
        self.i = 0
        self.n = len(self.sorted_images)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n

    def __next__(self):
        imgs = []
        labels = []
        resize_shapes = []
        for _ in range(self.batch_size):
            img_f = open(self.sorted_images[self.i], "rb")
            imgs.append(np.frombuffer(img_f.read(), dtype=np.uint8))
            labels.append(self.sorted_labels[self.i])
            # resize shortest size to `size`
            ar = self.sorted_ar[self.i]
            if ar <= 1:
                h, w = (self.size + 8) // 16 * 16, (self.size / ar + 8) // 16 * 16
            else:
                h, w = (self.size * ar + 8) // 16 * 16, (self.size + 8) // 16 * 16
            resize_shapes.append(np.array([w, h], dtype=np.float32))
            self.i = (self.i + 1) % self.n
        return (imgs, labels, resize_shapes)


class ValRectPipe(Pipeline):
    """Loader which returns images with AR almost the same as original, minimizing loss of information"""

    def __init__(self, bs=25, workers=4, sz=224, resize_method="linear"):
        super().__init__(bs, workers, env_rank(), seed=42)
        external_iterator = ValRectExternalInputIterator(batch_size=bs, size=sz)
        self.source = ops.ExternalSource(source=external_iterator, num_outputs=3)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        interp_type = types.INTERP_TRIANGULAR if resize_method == "linear" else types.INTERP_CUBIC
        self.resize = ops.Resize(device="gpu", interp_type=interp_type)
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=types.FLOAT,
            output_layout=types.NCHW,
        )

    def define_graph(self):
        image, label, resize_shape = self.source()
        image = self.decode(image)
        image = self.resize(image, size=resize_shape)
        image = self.normalize(image)
        return image, label.gpu()


class ValRectLoader:
    """Wrap dali to look like torch dataloader"""

    def __init__(self, bs=32, workers=4, sz=224, resize_method="linear"):
        """Returns train or val iterator over Imagenet data"""
        pipe = ValRectPipe(bs=bs, workers=workers, sz=sz, resize_method=resize_method)
        pipe.build()
        self.loader = DALIClassificationIterator(
            pipe, size=50000 // env_world_size(), auto_reset=True, fill_last_batch=False, dynamic_shape=True
        )

    def __len__(self):
        return math.ceil(self.loader._size / self.loader.batch_size)

    def __iter__(self):
        return ((batch[0]["data"], batch[0]["label"].squeeze().long()) for batch in self.loader)


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
        use_tfrecords=False,
    ):

        local_rank, world_size = env_rank(), env_world_size()
        super(HybridPipe, self).__init__(bs, workers, local_rank, seed=42)

        # only shuffle train data
        if use_tfrecords:
            records_files_f = ROOT_DATA_DIR + "/tfrecords/"
            records_files_f += "train-{:05d}-of-01024" if train else "validation-{:05d}-of-00128"
            records_files = [records_files_f.format(i) for i in range(1024 if train else 128)]
            idx_files_f = ROOT_DATA_DIR + "/record_idxs/"
            idx_files_f += "train-{:05d}-of-01024" if train else "validation-{:05d}-of-00128"
            idx_files = [idx_files_f.format(i) for i in range(1024 if train else 128)]
            self.input = dali.ops.TFRecordReader(
                path=records_files,
                index_path=idx_files,
                random_shuffle=train,
                initial_fill=10000,  # generate a lot of random numbers in advance
                features={
                    "image/class/label": dali.tfrecord.FixedLenFeature([], dali.tfrecord.int64, -1),
                    "image/filename": dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                    "image/encoded": dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                },
            )
        else:
            data_dir = ROOT_DATA_DIR + "320/" if sz < 224 and train else ROOT_DATA_DIR + "raw-data/"
            data_dir += "train/" if train else "val/"
            self.input = ops.FileReader(
                file_root=data_dir, random_shuffle=train, shard_id=local_rank, num_shards=world_size,
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
            # default Imagenet crop. 14% bigger and dividable by 16 then center crop
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
            dtype=types.FLOAT,
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
            images = self.normalize(images, mirror=self.coin(), crop_pos_x=self.rng1(), crop_pos_y=self.rng1())
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
        classes_divisor=1,  # reduce number of classes by // cls_div
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
            use_tfrecords=use_tfrecords,
        )
        pipe.build()
        self.loader = DALIClassificationIterator(
            pipe, reader_name="Reader", auto_reset=True, fill_last_batch=train,  # want real accuracy on validiation
        )
        self.cls_div = classes_divisor

    def __len__(self):
        return math.ceil(self.loader._size / self.loader.batch_size)

    def __iter__(self):
        # fmt: off
        return ((batch[0]["data"], batch[0]["label"].squeeze().long() // self.cls_div) for batch in self.loader)
        # fmt: on
