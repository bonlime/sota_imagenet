"""Dali dataloader for imagenet"""
import math
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
import nvidia.dali as dali
from nvidia.dali import ops
from nvidia.dali import types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
from pytorch_tools.utils.misc import env_rank, env_world_size

ROOT_DATA_DIR = "data/"  # images should be mounted or linked to data/ folder inside this repo
RESIZE_DICT = {
    "nn": types.INTERP_NN,
    "linear": types.INTERP_LINEAR,
    "triang": types.INTERP_TRIANGULAR,
    "cubic": types.INTERP_CUBIC,
    "lancz": types.INTERP_LANCZOS3,
}


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

    def __init__(self, bs=25, workers=4, sz=224, resize_method="triang"):
        super().__init__(bs, workers, env_rank(), seed=42)
        external_iterator = ValRectExternalInputIterator(batch_size=bs, size=sz)
        self.source = ops.ExternalSource(source=external_iterator, num_outputs=3)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", interp_type=RESIZE_DICT[resize_method])
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

    def __init__(self, bs=32, workers=4, sz=224, resize_method="triang"):
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


def mix(condition, true_case, false_case):
    """multiplex between two images"""
    neg_condition = condition ^ True
    return condition * true_case + neg_condition * false_case


class DefaultPipe(Pipeline):
    def __init__(
        self,
        train=False,
        bs=32,
        workers=4,
        sz=224,
        ctwist=True,
        jitter=False,
        blur=False,
        min_area=0.08,
        resize_method="triang",
        crop_method="default",
        random_interpolation=False,  # if given ignores `resize_method` and uses random
        use_tfrecords=False,
    ):

        local_rank, world_size = env_rank(), env_world_size()
        super().__init__(bs, workers, local_rank, seed=42)

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
        self.train_decode = ops.ImageDecoderRandomCrop(
            output_type=types.RGB,
            device="mixed",
            random_aspect_ratio=[0.75, 1.25],
            random_area=[min_area, 1.0],
            num_attempts=100,
        )
        self.val_decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        # train resize doesn't preserve aspect ratio on purpose
        # works much better with INTERP_TRIANGULAR
        self.train_resize = ops.Resize(device="gpu", interp_type=RESIZE_DICT[resize_method], resize_x=sz, resize_y=sz)
        # need all types of resize to support random interpolation
        self.resize_lin = ops.Resize(device="gpu", interp_type=types.INTERP_LINEAR, resize_x=sz, resize_y=sz)
        self.resize_tri = ops.Resize(device="gpu", interp_type=types.INTERP_TRIANGULAR, resize_x=sz, resize_y=sz)
        self.resize_cub = ops.Resize(device="gpu", interp_type=types.INTERP_CUBIC, resize_x=sz, resize_y=sz)
        self.resize_lan = ops.Resize(device="gpu", interp_type=types.INTERP_LANCZOS3, resize_x=sz, resize_y=sz)

        # default Imagenet crop. 14% bigger and dividable by 16 then center crop
        crop_size = math.ceil((sz * 1.14 + 8) // 16 * 16) if crop_method == "default" else sz
        self.val_resize = ops.Resize(device="gpu", interp_type=RESIZE_DICT[resize_method], resize_shorter=crop_size)
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            crop=(sz, sz),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            dtype=types.FLOAT,
            output_layout=types.NCHW,
        )
        # additional train augs
        self.contrast = ops.BrightnessContrast(device="gpu")
        self.hsv = ops.Hsv(device="gpu")
        self.jitter_op = ops.Jitter(device="gpu")
        self.coin = ops.CoinFlip()
        self.bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.blur_op = ops.GaussianBlur(device="gpu", window_size=11)  # 15 is just a random const
        # jitter is a very strong aug want to have it rarely
        self.coin_jitter = ops.CoinFlip(probability=0.2)
        self.rng1 = ops.Uniform(range=[0, 1])  # for crop
        self.rng2 = ops.Uniform(range=[0.85, 1.15])  # for color augs
        self.rng3 = ops.Uniform(range=[-15, 15])  # for hue
        self.blur_sigma_rng = ops.Uniform(range=[0.8, 1.4])  # default sigma for WS=11 is 1.4 want it also to be random
        self.train = train
        self.use_tfrecords = use_tfrecords
        self.ctwist = ctwist
        self.use_jitter = jitter
        self.use_blur = blur
        self.random_interpolation = random_interpolation

    def _train_resize(self, images):
        if not self.random_interpolation:
            return self.train_resize(images)
        lin_tri = mix(self.bool(self.coin()), self.resize_lin(images), self.resize_tri(images))
        cub_lan = mix(self.bool(self.coin()), self.resize_cub(images), self.resize_lan(images))
        return mix(self.bool(self.coin()), lin_tri, cub_lan)

    def define_graph(self):
        # Read images and labels
        if self.use_tfrecords:
            inputs = self.input(name="Reader")
            images = inputs["image/encoded"]
            labels = inputs["image/class/label"]
        else:
            images, labels = self.input(name="Reader")

        # Decode and augmentation
        if self.train:
            images = self.train_decode(images)
            if self.use_jitter:  # want to jitter before resize so that following op smoothes the jitter
                images = self.jitter_op(images, mask=self.coin_jitter())
            images = self._train_resize(images)
            if self.use_blur:  # optional 50% blur. use it after resize so that it's more determinitstic
                images = mix(self.bool(self.coin()), images, self.blur_op(images, sigma=self.blur_sigma_rng()))
            if self.ctwist:
                images = self.contrast(images, contrast=self.rng2(), brightness=self.rng2())
                images = self.hsv(images, hue=self.rng3(), saturation=self.rng2(), value=self.rng2())
            images = self.normalize(images, mirror=self.coin(), crop_pos_x=self.rng1(), crop_pos_y=self.rng1())
        else:
            images = self.val_decode(images)
            images = self.val_resize(images)
            images = self.normalize(images)

        return images, labels.gpu()


class FixMatchPipe(DefaultPipe):
    """pipe which returns augmented and non augmented versions of the same image"""

    def define_graph(self):
        # Read images and labels
        if self.use_tfrecords:
            inputs = self.input(name="Reader")
            images = inputs["image/encoded"]
            labels = inputs["image/class/label"]
        else:
            images, labels = self.input(name="Reader")
        ## Process val first
        val_images = self.val_decode(images)
        val_images = self.val_resize(val_images)
        val_images = self.normalize(val_images)
        ## Decode and augment train
        images = self.train_decode(images)
        if self.use_jitter:  # want to jitter before resize so that following op smoothes the jitter
            images = self.jitter_op(images, mask=self.coin_jitter())
        images = self._train_resize(images)
        if self.use_blur:  # optional 50% blur
            images = mix(self.bool(self.coin()), images, self.blur_op(images, sigma=self.blur_sigma_rng()))
        if self.ctwist:
            images = self.contrast(images, contrast=self.rng2(), brightness=self.rng2())
            images = self.hsv(images, hue=self.rng3(), saturation=self.rng2(), value=self.rng2())
        images = self.normalize(images, mirror=self.coin(), crop_pos_x=self.rng1(), crop_pos_y=self.rng1())

        return images, val_images, labels.gpu()


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
        resize_method="triang",
        classes_divisor=1,  # reduce number of classes by // cls_div
        use_tfrecords=False,
        crop_method="default",  # one of `default` or `full`
        jitter=False,  # use pixel jitter augmentation
        blur=False,  # optional gaussian blur
        random_interpolation=False,
        fixmatch=False,  # doubles the size of BS by also returning non augmented copies of image
    ):
        """Returns train or val iterator over Imagenet data"""
        fixmatch = fixmatch and train  # only changes train behavior
        pipe = (FixMatchPipe if fixmatch else DefaultPipe)(
            train=train,
            bs=bs,
            workers=workers,
            sz=sz,
            ctwist=ctwist,
            jitter=jitter,
            blur=blur,
            min_area=min_area,
            resize_method=resize_method,
            crop_method=crop_method,
            use_tfrecords=use_tfrecords,
            random_interpolation=random_interpolation,
        )
        pipe.build()
        self.loader = DALIGenericIterator(
            pipe,
            output_map=["data", "val_data", "label"] if fixmatch else ["data", "label"],
            reader_name="Reader",
            auto_reset=True,
            fill_last_batch=train,  # want real accuracy on validiation
        )
        self.cls_div = classes_divisor
        self.fixmatch = fixmatch

    def __len__(self):
        return math.ceil(self.loader._size / self.loader.batch_size)

    def __iter__(self):
        for b in self.loader:  # b = batch
            target = b[0]["label"].squeeze().long() // self.cls_div
            target = torch.cat([target, target]) if self.fixmatch else target
            images = torch.cat([b[0]["data"], b[0]["val_data"]], dim=0) if self.fixmatch else b[0]["data"]
            yield images, target
