import os
from nvidia import dali
from nvidia.dali.plugin.pytorch import DALIClassificationIterator



class HybridPipe(dali.pipeline.Pipeline):
    def __init__(self,
                 tfrec_filenames,
                 tfrec_idx_filenames,
                 sz,
                 bs,
                 num_threads,
                 device_id,
                 train):

        super(HybridPipe, self).__init__(bs, num_threads, device_id)
        self.input = dali.ops.TFRecordReader(
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            random_shuffle=True,
            initial_fill=10000,  # generate a lot of random numbers in advance
            features={
                'image/encoded': dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                'image/height': dali.tfrecord.FixedLenFeature([1], dali.tfrecord.int64,  -1),
                'image/width': dali.tfrecord.FixedLenFeature([1], dali.tfrecord.int64,  -1),
                'image/class/label': dali.tfrecord.FixedLenFeature([], dali.tfrecord.int64,  -1),
                'image/class/synset': dali.tfrecord.FixedLenFeature([], dali.tfrecord.string, ''),
                'image/colorspace': dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                'image/channels': dali.tfrecord.FixedLenFeature([1], dali.tfrecord.int64,  -1),
                'image/format': dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, ""),
                'image/filename': dali.tfrecord.FixedLenFeature((), dali.tfrecord.string, "")})

        if train:
            self.decode = dali.ops.nvJPEGDecoderRandomCrop(
                device="mixed",
                output_type=dali.types.RGB,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0], # Maybe can make this value higher during last epochs
                num_attempts=100)
        else:
            self.decode = dali.ops.nvJPEGDecoder(
                device="mixed",
                output_type=dali.types.RGB)
        self.resize = dali.ops.Resize(device='gpu', resize_shorter=int(sz*1.14)) #, resize_x=width, resize_y=height)
        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dali.types.FLOAT,
            crop=(sz, sz),
            image_type=dali.types.RGB,
            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
            std=[0.229 * 255,0.224 * 255,0.225 * 255],
            output_layout=dali.types.NCHW) 
        self.mirror = dali.ops.CoinFlip()
        self.uniform = dali.ops.Uniform(range=[0,1])
        self.train = train

    def define_graph(self):
        # Read images and labels
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"].gpu()
        # Decode and augmentation
        images = self.decode(images)
        images = self.resize(images)
        if self.train:
            images = self.normalize(images, mirror=self.mirror(), 
                                    crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        else:
            images = self.normalize(images)

        return images, labels

RECORDS_DIR = '/home/zakirov/datasets/imagenet_2012/'
IDX_DIR = '/home/zakirov/datasets/imagenet_2012/record_idxs/'

class DALIWrapper(DALIClassificationIterator):
    def __init__(self, *args, **kwargs):
        super(DALIWrapper, self).__init__(*args, **kwargs)

    def __len__(self):
        return self._size // self.batch_size

def get_loader(sz, bs, workers, device_id, train):
    if train:
        tfrecord_f = os.path.join(RECORDS_DIR, 'train', 'train-{:05d}-of-01024')
        idx_f = os.path.join(IDX_DIR, 'train', 'train-{:05d}-of-01024.idx')
        filenames = [tfrecord_f.format(i) for i in range(1024)]
        idx_filenames = [idx_f.format(i) for i in range(1024)]
    else:
        tfrecord_f = os.path.join(RECORDS_DIR, 'validation', 'validation-{:05d}-of-00128')
        idx_f = os.path.join(IDX_DIR, 'validation', 'validation-{:05d}-of-00128.idx')
        filenames = [tfrecord_f.format(i) for i in range(128)]
        idx_filenames = [idx_f.format(i) for i in range(128)]

    pipe = HybridPipe(
        tfrec_filenames=filenames,
        tfrec_idx_filenames=idx_filenames,
        sz=sz, bs=bs, num_threads=workers,
        device_id=device_id, train=train)
    pipe.build()
    loader = DALIWrapper(pipe, size=pipe.epoch_size('Reader'))
    return loader
