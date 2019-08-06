import os
from nvidia import dali
from nvidia.dali.plugin.pytorch import DALIClassificationIterator



class HybridPipe(dali.pipeline.Pipeline):
    def __init__(self,
                 data_dir,
                 sz,
                 bs,
                 num_threads,
                 device_id,
                 train):

        super(HybridPipe, self).__init__(bs, num_threads, device_id)
        self.input = dali.ops.FileReader(file_root=data_dir, random_shuffle=True)

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
        # works much better with INTERP_TRIANGULAR 
        self.resize = dali.ops.Resize(device='gpu', interp_type=dali.types.INTERP_TRIANGULAR,
                                      resize_shorter=int(sz*1.14)) 
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
        images, labels = self.input(name="Reader")
        
        # Decode and augmentation
        images = self.decode(images)
        images = self.resize(images)
        if self.train:
            images = self.normalize(images, mirror=self.mirror(), 
                                    crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        else:
            images = self.normalize(images)

        return images, labels.gpu()


DATA_DIR = '/home/zakirov/datasets/imagenet_2012/raw_data/' 

class DALIWrapper:
    """Wrap dali to look like torch dataloader"""
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return self.loader._size // self.loader.batch_size

    def __iter__(self):
        return ( (batch[0]['data'], batch[0]['label'].squeeze().long()) for batch in self.loader)

def get_loader(sz, bs, workers, device_id, train):
    if int(sz*1.14) <= 160:
        data_dir = DATA_DIR + '160/'
    elif int(sz*1.14) <= 292:
        data_dir = DATA_DIR + '292/'
    else:
        data_dir = DATA_DIR
    data_dir = data_dir + 'train/' if train else data_dir + 'validation/'
    print(data_dir)
    pipe = HybridPipe(
        data_dir=data_dir,
        sz=sz, bs=bs, num_threads=workers,
        device_id=device_id, train=train)
    pipe.build()
    loader = DALIClassificationIterator(pipe, size=pipe.epoch_size('Reader'), auto_reset=True)
    return DALIWrapper(loader)