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
                 train,
                 min_area=0.1):

        super(HybridPipe, self).__init__(bs, num_threads, device_id)
        # only shuffle train data
        self.input = dali.ops.FileReader(file_root=data_dir, random_shuffle=train)

        if train:
            self.decode = dali.ops.ImageDecoderRandomCrop(
                device="mixed",
                output_type=dali.types.RGB,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[min_area, 1.0],
                num_attempts=100)
        else:
            self.decode = dali.ops.ImageDecoder(
                device="mixed",
                output_type=dali.types.RGB)
        # works much better with INTERP_TRIANGULAR 
        self.resize = dali.ops.Resize(device='gpu', interp_type=dali.types.INTERP_TRIANGULAR,
                                      resize_shorter=int(sz*1.14))
        
        self.ctwist = dali.ops.ColorTwist(device = "gpu")
        self.jitter = dali.ops.Jitter(device ="gpu")
        self.normalize = dali.ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=dali.types.FLOAT,
            crop=(sz, sz),
            image_type=dali.types.RGB,
            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
            std=[0.229 * 255,0.224 * 255,0.225 * 255],
            output_layout=dali.types.NCHW) 
        self.coin = dali.ops.CoinFlip()
        self.rng1 = dali.ops.Uniform(range=[0,1])
        self.rng2 = dali.ops.Uniform(range=[0.8,1.2])
        self.rng3 = dali.ops.Uniform(range=[-0.5,0.5])
        self.train = train

    def define_graph(self):
        # Read images and labels
        images, labels = self.input(name="Reader")
        
        # Decode and augmentation
        images = self.decode(images)
        images = self.resize(images)
        if self.train:
            images = self.ctwist(images, 
                                saturation=self.rng1(), 
                                contrast=self.rng2(),
                                brightness=self.rng2(),
                                hue=self.rng3())
            images = self.jitter(images, mask=self.coin())
            images = self.normalize(images, mirror=self.coin(), 
                                    crop_pos_x=self.rng1(), crop_pos_y=self.rng1())
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

def get_loader(sz, bs, workers, device_id, train, min_area=0.1):
    # gives lower performance
    #if int(sz*1.14) <= 160:
    #    data_dir = DATA_DIR + '160/'
    #elif int(sz*1.14) <= 292:
    #    data_dir = DATA_DIR + '292/'
    #else:
    data_dir = DATA_DIR
    data_dir = data_dir + 'train/' if train else data_dir + 'validation/'
    print(data_dir)
    pipe = HybridPipe(
        data_dir=data_dir,
        sz=sz, bs=bs, num_threads=workers,
        device_id=device_id, train=train, min_area=min_area)
    pipe.build()
    loader = DALIClassificationIterator(pipe, size=pipe.epoch_size('Reader'), auto_reset=True)
    return DALIWrapper(loader)