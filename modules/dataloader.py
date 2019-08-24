import os.path
import argparse
import os
import shutil
import time
import warnings
from pathlib import Path
import numpy as np
import sys
import math

import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.sampler import Sampler
import torchvision
import pickle
from tqdm import tqdm

# kinda slow
def get_loaders(traindir, valdir, sz, bs, val_bs=None, workers=8, rect_val=False, rect_train=False, min_scale=0.15, distributed=False):
    val_bs = val_bs or bs
    train_dtst, train_sampler = create_dataset(traindir, bs, sz, rect_train, distributed, True)
    train_loader = torch.utils.data.DataLoader(
        train_dtst, 
        num_workers=workers, pin_memory=True, collate_fn=fast_collate,
        batch_sampler=train_sampler)

    val_dtst, val_sampler = create_dataset(valdir, bs, sz, rect_val, distributed, False)
    val_loader = torch.utils.data.DataLoader(
        val_dtst,
        num_workers=workers, pin_memory=True, collate_fn=fast_collate,
        batch_sampler=val_sampler)

    train_loader = BatchTransformDataLoader(train_loader)
    val_loader = BatchTransformDataLoader(val_loader)

    return train_loader, val_loader, train_sampler, val_sampler


def create_dataset(imgs_dir, batch_size, target_size, rect, distributed, train):
    idx_sorted = None
    if train:
        start_transform = [
            transforms.RandomResizedCrop(target_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        start_transform = [transforms.Resize(int(target_size*1.14)), ]
    if rect:
        idx_ar_sorted = sort_ar(imgs_dir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, batch_size)
        end_transform = [CropArTfm(idx2ar, target_size)]
    else:
        end_transform = [transforms.CenterCrop(target_size)]
    dtst = RectDataset(imgs_dir, transform=start_transform + end_transform)
    idxs = idx_sorted or list(range(len(dtst)))
    sampler = DistRectSampler(idxs, batch_size=batch_size, distributed=distributed)
    return dtst, sampler


class BatchTransformDataLoader():
    # Mean normalization on batch level instead of individual
    # https://github.com/NVIDIA/apex/blob/59bf7d139e20fb4fa54b09c6592a2ff862f3ac7f/examples/imagenet/main.py#L222
    def __init__(self, loader, fp16=True):
        self.loader = loader
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)

    def __len__(self): return len(self.loader)

    def process_tensors(self, input, target, non_blocking=True):
        input = input.cuda(non_blocking=non_blocking)
        input = input.float()
        if len(input.shape) < 3:
            return input, target.cuda(non_blocking=non_blocking)
        return input.sub_(self.mean).div_(self.std), target.cuda(non_blocking=non_blocking)

    def update_batch_size(self, bs):
        self.loader.batch_sampler.batch_size = bs

    def __iter__(self):
        return (self.process_tensors(input, target, non_blocking=True) for input, target in self.loader)

def fast_collate(batch):
    if not batch:
        return torch.tensor([]), torch.tensor([])
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets

class RectDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            for tfm in self.transform:
                if isinstance(tfm, CropArTfm):
                    sample = tfm(sample, index)
                else:
                    sample = tfm(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class DistRectSampler(Sampler):
    # DistValSampler distrbutes batches equally (based on batch size) to every gpu (even if there aren't enough images)
    # WARNING: Some baches will contain an empty array to signify there aren't enough images
    # Distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = env_world_size()
            self.global_rank = env_rank()
        else:
            self.global_rank = 0
            self.world_size = 1

        # expected number of batches per sample. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_batches = math.ceil(len(self.indices) / self.world_size / self.batch_size)

        # num_samples = total images / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_batches * self.batch_size

    def __iter__(self):
        offset = self.num_samples * self.global_rank
        sampled_indices = self.indices[offset:offset+self.num_samples]
        # shufle offset idxs so that batch shape is random rather than increasing
        idxs = list(range(self.expected_num_batches))
        #np.random.shuffle(idxs)
        for i in idxs:
            offset = i*self.batch_size
            yield sampled_indices[offset:offset+self.batch_size]

    def __len__(self): return self.expected_num_batches
    def set_epoch(self, epoch): return


class CropArTfm(object):
    def __init__(self, idx2ar, target_size):
        self.idx2ar, self.target_size = idx2ar, target_size

    def __call__(self, img, idx):
        target_ar = self.idx2ar[idx]
        if target_ar < 1:
            w = int(self.target_size/target_ar)
            size = (w//8*8, self.target_size)
        else:
            h = int(self.target_size*target_ar)
            size = (self.target_size, h//8*8)
        return torchvision.transforms.functional.center_crop(img, size)


def sort_ar(imgs_dir):
    idx2ar_file = imgs_dir+'/../sorted_idxar.p'
    if os.path.isfile(idx2ar_file):
        return pickle.load(open(idx2ar_file, 'rb'))
    print('Creating AR indexes. Please be patient this may take a couple minutes...')
    dataset = datasets.ImageFolder(imgs_dir)
    sizes = []
    for path in tqdm(dataset.samples, total=len(dataset)):
        with PIL.Image.open(path[0]) as fp:
            sizes.append(fp.size)
    idx_ar = [(i, round(s[0]/s[1], 5)) for i, s in enumerate(sizes)]
    sorted_idxar = sorted(idx_ar, key=lambda x: x[1])
    pickle.dump(sorted_idxar, open(idx2ar_file, 'wb'))
    print('Done')
    return sorted_idxar


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def map_idx2ar(idx_ar_sorted, batch_size):
    ar_chunks = list(chunks(idx_ar_sorted, batch_size))
    idx2ar = {}
    for chunk in ar_chunks:
        idxs, ars = list(zip(*chunk))
        mean = round(np.mean(ars), 5)
        for idx in idxs:
            idx2ar[idx] = mean
    return idx2ar


class DataManager():
    def __init__(self, phases):
        self.phases = self.preload_phase_data(phases)
        
    def set_epoch(self, epoch):
        cur_phase = self.get_phase(epoch)
        if cur_phase: self.set_data(cur_phase)
        if hasattr(self.trn_smp, 'set_epoch'): self.trn_smp.set_epoch(epoch)
        if hasattr(self.val_smp, 'set_epoch'): self.val_smp.set_epoch(epoch)

    def get_phase(self, epoch):
        return next((p for p in self.phases if p['ep'] == epoch), None)

    def set_data(self, phase):
        """Initializes data loader."""
        if phase.get('keep_dl', False):
            log.event('Batch size changed: {}'.format(phase["bs"]))
            tb.log_size(phase['bs'])
            self.trn_dl.update_batch_size(phase['bs'])
            return
        
        log.event('Dataset changed.\nImage size: {}\nBatch size: {}\n Train Directory: {}\nValidation Directory: {}'.format(
            phase["sz"], phase["bs"], phase["trndir"], phase["valdir"]))
        tb.log_size(phase['bs'], phase['sz'])

        self.trn_dl, self.val_dl, self.trn_smp, self.val_smp = phase['data']
        self.phases.remove(phase)

        # clear memory before we begin training
        gc.collect()
        
    def preload_phase_data(self, phases):
        for phase in phases:
            if not phase.get('keep_dl', False):
                self.expand_directories(phase)
                phase['data'] = self.preload_data(**phase)
        return phases

    def expand_directories(self, phase):
        trndir = phase.get('trndir', '')
        valdir = phase.get('valdir', trndir)
        phase['trndir'] = args.data+trndir+'/train'
        phase['valdir'] = args.data+valdir+'/validation'

    def preload_data(self, ep, sz, bs, trndir, valdir, **kwargs): # dummy ep var to prevent error
        if 'lr' in kwargs: del kwargs['lr']  # in case we mix schedule and data phases
        if 'mom' in kwargs: del kwargs['mom'] # in case we mix schedule and data phases
        """Pre-initializes data-loaders. Use set_data to start using it."""
        if sz == 128: val_bs = max(bs, 512)
        elif sz == 224: val_bs = max(bs, 256)
        else: val_bs = max(bs, 128)
        return dataloader.get_loaders(trndir, valdir, bs=bs, val_bs=val_bs, sz=sz, workers=args.workers, distributed=args.distributed, **kwargs)