import argparse
import os
import shutil
import time
import warnings
import math
from datetime import datetime
from pathlib import Path
import sys
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist

import pytorch_tools.models as models
import pytorch_tools as pt
from pytorch_tools.fit_wrapper.callbacks import PhasesScheduler, Logger
from pytorch_tools.utils.misc import listify
from pytorch_tools.optim import optimizer_from_name

# for fp16
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
import copy

import modules.dist_utils as dist_utils
from modules.dali_dataloader import get_loader
from modules.experimental_utils import bnwd_optim_params
from modules.logger import FileLogger
from modules.phases import LOADED_PHASES


def get_parser():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # parser.add_argument('data', metavar='DIR', help='path to dataset')
    add_arg = parser.add_argument
    add_arg('--arch', '-a', metavar='ARCH', default='resnet18',
            choices=model_names,
            help='model architecture: ' +
            ' | '.join(model_names) +
            ' (default: resnet18)')
    add_arg('--pretrained', dest='pretrained', action='store_true',
            help='use pre-trained model')
    # add_arg('--gpu', type=int, default='0',
    #         help='GPU to use')
    add_arg('--phases', type=str,
            help='Specify epoch order of data resize and learning rate schedule: [{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]')
    add_arg('--load-phases', action='store_true',
            help='Flag to load phases from modules.phases config')
    # add_arg('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    add_arg('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
    add_arg('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
    add_arg('--weight-decay', '--wd', default=1e-4, type=float,
            metavar='W', help='weight decay (default: 1e-4)')
    # TODO actually add this feature
    # add_arg('--init-bn0', action='store_true', help='Intialize running batch norm mean to 0')
    add_arg('--print-freq', '-p', default=5, type=int,
            metavar='N', help='log/print every this many steps (default: 5)')
    add_arg('--no-bn-wd', action='store_true', help='Remove batch norm from weight decay')
    add_arg('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
    add_arg('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
    add_arg('--opt_level', default='O0', type=str, choices=['O0', 'O1', 'O2', 'O3'],
            help='optimizatin level for apex. (default: "00")')
    # add_arg('--distributed', action='store_true', help='Run distributed training. Default True') #infer automaticaly
    # add_arg('--dist-url', default='env://', type=str,
    #                    help='url used to set up distributed training')
    # add_arg('--dist-backend', default='nccl', type=str, help='distributed backend')
    add_arg('--local_rank', default=0, type=int,
            help='Used for multi-process training. Can either be manually set ' +
            'or automatically set by using \'python -m multiproc\'.')
    # TODO write logs into separete folders
    add_arg('--logdir', default='logs', type=str,
            help='where logs go')
    add_arg('-n', '--name', type=str, default='', dest='name',
            help='Name of this run. If empty it would be a timestamp')
    add_arg('--short-epoch', action='store_true',
            help='make epochs short (for debugging)')
    add_arg('--optim', type=str, default='SGD', choices=['sgd', 'sgdw', 'adam', 'adamw', 'rmsprop', 'radam'],
            help='Optimizer to use (default: sgd)')
    add_arg('--optim-params', type=str, default='{}', help='Additional optimizer params as kwargs')
    return parser


# makes it slightly faster
cudnn.benchmark = True
args = get_parser().parse_args()

# detect distributed
args.world_size = int(os.environ.get('WORLD_SIZE', 1))
args.distributed = args.world_size > 1

# Only want master rank logging to tensorboard
# is_master = (not args.distributed) or (args.local_rank == 0) #(dist_utils.env_rank() == 0)
IS_MASTER = args.local_rank == 0
name = args.name or str(datetime.now()).split('.')[0].replace(' ', '_')

# save script and runing comand so we can reproduce from logs
OUTDIR = os.path.join(args.logdir, name)
os.makedirs(OUTDIR, exist_ok=True)
shutil.copy2(os.path.realpath(__file__), '{}'.format(OUTDIR))
with open(OUTDIR + '/run.cmd', 'w') as fp:
    fp.write(' '.join(sys.argv[1:]) + '\n')
PHASES = LOADED_PHASES if args.load_phases else eval(args.phases)
with open(OUTDIR + '/phases.json', 'w') as fp:
    json.dump(PHASES, fp)
print(IS_MASTER)
log = FileLogger(OUTDIR, is_master=IS_MASTER)


def main():
    log.console(args)

    if args.distributed:
        log.console('Distributed initializing process group')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=args.world_size)
        # assert(dist_utils.env_world_size() == dist.get_world_size())
        # log.console("Distributed: success (%d/%d)" % (args.local_rank, dist.get_world_size()))

    log.console("Loading model")
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained='imagenet')
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    model = model.cuda()

    # best_top5 = 93  # only save models over 93%. Otherwise it stops to save every time
    optim_params = bnwd_optim_params(model) if args.no_bn_wd else model.parameters()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # start with 0 lr. Scheduler will change this later
    kwargs = eval(args.optim_params)
    optimizer = optimizer_from_name(args.optim)(optim_params, lr=0, weight_decay=args.weight_decay, **kwargs)

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      loss_scale=1 if args.opt_level == 'O0' else 2048,
                                      max_loss_scale=2.**13,
                                      min_loss_scale=1.,
                                      verbosity=1)

    if args.distributed:
        model = DDP(model, delay_allreduce=True)  # device_ids=[args.local_rank], output_device=args.local_rank)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_top5 = checkpoint['best_top5']
        optimizer.load_state_dict(checkpoint['optimizer'])

    # data phases are parsed from start and shedule phases are parsed from the end
    # it allows mixtures like this: [{ep:0, bs:16, sz:128}, {ep:0, lr:1, mom:0.9}]
    dm = DaliDataManager(PHASES)  # + args.start_epoch here

    start_time=datetime.now()  # Loading start to after everything is loaded

    runner = pt.fit_wrapper.Runner(model, optimizer, criterion, verbose=IS_MASTER,
                                   metrics=[pt.metrics.Accuracy(), pt.metrics.Accuracy(5)],
                                   callbacks=[PhasesScheduler(optimizer, [copy.deepcopy(p) for p in PHASES if 'lr' in p]),
                                              Logger(OUTDIR, logger=log.logger)])
    if args.evaluate:
        dm.set_stage(0)
        return runner.evaluate(dm.val_dl)

    for idx in range(len(dm.stages)):
        dm.set_stage(idx)
        runner.fit(dm.trn_dl,
                   steps_per_epoch=(None, 100)[args.short_epoch],
                   val_loader=dm.val_dl,
                   val_steps=(None, 10)[args.short_epoch],
                   epochs=dm.stage_len + dm.stages[idx]['ep'],
                   start_epoch=dm.stages[idx]['ep'])
    return runner._val_metrics[0].avg, [m.avg for m in runner._val_metrics[1]]


def distributed_predict(input, target, model, criterion):
    # Allows distributed prediction on uneven batches. Test set isn't always large enough for every GPU to get a batch
    batch_size=input.size(0)
    output=loss=corr1=corr5=valid_batches=0

    if batch_size:
        with torch.no_grad():
            output=model(input)
            loss=criterion(output, target).data
        # measure accuracy and record loss
        valid_batches=1
        corr1, corr5=correct(output.data, target, topk=(1, 5))

    metrics=torch.tensor([batch_size, valid_batches, loss, corr1, corr5]).float().cuda()
    batch_total, valid_batches, reduced_loss, corr1, corr5=dist_utils.sum_tensor(metrics).cpu().numpy()
    reduced_loss=reduced_loss/valid_batches

    top1=corr1*(100.0/batch_total)
    top5=corr5*(100.0/batch_total)
    return top1, top5, reduced_loss, batch_total

class DaliDataManager():
    """Almost the same as DataManager but lazy and only gets dataloaders when asked"""

    def __init__(self, phases):
        self.stages=[copy.deepcopy(p) for p in phases if 'bs' in p]
        eps=[listify(p['ep']) for p in phases]
        self.tot_epochs=max([max(ep) for ep in eps])

    def set_stage(self, idx):
        stage=self.stages[idx]
        self._set_data(stage)
        if (idx+1) < len(self.stages):
            self.stage_len=self.stages[idx+1]['ep'] - stage['ep']
        else:
            self.stage_len=self.tot_epochs - stage['ep']

    def _set_data(self, phase):
        log.event('Dataset changed.\nImage size: {}\nBatch size: {}'.format(phase["sz"], phase["bs"]))
        #tb.log_size(phase['bs'], phase['sz'])
        if getattr(self, 'trn_dl', None):
            # remove if exist. prevents DALI errors
            del self.trn_dl
            del self.val_dl
            torch.cuda.empty_cache()
        self.trn_dl, self.val_dl=self._load_data(**phase)

    def _load_data(self, ep, sz, bs, **kwargs):
        if 'lr' in kwargs:
            del kwargs['lr']  # in case we mix schedule and data phases
        if 'mom' in kwargs:
            del kwargs['mom']  # in case we mix schedule and data phases
        self.rect=kwargs.get('rect_val', False)
        if self.rect:
            del kwargs['rect_val']
        if sz == 128:
            val_bs=max(bs, 512)
        elif sz == 224:
            val_bs=max(bs, 256)
        else:
            val_bs=max(bs, 128)
            
        trn_loader=get_loader(sz=sz, bs=bs, workers=args.workers, train=True, local_rank=args.local_rank, world_size=args.world_size, **kwargs)
        val_loader=get_loader(sz=sz, bs=val_bs, workers=args.workers, train=False, local_rank=args.local_rank, world_size=args.world_size, **kwargs)
        return trn_loader, val_loader

if __name__ == '__main__':
    _, res=main()
    acc1, acc5 = res[0], res[1]
    # need to calculate mean of val metrics between processes, because each validated on different images
    if args.distributed:
        print('Distributed')
        metrics = torch.tensor([acc1, acc5]).float().cuda()
        acc1, acc5 = dist_utils.sum_tensor(metrics).cpu().numpy() / args.world_size
    print("Before reduce at {}: Acc@1 {:.3f} Acc@5 {:.3f}".format(args.local_rank, res[0], res[1]))
    print("Acc@1 {:.3f} Acc@5 {:.3f}".format(acc1, acc5))
