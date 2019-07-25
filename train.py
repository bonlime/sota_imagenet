import argparse, os, shutil, time, warnings, math
from datetime import datetime
from pathlib import Path
import sys, os
import math
import collections
import gc
import json

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import torchvision.models as models
# for fp16
from apex import amp
import copy

from modules import dali_dataloader
from modules.dataloader import fast_collate, create_dataset, BatchTransformDataLoader
from modules import experimental_utils
from modules  import dist_utils
from modules.logger import TensorboardLogger, FileLogger
from modules.meter import AverageMeter, TimeMeter   

def get_parser():
    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    parser.add_argument('--gpu', type=int, default='0', 
                        help='GPU to use')
    parser.add_argument('--phases', type=str,
                    help='Specify epoch order of data resize and learning rate schedule: [{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]')
    parser.add_argument('--load-phases', action='store_true', 
                        help='Flag to load phases.json config'),
    # parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # TODO actually add this feature
    # parser.add_argument('--init-bn0', action='store_true', help='Intialize running batch norm mean to 0')
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='log/print every this many steps (default: 5)')
    parser.add_argument('--no-bn-wd', action='store_true', help='Remove batch norm from weight decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--opt_level', default='O0', type=str, choices=['O0','O1','O2','O3'], 
                        help='optimizatin level for apex. (default: "00")')
    parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
    # parser.add_argument('--dist-url', default='env://', type=str,
    #                    help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    # TODO write logs into separete folders
    parser.add_argument('--logdir', default='logs', type=str,
                        help='where logs go')
    # parser.add_argument('--skip-auto-shutdown', action='store_true',
    #                     help='Shutdown instance at the end of training or failure')
    # parser.add_argument('--auto-shutdown-success-delay-mins', default=10, type=int,
    #                     help='how long to wait until shutting down on success')
    # parser.add_argument('--auto-shutdown-failure-delay-mins', default=60, type=int,
    #                     help='how long to wait before shutting down on error')
    parser.add_argument('--short-epoch', action='store_true',
                        help='make epochs short (for debugging)')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()

# set gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)


# Only want master rank logging to tensorboard
is_master = (not args.distributed) or (dist_utils.env_rank()==0)
is_rank0 = args.local_rank == 0
tb = TensorboardLogger(args.logdir, is_master=is_master)
log = FileLogger(args.logdir, is_master=is_master, is_rank0=is_rank0)

lr = 1.0
bs = [512, 224, 128] # largest batch size that fits in memory for each image size
bs_scale = [x/bs[0] for x in bs]
one_machine = [
  {'ep':0,  'sz':128, 'bs':bs[0], 'rect_train': False},
  {'ep':(0,7),  'lr':(lr,lr*2), 'mom':(0.9, 0.8)}, # lr warmup is better with --init-bn0
  {'ep':(7,13), 'lr':(lr*2,lr/4), 'mom':(0.8, 0.9)}, # trying one cycle
  {'ep':13, 'sz':224, 'bs':bs[1], 'rect_train':False},
  {'ep':(13,22),'lr':(lr*bs_scale[1],lr/10*bs_scale[1]), 'mom':(0.9,0.9)},
  {'ep':(22,25),'lr':(lr/10*bs_scale[1],lr/100*bs_scale[1]), 'mom':(0.9,0.9)},
  {'ep':25, 'sz':288, 'bs':bs[2], 'min_scale':0.5, 'rect_val':True, 'rect_train':False},
  {'ep':(25,28),'lr':(lr/100*bs_scale[2],lr/1000*bs_scale[2]), 'mom':(0.9,0.9)}
]

def main():
    os.system('shutdown -c')  # cancel previous shutdown command
    log.console(args)
    #tb.log('sizes/world', dist_utils.env_world_size())
    
    if args.distributed:
        log.console('Distributed initializing process group')
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=dist_utils.env_world_size())
        assert(dist_utils.env_world_size() == dist.get_world_size())
        log.console("Distributed: success (%d/%d)"%(args.local_rank, dist.get_world_size()))


    log.console("Loading model")
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    model = model.cuda()

    if args.distributed: 
        raise NotImplementedError
        model = dist_utils.DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    best_top5 = 93 # only save models over 93%. Otherwise it stops to save every time
    optim_params = experimental_utils.bnwd_optim_params(model) if args.no_bn_wd else model.parameters()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(optim_params, 0, weight_decay=args.weight_decay) # start with 0 lr. Scheduler will change this later
    
    
    model, optimizer = amp.initialize(model, optimizer, 
                                      opt_level=args.opt_level)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.local_rank))
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_top5 = checkpoint['best_top5']
        optimizer.load_state_dict(checkpoint['optimizer'])
            
    # save script so we can reproduce from logs
    shutil.copy2(os.path.realpath(__file__), '{}'.format(args.logdir))

    log.console("Creating data loaders (this could take up to 10 minutes if volume needs to be warmed up)")
    # data phases are parsed from start and shedule phases are parsed from the end
    # it allows mixtures like this: [{ep:0, bs:16, sz:128}, {ep:0, lr:1, mom:0.9}]
    
    if args.load_phases:
        phases = one_machine
        #with open('phases.json') as fp:
        #    phases = json.load(fp)
    else: phases = eval(args.phases)
    dm = DaliDataManager([copy.deepcopy(p) for p in phases if 'bs' in p])
    scheduler = Scheduler(optimizer, [copy.deepcopy(p) for p in phases if 'lr' in p])

    start_time = datetime.now() # Loading start to after everything is loaded
    if args.evaluate:
        dm.set_epoch(0) 
        return validate(dm.val_dl, model, criterion, 0, start_time)

    if args.distributed:
        log.console('Syncing machines before training')
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

    log.event("~~epoch\thours\ttop1\ttop5\n")
    for epoch in range(args.start_epoch, scheduler.tot_epochs):
        dm.set_epoch(epoch)

        train(dm.trn_dl, model, criterion, optimizer, scheduler, epoch)
        top1, top5 = validate(dm.val_dl, model, criterion, epoch, start_time)

        time_diff = (datetime.now()-start_time).total_seconds()/3600.0
        log.event('~~{}\t{:.5f}\t\t{:.3f}\t\t{:.3f}\n'.format(epoch, time_diff, top1, top5))

        is_best = top5 > best_top5
        best_top5 = max(top5, best_top5)
        if args.local_rank == 0:
            if is_best: save_checkpoint(epoch, model, best_top5, optimizer, is_best=True, filename='model_best.pth.tar')
            phase = dm.get_phase(epoch)
            if phase: save_checkpoint(epoch, model, best_top5, optimizer, filename='sz{}_checkpoint.path.tar'.format(phase["bs"]))


def train(trn_loader, model, criterion, optimizer, scheduler, epoch):
    timer = TimeMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(trn_loader):
        if args.short_epoch and (i > 10): break
        batch_num = i+1
        timer.batch_start()
        scheduler.update_lr_mom(epoch, i+1, len(trn_loader))

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute grads
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # Train batch done. Logging results
        timer.batch_end()
        corr1, corr5 = correct(output.data, target, topk=(1, 5))
        reduced_loss, batch_total = to_python_float(loss.data), to_python_float(input.size(0))
        if args.distributed: # Must keep track of global batch size, since not all machines are guaranteed equal batches at the end of an epoch
            metrics = torch.tensor([batch_total, reduced_loss, corr1, corr5]).float().cuda()
            batch_total, reduced_loss, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()
            reduced_loss = reduced_loss/dist_utils.env_world_size()
        top1acc = to_python_float(corr1)*(100.0/batch_total)
        top5acc = to_python_float(corr5)*(100.0/batch_total)

        losses.update(reduced_loss, batch_total)
        top1.update(top1acc, batch_total)
        top5.update(top5acc, batch_total)

        should_print = (batch_num%args.print_freq == 0) or (batch_num==len(trn_loader))
        if args.local_rank == 0 and should_print:
            tb.log_memory()
            tb.log_trn_times(timer.batch_time.val, timer.data_time.val, input.size(0))
            tb.log_trn_loss(losses.val, top1.val, top5.val)

            tb.log("sizes/batch_total", batch_total)

            output = ('Epoch: [{}][{}/{}]\t'.format(epoch, batch_num, len(trn_loader)) + 
                      'Time {:.3f} ({:.3f})\t'.format(timer.batch_time.val, timer.batch_time.avg) + 
                      'Loss {:.4f} ({:.4f})\t'.format(losses.val, losses.avg) + 
                      'Acc@1 {:.3f} ({:.3f})\t'.format(top1.val, top1.avg) + 
                      'Acc@5 {:.3f} ({:.3f})\t'.format(top5.val, top5.avg) + 
                      'Data {:.3f} ({:.3f})\t'.format(timer.data_time.val, timer.data_time.avg))
            log.verbose(output)

        tb.update_step_count(batch_total)


def validate(val_loader, model, criterion, epoch, start_time):
    timer = TimeMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    eval_start_time = time.time()

    for i, (input, target) in enumerate(val_loader):
        
        if args.short_epoch and (i > 10): break
        batch_num = i+1
        timer.batch_start()
        if args.distributed:
            top1acc, top5acc, loss, batch_total = distributed_predict(input, target, model, criterion)
        else:
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target).data
            batch_total = input.size(0)
            top1acc, top5acc = accuracy(output.data, target, topk=(1,5))
        # Eval batch done. Logging results
        timer.batch_end()
        losses.update(to_python_float(loss), to_python_float(batch_total))
        top1.update(to_python_float(top1acc), to_python_float(batch_total))
        top5.update(to_python_float(top5acc), to_python_float(batch_total))
        should_print = (batch_num%args.print_freq == 0) or (batch_num==len(val_loader))
        if args.local_rank == 0 and should_print:
            output = ('Test:  [{}][{}/{}]\t'.format(epoch, batch_num, len(val_loader)) + 
                      'Time {:.3f} ({:.3f})\t'.format(timer.batch_time.val, timer.batch_time.avg) + 
                      'Loss {:.4f} ({:.4f})\t'.format(losses.val, losses.avg) + 
                      'Acc@1 {:.3f} ({:.3f})\t'.format(top1.val, top1.avg) + 
                      'Acc@5 {:.3f} ({:.3f})'.format(top5.val, top5.avg))
            log.verbose(output)

    tb.log_eval(top1.avg, top5.avg, time.time()-eval_start_time)
    tb.log('epoch', epoch)

    return top1.avg, top5.avg

def distributed_predict(input, target, model, criterion):
    # Allows distributed prediction on uneven batches. Test set isn't always large enough for every GPU to get a batch
    batch_size = input.size(0)
    output = loss = corr1 = corr5 = valid_batches = 0

    if batch_size:
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target).data
        # measure accuracy and record loss
        valid_batches = 1
        corr1, corr5 = correct(output.data, target, topk=(1, 5))

    metrics = torch.tensor([batch_size, valid_batches, loss, corr1, corr5]).float().cuda()
    batch_total, valid_batches, reduced_loss, corr1, corr5 = dist_utils.sum_tensor(metrics).cpu().numpy()
    reduced_loss = reduced_loss/valid_batches

    top1 = corr1*(100.0/batch_total)
    top5 = corr5*(100.0/batch_total)
    return top1, top5, reduced_loss, batch_total

VAL_DIR = '/home/zakirov/datasets/imagenet_2012/raw_data/validation'
class DaliDataManager():
    """Almost the same as DataManager but lazy and only gets dataloaders when asked"""
    def __init__(self, phases):
        self._phases = phases

    def set_epoch(self, epoch):
        cur_phase = next((p for p in self._phases if p['ep'] == epoch), None)
        if cur_phase: self._set_data(cur_phase)
    
    def _set_data(self, phase):
        log.event('Dataset changed.\nImage size: {}\nBatch size: {}'.format(phase["sz"], phase["bs"]))
        tb.log_size(phase['bs'], phase['sz'])
        self.trn_dl, self.val_dl = self._load_data(**phase)

    def _load_data(self, ep, sz, bs, **kwargs):
        if 'lr' in kwargs: del kwargs['lr']  # in case we mix schedule and data phases
        if 'mom' in kwargs: del kwargs['mom'] # in case we mix schedule and data phases
        dali_val = False
        if 'dali_val' in kwargs:
            del kwargs['dali_val']
            dali_val = True
        rect = False
        if 'rect_val' in kwargs: 
            del kwargs['rect_val']
            rect = True
        if sz == 128: val_bs = max(bs, 512)
        elif sz == 224: val_bs = max(bs, 256)
        else: val_bs = max(bs, 128)
        trn_loader =  dali_dataloader.get_loader(sz=sz, bs=bs, workers=args.workers, 
                                                  device_id=0, train=True, **kwargs)
        if dali_val:
            val_loader =  dali_dataloader.get_loader(sz=sz, bs=val_bs, workers=args.workers, 
                                                  device_id=0, train=False, **kwargs)
        else:
            val_dtst, val_sampler = create_dataset(VAL_DIR, val_bs, sz, rect, args.distributed, False)
            val_loader = torch.utils.data.DataLoader(
                val_dtst,
                num_workers=args.workers, pin_memory=True, collate_fn=fast_collate,
                batch_sampler=val_sampler)
            val_loader = BatchTransformDataLoader(val_loader)
        return trn_loader, val_loader

# ### Learning rate scheduler
class Scheduler():
    def __init__(self, optimizer, phases):
        self.optimizer = optimizer
        self.current_lr = None
        self.current_mom = None
        self.phases = [self.format_phase(p) for p in phases]
        self.tot_epochs = max([max(p['ep']) for p in self.phases])

    def format_phase(self, phase):
        phase['ep'] = listify(phase['ep'])
        phase['lr'] = listify(phase['lr'])
        phase['mom'] = listify(phase['mom'])
        if len(phase['lr']) == 2 or len(phase['mom']) == 2:
            phase['mode'] = phase.get('mode', 'linear')
            assert (len(phase['ep']) == 2), 'Linear learning rates must contain end epoch'
        return phase
    
    def get_current_phase(self, epoch):
        for phase in reversed(self.phases): 
            if (epoch >= phase['ep'][0]): return phase
        raise Exception('Epoch out of range')

    @staticmethod
    def _schedule(start, end, pct, mode):
        """anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        if mode == 'linear':
            return start + (end - start) * pct
        elif mode == 'cos':
            return end + (start - end)/2 * (math.cos(math.pi * pct) + 1)

    def get_lr_mom(self, epoch, batch_curr, batch_tot):
        phase = self.get_current_phase(epoch)
        # TODO allow mom changing separately
        if len(phase['lr']) == 1: 
            new_lr, new_mom = phase['lr'][0], phase['mom'][0] # constant learning rate
        else:
            mom_start, mom_end = phase['mom']
            lr_start, lr_end = phase['lr']
            ep_start, ep_end = phase['ep']
            ep_curr, ep_tot = epoch - ep_start, ep_end - ep_start
            perc = (ep_curr * batch_tot + batch_curr) / (ep_tot * batch_tot)
            #print("Perc: {:.2f}. LR: {:4f} - {:4f}. Mom: {:2f} - {:2f}".format(perc*100, lr_start, lr_end, mom_start, mom_end))
            new_lr = self._schedule(lr_start, lr_end, perc, phase['mode'])
            new_mom = self._schedule(mom_start, mom_end, perc, phase['mode'])
        return new_lr, new_mom

    def update_lr_mom(self, epoch, batch_num, batch_tot):
        lr, mom = self.get_lr_mom(epoch, batch_num, batch_tot)
        if self.current_lr == lr and self.current_mom == mom: return

        if ((batch_num == 1) or (batch_num == batch_tot)): 
            log.event('Changing LR from {} to {}'.format(self.current_lr, lr))
            log.event('Changing Momentum from {} to {}'.format(self.current_mom, mom))

        self.current_lr = lr
        self.current_mom = mom
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = mom
            
        tb.log("sizes/lr", lr)
        tb.log("sizes/momentum", mom)

def listify(p=None, q=None):
    if p is None: p=[]
    elif not isinstance(p, collections.Iterable): p=[p]
    n = q if type(q)==int else 1 if q is None else len(q)
    if len(p)==1: p = p * n
    return p

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if isinstance(t, (float, int)): return t
    if hasattr(t, 'item'): return t.item()
    else: return t[0]

def save_checkpoint(epoch, model, best_top5, optimizer, is_best=False, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch+1, 'state_dict': model.state_dict(),
        'best_top5': best_top5, 'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, filename)
    if is_best: shutil.copyfile(filename, '{}/{}'.format(args.logdir, filename))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    corrrect_ks = correct(output, target, topk)
    batch_size = target.size(0)
    return [correct_k.float().mul_(100.0 / batch_size) for correct_k in corrrect_ks]

def correct(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0, keepdim=True)
        res.append(correct_k)
    return res



if __name__ == '__main__':
    main()
    # try:
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore", category=UserWarning)
    #         main()
    #     #if not args.skip_auto_shutdown: os.system(f'sudo shutdown -h -P +{args.auto_shutdown_success_delay_mins}')
    # except Exception as e:
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     import traceback
    #     traceback.print_tb(exc_traceback, file=sys.stdout)
    #     log.event(e)
    #     # in case of exception, wait 2 hours before shutting down
    #     #if not args.skip_auto_shutdown: os.system(f'sudo shutdown -h -P +{args.auto_shutdown_failure_delay_mins}')
    # tb.close()


