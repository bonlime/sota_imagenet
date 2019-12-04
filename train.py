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
from pytorch_tools.fit_wrapper.callbacks import Logger
from pytorch_tools.fit_wrapper.callbacks import TensorBoard
from pytorch_tools.fit_wrapper.callbacks import CheckpointSaver
from pytorch_tools.fit_wrapper.callbacks import PhasesScheduler
from pytorch_tools.fit_wrapper.callbacks import Callback as NoClbk

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
from modules.mixup import MixUpWrapper
from modules.cutmix import CutMixWrapper

# from absl import flags
# from absl.flags import FLAGS
import modules.config as cfg
from modules.config import FLAGS

# FLAGS = cfg.FLAGS


def parse_args():

    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    add_arg = parser.add_argument
    add_arg(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    add_arg("--model-params", type=str, default="{}", help="Additional model params as kwargs")
    add_arg(
        "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
    )
    add_arg(
        "--phases",
        type=str,
        help="Specify epoch order of data resize and learning rate schedule:"
        '[{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]',
    )
    add_arg(
        "--load-phases",
        action="store_true",
        help="Flag to load phases from modules.phases config",
    )
    add_arg(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    add_arg(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    add_arg(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    add_arg(
        "--print-freq",
        "-p",
        default=5,
        type=int,
        metavar="N",
        help="log/print every this many steps (default: 5)",
    )
    add_arg("--no-bn-wd", action="store_true", help="Remove batch norm from weight decay")
    add_arg(
        "--mixup",
        type=float,
        default=0,
        help="Alpha for mixup augmentation. If 0 then mixup is diabled",
    )
    add_arg(
        "--cutmix",
        type=float,
        default=0,
        help="Alpha for cutmix augmentation. If 0 then cutmix is diabled")
    add_arg("--smooth", action="store_true", help="Use label smoothing")
    add_arg("--ctwist", action="store_true", help="Turns on color twist augmentation")
    add_arg(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    add_arg(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    add_arg(
        "--opt_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2", "O3"],
        help='optimizatin level for apex. (default: "00")',
    )
    add_arg(
        "--local_rank",
        "--gpu",
        default=0,
        type=int,
        help="Used for multi-process training. Can either be manually set "
        + "or automatically set by using 'python -m multiproc'.",
    )
    add_arg("--logdir", default="logs", type=str, help="where logs go")
    add_arg(
        "-n",
        "--name",
        type=str,
        default="",
        dest="name",
        help="Name of this run. If empty it would be a timestamp",
    )
    add_arg("--short-epoch", action="store_true", help="make epochs short (for debugging)")
    add_arg(
        "--optim",
        type=str,
        default="SGD",  # choices=['sgd', 'sgdw', 'adam', 'adamw', 'rmsprop', 'radam'],
        help="Optimizer to use (default: sgd)",
    )
    add_arg(
        "--optim-params", type=str, default="{}", help="Additional optimizer params as kwargs"
    )
    add_arg("--deterministic", action="store_true")
    add_arg(
        "--lookahead",
        action="store_true",
        help="Flag to wrap optimizer with Lookahead wrapper",
    )

    args = parser.parse_args()
    # set some defaults
    args.sz = 224
    args.bs = 256
    args.min_area = 0.08
    # detect distributed
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1

    # Only want master rank logging to tensorboard
    args.is_master = not args.distributed or args.local_rank == 0
    timestamp = str(datetime.now()).split(".")[0].replace(" ", "_")
    args.name = args.name + "_" + timestamp if args.name else timestamp
    cfg.FLAGS = args
    return None


parse_args()
# makes it slightly faster
cudnn.benchmark = True
if cfg.FLAGS.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(cfg.FLAGS.local_rank)


# save script and runing comand so we can reproduce from logs
OUTDIR = os.path.join(cfg.FLAGS.logdir, cfg.FLAGS.name)
os.makedirs(OUTDIR, exist_ok=True)
shutil.copy2(os.path.realpath(__file__), "{}".format(OUTDIR))
with open(OUTDIR + "/run.cmd", "w") as fp:
    fp.write(" ".join(sys.argv[1:]) + "\n")
PHASES = LOADED_PHASES if cfg.FLAGS.load_phases else eval(cfg.FLAGS.phases)
with open(OUTDIR + "/phases.json", "w") as fp:
    json.dump(PHASES, fp)
log = FileLogger(OUTDIR, is_master=cfg.FLAGS.is_master)


def main():
    log.console(cfg.FLAGS)

    if cfg.FLAGS.distributed:
        log.console("Distributed initializing process group")
        torch.cuda.set_device(cfg.FLAGS.local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=cfg.FLAGS.world_size
        )

    log.console("Loading model")
    kwargs = eval(cfg.FLAGS.model_params)
    if cfg.FLAGS.pretrained:
        print("=> using pre-trained model '{}'".format(cfg.FLAGS.arch))
        model = models.__dict__[cfg.FLAGS.arch](pretrained="imagenet", **kwargs)
    else:
        print("=> creating model '{}'".format(cfg.FLAGS.arch))
        model = models.__dict__[cfg.FLAGS.arch](**kwargs)
    model = model.cuda()

    optim_params = bnwd_optim_params(model) if cfg.FLAGS.no_bn_wd else model.parameters()

    # define loss function (criterion) and optimizer
    # it's a good idea to use smooth with mixup but don't force it
    # cfg.FLAGS.smooth = cfg.FLAGS.smooth or cfg.FLAGS.mixup
    criterion = pt.losses.CrossEntropyLoss(smoothing=0.1 if cfg.FLAGS.smooth else 0.0).cuda()
    # start with 0 lr. Scheduler will change this later
    kwargs = eval(cfg.FLAGS.optim_params)
    optimizer = optimizer_from_name(cfg.FLAGS.optim)(
        optim_params, lr=0, weight_decay=cfg.FLAGS.weight_decay, **kwargs
    )

    if cfg.FLAGS.resume:
        checkpoint = torch.load(
            cfg.FLAGS.resume,
            map_location=lambda storage, loc: storage.cuda(cfg.FLAGS.local_rank),
        )
        has_module_in_sd = list(checkpoint["state_dict"].keys())[0].split(".")[0] == "module"
        if has_module_in_sd and not cfg.FLAGS.distributed:
            # remove `modules` from names
            new_sd = {}
            for k, v in checkpoint["state_dict"].items():
                new_key = ".".join(k.split(".")[1:])
                new_sd[new_key] = v
            checkpoint["state_dict"] = new_sd
        model.load_state_dict(checkpoint["state_dict"])
        cfg.FLAGS.start_epoch = checkpoint["epoch"]
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:  # may raise an error if another optimzer was used
            print("Failed to load state dict into optimizer")

    if cfg.FLAGS.lookahead:
        optimizer = pt.optim.Lookahead(optimizer)

    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level=cfg.FLAGS.opt_level,
        loss_scale=1 if cfg.FLAGS.opt_level == "O0" else 128.0,  # 2048,
        max_loss_scale=2.0 ** 13,
        min_loss_scale=1.0,
        verbosity=0,
    )
    logger_clb = Logger(OUTDIR, logger=log.logger)
    if cfg.FLAGS.distributed:
        model = DDP(model, delay_allreduce=True)
        logger_clb = DistributedLogger(OUTDIR, logger=log.logger)

    # data phases are parsed from start and shedule phases are parsed from the end
    # it allows mixtures like this: [{ep:0, bs:16, sz:128}, {ep:0, lr:1, mom:0.9}]
    dm = DaliDataManager(PHASES)  # + cfg.FLAGS.start_epoch here

    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion,
        verbose=cfg.FLAGS.is_master,
        metrics=[pt.metrics.Accuracy(), pt.metrics.Accuracy(5)],
        callbacks=[
            PhasesScheduler(optimizer, [copy.deepcopy(p) for p in PHASES if "lr" in p]),
            logger_clb,
            TensorBoard(OUTDIR, log_every=25) if cfg.FLAGS.is_master else NoClbk(),
            CheckpointSaver(OUTDIR, save_name="model.chpn")
            if cfg.FLAGS.is_master
            else NoClbk(),
        ],
    )
    if cfg.FLAGS.evaluate:
        dm.set_stage(0)
        return runner.evaluate(dm.val_dl)

    for idx in range(len(dm.stages)):
        dm.set_stage(idx)
        runner.fit(
            dm.trn_dl,
            steps_per_epoch=(None, 10)[cfg.FLAGS.short_epoch],
            val_loader=dm.val_dl,
            val_steps=(None, 10)[cfg.FLAGS.short_epoch],
            epochs=dm.stage_len + dm.stages[idx]["ep"],
            start_epoch=dm.stages[idx]["ep"],
        )
    return runner._val_metrics[0].avg, [m.avg for m in runner._val_metrics[1]]


class DaliDataManager:
    """Almost the same as DataManager but lazy and only gets dataloaders when asked"""

    def __init__(self, phases):
        self.stages = [copy.deepcopy(p) for p in phases if "bs" in p]
        eps = [listify(p["ep"]) for p in phases]
        self.tot_epochs = max([max(ep) for ep in eps])

    def set_stage(self, idx):
        stage = self.stages[idx]
        self._set_data(stage)
        if (idx + 1) < len(self.stages):
            self.stage_len = self.stages[idx + 1]["ep"] - stage["ep"]
        else:
            self.stage_len = self.tot_epochs - stage["ep"]

    def _set_data(self, phase):
        log.event(
            "Dataset changed.\nImage size: {}\nBatch size: {}".format(phase["sz"], phase["bs"])
        )
        # tb.log_size(phase['bs'], phase['sz'])
        if getattr(self, "trn_dl", None):
            # remove if exist. prevents DALI errors
            del self.trn_dl
            del self.val_dl
            torch.cuda.empty_cache()
        self.trn_dl, self.val_dl = self._load_data(**phase)

    def _load_data(self, ep, sz, bs, **kwargs):

        if "lr" in kwargs:
            del kwargs["lr"]  # in case we mix schedule and data phases
        if "mom" in kwargs:
            del kwargs["mom"]  # in case we mix schedule and data phases
        # change global parameters from phases
        for k, v in kwargs.items():
            if hasattr(cfg.FLAGS, k):
                setattr(cfg.FLAGS, k, v)

        # 50.000 should be dividable by val_bs * num_gpu
        # otherwise reduced accuracy differs from acc on 1 gpu
        if sz == 128:
            val_bs = 500
        elif sz == 224:
            val_bs = 250
        else:
            val_bs = 125
        cfg.FLAGS.sz = sz
        cfg.FLAGS.bs = bs
        trn_loader = get_loader(True)
        cfg.FLAGS.bs = val_bs
        val_loader = get_loader(False)

        if cfg.FLAGS.cutmix != 0:
            trn_loader = CutMixWrapper(cfg.FLAGS.cutmix, 1000, trn_loader)
        if cfg.FLAGS.mixup != 0:
            trn_loader = MixUpWrapper(cfg.FLAGS.mixup, 1000, trn_loader)
        return trn_loader, val_loader


class DistributedLogger(Logger):
    """Reduces metrics before printing"""

    def on_epoch_end(self):
        trn_l = self.runner._train_metrics[0].avg
        trn_acc1, trn_acc5 = (m.avg for m in self.runner._train_metrics[1])

        val_l = self.runner._val_metrics[0].avg
        val_acc1, val_acc5 = (m.avg for m in self.runner._val_metrics[1])

        tensor = (
            torch.tensor([trn_l, trn_acc1, trn_acc5, val_l, val_acc1, val_acc5]).float().cuda()
        )
        trn_l, trn_acc1, trn_acc5, val_l, val_acc1, val_acc5 = (
            dist_utils.reduce_tensor(tensor).cpu().numpy()
        )

        # replace with reduced metrics. it's dirty but works
        self.runner._train_metrics[0].avg = trn_l
        self.runner._train_metrics[1][0].avg = trn_acc1
        self.runner._train_metrics[1][1].avg = trn_acc5
        self.runner._val_metrics[0].avg = val_l
        self.runner._val_metrics[1][0].avg = val_acc1
        self.runner._val_metrics[1][1].avg = val_acc5

        trn_str = "Train       loss: {:.4f} | Acc@1 {:.4f} | Acc@5 {:.4f}".format(
            trn_l, trn_acc1, trn_acc5
        )
        self.logger.info(trn_str)
        self.logger.info(
            "Val reduced loss: {:.4f} | Acc@1 {:.4f} | Acc@5 {:.4f}".format(
                val_l, val_acc1, val_acc5
            )
        )


if __name__ == "__main__":
    start_time = time.time()  # Loading start to after everything is loaded
    _, res = main()
    acc1, acc5 = res[0], res[1]
    # need to calculate mean of val metrics between processes, because each validated on different images
    if cfg.FLAGS.distributed:
        # print('Distributed')
        metrics = torch.tensor([acc1, acc5]).float().cuda()
        acc1, acc5 = dist_utils.reduce_tensor(metrics).cpu().numpy()
    # print("Before reduce at {}: Acc@1 {:.3f} Acc@5 {:.3f}".format(cfg.FLAGS.local_rank, res[0], res[1]))
    if cfg.FLAGS.is_master:
        log.console("Acc@1 {:.3f} Acc@5 {:.3f}".format(acc1, acc5))
        m = (time.time() - start_time) / 60
        log.console("Total time: {}h {:.1f}m".format(int(m / 60), m % 60))
