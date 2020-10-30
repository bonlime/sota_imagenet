import os
import sys
import yaml
import copy
import math
import json
import time
import shutil
import warnings
from pathlib import Path
from loguru import logger
from datetime import datetime

import timm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist

from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP_torch

import pytorch_tools as pt
import pytorch_tools.models as models
import pytorch_tools.fit_wrapper.callbacks as pt_clb
from pytorch_tools.fit_wrapper.callbacks import Callback as NoClbk

from pytorch_tools.utils.misc import listify
from pytorch_tools.optim import optimizer_from_name

from src.arg_parser import parse_args
from src.dali_dataloader import DaliLoader, ValRectLoader
from src.utils import HardNegativeWrapper
from src.utils import FixMatchLoss

FLAGS = parse_args()
# makes it slightly faster
cudnn.benchmark = True
if FLAGS.deterministic:
    pt.utils.misc.set_random_seed(42)

# save script and configs so we can reproduce from logs
OUTDIR = os.path.join(FLAGS.logdir, FLAGS.name)
os.makedirs(OUTDIR, exist_ok=True)
shutil.copy2(os.path.realpath(__file__), f"{OUTDIR}")

yaml.dump(vars(FLAGS), open(OUTDIR + "/config.yaml", "w"), default_flow_style=None)
# setup logger
if FLAGS.is_master:
    config = {
        "handlers": [
            {"sink": sys.stdout, "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
            {"sink": f"{OUTDIR}/logs.txt", "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
        ],
    }
    logger.configure(**config)
else:
    logger.configure(handlers=[])


def main():
    logger.info(FLAGS)
    logger.info(f"Pytorch-tools version: {pt.__version__}")
    logger.info(f"Torch version: {torch.__version__}")

    if FLAGS.distributed:
        logger.info("Distributed initializing process group")
        torch.cuda.set_device(FLAGS.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=FLAGS.world_size)

    logger.info("Loading model")
    if FLAGS.arch.startswith("timm_"):
        # allow using timms models through config
        model = timm.models.__dict__[FLAGS.arch[5:]](**FLAGS.model_params)
    elif FLAGS.arch == "GENet_normal":
        sys.path.append("/mnt/GPU-Efficient-Networks/")
        import GENet

        model = GENet.genet_normal(pretrained=False)
    else:
        model = models.__dict__[FLAGS.arch](**FLAGS.model_params)
    if FLAGS.weight_standardization:
        model = pt.modules.weight_standartization.conv_to_ws_conv(model)
    logger.info(f"Model params: {pt.utils.misc.count_parameters(model)[0]/1e6:.2f}M")
    model = model.cuda()
    optim_params = pt.utils.misc.filter_bn_from_wd(model) if FLAGS.no_bn_wd else model.parameters()

    if FLAGS.sigmoid_trick:
        if hasattr(model, "last_linear"):  # speedup convergence
            nn.init.constant_(model.last_linear.bias, -4.59)
            print("Using sigmoid trick")

    # define loss function (criterion) and optimizer
    # it's a good idea to use smooth with mixup but don't force it
    # FLAGS.smooth = FLAGS.smooth or FLAGS.mixup
    if FLAGS.criterion == "cce":
        # dirty way to support older code. TODO: rewrite
        FLAGS.criterion_params["smoothing"] = 0.1 if FLAGS.smooth else 0.0
        criterion = pt.losses.CrossEntropyLoss(**FLAGS.criterion_params).cuda()
    elif FLAGS.criterion == "sigmoid":
        criterion = torch.nn.MultiLabelSoftMarginLoss(**FLAGS.criterion_params).cuda()
    elif FLAGS.criterion == "focal":
        criterion = pt.losses.FocalLoss(**FLAGS.criterion_params)
    elif FLAGS.criterion == "kld":  # the most suitable loss for sigmoid output with cutmix
        criterion = pt.losses.BinaryKLDivLoss(**FLAGS.criterion_params).cuda()
    elif FLAGS.fixmatch:
        logger.info(f"Using special fixmatch criterion")
        criterion = FixMatchLoss(**FLAGS.criterion_params).cuda()

    if FLAGS.hard_pct > 0:  # maybe wrap with HNM
        criterion = HardNegativeWrapper(criterion, FLAGS.hard_pct)
    # start with 0 lr. Scheduler will change this later
    optimizer = optimizer_from_name(FLAGS.optim)(
        optim_params, lr=0, weight_decay=FLAGS.weight_decay, **FLAGS.optim_params
    )

    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume, map_location=lambda storage, loc: storage.cuda(FLAGS.local_rank),)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        FLAGS.start_epoch = checkpoint["epoch"]
        print("Checkpoint best epoch: ", checkpoint["epoch"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:  # may raise an error if another optimzer was used
            logger.info("Failed to load state dict into optimizer")

    if FLAGS.lookahead:
        optimizer = pt.optim.Lookahead(optimizer, la_alpha=0.5)

    # Important to create EMA Callback after cuda() and AMP but before DDP wrapper
    ema_clb = pt_clb.ModelEma(model, FLAGS.ema_decay) if FLAGS.ema_decay else NoClbk()
    if FLAGS.distributed:
        model = DDP_torch(model, device_ids=[FLAGS.local_rank])

    # data phases are parsed from start and shedule phases are parsed from the end
    # it allows mixtures like this: [{ep:0, bs:16, sz:128}, {ep:0, lr:1, mom:0.9}]
    dm = DaliDataManager(FLAGS.phases)  # + FLAGS.start_epoch here

    model_saver = pt_clb.CheckpointSaver(OUTDIR, save_name="model.chpn") if FLAGS.is_master else NoClbk()
    # common callbacks
    callbacks = [
        pt_clb.BatchMetrics([pt.metrics.Accuracy(), pt.metrics.Accuracy(5)]),
        pt_clb.PhasesScheduler([copy.deepcopy(p) for p in FLAGS.phases if "lr" in p]),
        pt_clb.FileLogger(),
        pt_clb.Mixup(FLAGS.mixup, 1000) if FLAGS.mixup else NoClbk(),
        pt_clb.Cutmix(FLAGS.cutmix, 1000) if FLAGS.cutmix else NoClbk(),
        model_saver,  # need to have CheckpointSaver before EMA so moving it here
        ema_clb,  # ModelEMA MUST go after checkpoint saver to work, otherwise it would save main model instead of EMA
    ]
    if FLAGS.is_master:  # callback for master process
        callbacks.extend(
            [pt_clb.Timer(), pt_clb.ConsoleLogger(), pt_clb.TensorBoard(OUTDIR, log_every=25),]
        )
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion,
        callbacks=callbacks,
        use_fp16=FLAGS.opt_level != "O0",
        accumulate_steps=FLAGS.accumulate_steps,
    )
    if FLAGS.evaluate:
        dm.set_stage(0)
        runner.callbacks.on_begin()
        runner.evaluate(dm.val_dl)
        return runner.state.loss_meter.avg, runner.state.metric_meters

    for idx in range(len(dm.stages)):
        dm.set_stage(idx)
        runner.fit(
            dm.trn_dl,
            steps_per_epoch=(None, 10)[FLAGS.short_epoch],
            val_loader=dm.val_dl,
            val_steps=(None, 20)[FLAGS.short_epoch],
            epochs=dm.stage_len + dm.stages[idx]["ep"],
            start_epoch=dm.stages[idx]["ep"],
        )
    # print number of params again for easier copy-paste
    logger.info(f"Model params: {pt.utils.misc.count_parameters(model)[0]/1e6:.2f}M")
    return runner.state.val_loss.avg, runner.state.val_metrics


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
        logger.info(f"Dataset changed.\nImage size: {phase['sz']}\nBatch size: {phase['bs']}")
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
            if hasattr(FLAGS, k):
                setattr(FLAGS, k, v)

        # 50.000 should be dividable by val_bs * num_gpu
        # otherwise reduced accuracy differs from acc on 1 gpu
        val_sz = kwargs.pop("val_sz", sz)  # maybe use differend size for validation
        if val_sz == 128:
            val_bs = 500
        elif val_sz == 224:
            val_bs = 250
        else:
            val_bs = 125
        trn_loader = DaliLoader(
            train=True,
            bs=bs,
            workers=FLAGS.workers,
            sz=sz,
            ctwist=FLAGS.ctwist,
            min_area=FLAGS.min_area,
            resize_method=FLAGS.resize_method,
            classes_divisor=FLAGS.classes_divisor,
            use_tfrecords=FLAGS.use_tfrecords,
            crop_method=FLAGS.crop_method,
            jitter=FLAGS.jitter,
            blur=FLAGS.blur,
            random_interpolation=FLAGS.random_interpolation,
            fixmatch=FLAGS.fixmatch,
        )
        if FLAGS.rect_validation:
            val_loader = ValRectLoader(bs=val_bs, workers=FLAGS.workers, sz=val_sz, resize_method=FLAGS.resize_method)
        else:
            val_loader = DaliLoader(
                False,
                bs=val_bs,
                workers=FLAGS.workers,
                sz=val_sz,
                resize_method=FLAGS.resize_method,
                classes_divisor=FLAGS.classes_divisor,
                use_tfrecords=FLAGS.use_tfrecords,
                crop_method=FLAGS.crop_method,
            )
        return trn_loader, val_loader


if __name__ == "__main__":
    start_time = time.time()  # Loading start to after everything is loaded
    _, metrics = main()
    # metrics here are already reduced by runner. no need to anything additionally
    if FLAGS.is_master:
        logger.info(f"Acc@1 {metrics['Acc@1'].avg:.3f} Acc@5 {metrics['Acc@5'].avg:.3f}")
        m = (time.time() - start_time) / 60
        logger.info(f"Total time: {int(m / 60)}h {m % 60:.1f}m")
