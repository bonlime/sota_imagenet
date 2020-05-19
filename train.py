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
import configargparse as argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist

# for fp16
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

import pytorch_tools as pt
import pytorch_tools.models as models
import pytorch_tools.fit_wrapper.callbacks as pt_clb
from pytorch_tools.fit_wrapper.callbacks import Callback as NoClbk

from pytorch_tools.utils.misc import listify
from pytorch_tools.optim import optimizer_from_name

from src.dali_dataloader import DaliLoader


def parse_args():

    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(
        description="PyTorch ImageNet Training",
        default_config_files=["configs/base.yaml"],
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )

    ## MODEL
    add_arg = parser.add_argument
    add_arg(
        "--arch",
        "-a",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    add_arg("--model_params", type=eval, default={}, help="Additional model params as kwargs")
    add_arg("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
    add_arg(
        "--weight_standardization", action="store_true", help="Change convs to WS Convs. See paper for details"
    )
    add_arg("--ema_decay", type=float, default=0, help="If not zero, enables EMA decay for model weights")

    
    ## OPTIMIZER
    add_arg("--optim", type=str, default="SGD", help="Optimizer to use (default: sgd)")
    add_arg("--optim_params", type=eval, default={}, help="Additional optimizer params as kwargs")
    add_arg("--lookahead", action="store_true", help="Flag to wrap optimizer with Lookahead wrapper")
    add_arg(
        "--weight_decay", "--wd", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)",
    )
    add_arg("--no_bn_wd", action="store_true", help="Remove batch norm from weight decay")


    ## DATALOADER
    add_arg("--sz", type=int, default=224)
    add_arg("--bs", type=int, default=256)
    add_arg("--min_area", type=float, default=0.08)
    add_arg("--workers", "-j", default=4, type=int, help="number of data loading workers (default: 4)")
    add_arg(
        "--mixup", type=float, default=0, help="Alpha for mixup augmentation. If 0 then mixup is diabled",
    )
    add_arg(
        "--cutmix", type=float, default=0, help="Alpha for cutmix augmentation. If 0 then cutmix is diabled",
    )
    add_arg("--cutmix_prob", type=float, default=0.5)
    add_arg("--ctwist", action="store_true", help="Turns on color twist augmentation")


    ## CRITERION
    add_arg("--smooth", action="store_true", help="Use label smoothing")
    add_arg("--sigmoid", action='store_true', help='Use sigmoid instead of softmax')


    ## TRAINING
    add_arg(
        "--phases",
        type=eval,
        action="append",
        help="Specify epoch order of data resize and learning rate schedule:"
        '[{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]',
    )
    add_arg(
        "--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)",
    )
    add_arg("--resume", default="", type=str, help="path to latest checkpoint (default: none)")
    add_arg("--evaluate", "-e", action="store_true", help="evaluate model on validation set")
    add_arg(
        "--opt_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2", "O3"],
        help='optimizatin level for apex. (default: "00")',
    )
    add_arg("--short_epoch", action="store_true", help="make epochs short (for debugging)")


    ## OTHER 
    add_arg(
        "--local_rank",
        "--gpu",
        default=0,
        type=int,
        help="Used for multi-process training. Can either be manually set "
        + "or automatically set by using 'python -m multiproc'.",
    )
    add_arg("--deterministic", action="store_true")


    ## LOGGING
    add_arg("--logdir", default="logs", type=str, help="where logs go")
    add_arg(
        "-n",
        "--name",
        type=str,
        default="",
        dest="name",
        help="Name of this run. If empty it would be a timestamp",
    )
    add_arg("--no_timestamp", action="store_true", help="Disables adding timestamp to run name")
    

    args, not_parsed = parser.parse_known_args()
    print(f"Not parsed args: {not_parsed}")
    
    # detect distributed
    args.world_size = pt.utils.misc.env_world_size()
    args.distributed = args.world_size > 1

    # Only want master rank logging to tensorboard
    args.is_master = not args.distributed or args.local_rank == 0
    timestamp = pt.utils.misc.get_timestamp()
    if args.name:
        args.name = args.name if args.no_timestamp else args.name + "_" + timestamp
    else:
        args.name = timestamp
    return args


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

    if FLAGS.distributed:
        logger.info("Distributed initializing process group")
        torch.cuda.set_device(FLAGS.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=FLAGS.world_size)

    logger.info("Loading model")
    if FLAGS.pretrained:
        logger.info(f"=> using pre-trained model '{FLAGS.arch}'")
        model = models.__dict__[FLAGS.arch](pretrained="imagenet", **FLAGS.model_params)
    else:
        logger.info(f"=> creating model '{FLAGS.arch}'")
        model = models.__dict__[FLAGS.arch](**FLAGS.model_params)
    if FLAGS.weight_standardization:
        model = pt.modules.weight_standartization.conv_to_ws_conv(model)
    model = model.cuda()
    optim_params = pt.utils.misc.filter_bn_from_wd(model) if FLAGS.no_bn_wd else model.parameters()

    # define loss function (criterion) and optimizer
    # it's a good idea to use smooth with mixup but don't force it
    # FLAGS.smooth = FLAGS.smooth or FLAGS.mixup
    if FLAGS.sigmoid:
        # use reduction sum just to have bigger numbers in logs
        criterion = torch.nn.MultiLabelSoftMarginLoss(reduction="sum").cuda()
    else:
        criterion = pt.losses.CrossEntropyLoss(smoothing=0.1 if FLAGS.smooth else 0.0).cuda()
    # start with 0 lr. Scheduler will change this later
    optimizer = optimizer_from_name(FLAGS.optim)(
        optim_params, lr=0, weight_decay=FLAGS.weight_decay, **FLAGS.optim_params
    )

    if FLAGS.resume:
        checkpoint = torch.load(
            FLAGS.resume, map_location=lambda storage, loc: storage.cuda(FLAGS.local_rank),
        )
        has_module_in_sd = list(checkpoint["state_dict"].keys())[0].split(".")[0] == "module"
        if has_module_in_sd and not FLAGS.distributed:
            # remove `modules` from names
            new_sd = {}
            for k, v in checkpoint["state_dict"].items():
                new_key = ".".join(k.split(".")[1:])
                new_sd[new_key] = v
            checkpoint["state_dict"] = new_sd
        model.load_state_dict(checkpoint["state_dict"])
        FLAGS.start_epoch = checkpoint["epoch"]
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:  # may raise an error if another optimzer was used
            logger.info("Failed to load state dict into optimizer")

    if FLAGS.lookahead:
        optimizer = pt.optim.Lookahead(optimizer, la_alpha=0.5)

    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level=FLAGS.opt_level,
        loss_scale=1 if FLAGS.opt_level == "O0" else 128.0,  # 2048,
        max_loss_scale=2.0 ** 13,
        min_loss_scale=1.0,
        verbosity=0,
    )
    # Important to create EMA Callback after cuda() and AMP but before DDP wrapper
    ema_clb = pt_clb.ModelEma(model, FLAGS.ema_decay) if FLAGS.ema_decay else NoClbk()
    if FLAGS.distributed:
        model = DDP(model, delay_allreduce=True)

    # data phases are parsed from start and shedule phases are parsed from the end
    # it allows mixtures like this: [{ep:0, bs:16, sz:128}, {ep:0, lr:1, mom:0.9}]
    dm = DaliDataManager(FLAGS.phases)  # + FLAGS.start_epoch here

    model_saver = pt_clb.CheckpointSaver(OUTDIR, save_name="model.chpn") if FLAGS.is_master else NoClbk()
    # common callbacks
    callbacks = [
        pt_clb.PhasesScheduler([copy.deepcopy(p) for p in FLAGS.phases if "lr" in p]),
        pt_clb.FileLogger(OUTDIR, logger=logger),
        pt_clb.Mixup(FLAGS.mixup, 1000) if FLAGS.mixup else NoClbk(),
        pt_clb.Cutmix(FLAGS.cutmix, 1000) if FLAGS.cutmix else NoClbk(),
        model_saver, # need to have CheckpointSaver before EMA so moving it here
        ema_clb, # ModelEMA MUST go after checkpoint saver to work, otherwise it would save main model instead of EMA
    ]
    if FLAGS.is_master:  # callback for master process
        callbacks.extend(
            [
                pt_clb.Timer(),
                pt_clb.ConsoleLogger(),
                pt_clb.TensorBoard(OUTDIR, log_every=25),
            ]
        )
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion,
        metrics=[pt.metrics.Accuracy(), pt.metrics.Accuracy(5)],
        callbacks=callbacks,
    )
    if FLAGS.evaluate:
        dm.set_stage(0)
        return runner.evaluate(dm.val_dl)

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
    return runner.state.val_loss.avg, [m.avg for m in runner.state.val_metrics]


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
        if sz == 128:
            val_bs = 500
        elif sz == 224:
            val_bs = 250
        else:
            val_bs = 125
        FLAGS.sz = sz
        FLAGS.bs = bs
        trn_loader = DaliLoader(True, FLAGS.bs, FLAGS.workers, FLAGS.sz, FLAGS.ctwist, FLAGS.min_area)
        FLAGS.bs = val_bs
        val_loader = DaliLoader(False, FLAGS.bs, FLAGS.workers, FLAGS.sz, FLAGS.ctwist, FLAGS.min_area)
        return trn_loader, val_loader



if __name__ == "__main__":
    start_time = time.time()  # Loading start to after everything is loaded
    _, res = main()
    acc1, acc5 = res[0], res[1]
    # need to calculate mean of val metrics between processes, because each validated on different images
    if FLAGS.distributed:
        metrics = torch.tensor([acc1, acc5]).float().cuda()
        acc1, acc5 = pt.utils.misc.reduce_tensor(metrics).cpu().numpy()
    if FLAGS.is_master:
        logger.info(f"Acc@1 {acc1:.3f} Acc@5 {acc5:.3f}")
        m = (time.time() - start_time) / 60
        logger.info(f"Total time: {int(m / 60)}h {m % 60:.1f}m")
