import os
import sys
import time
import subprocess
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import pytorch_tools as pt
import pytorch_tools.fit_wrapper.callbacks as pt_clb
from pytorch_tools.fit_wrapper.callbacks import Callback as NoClbk

import hydra
from omegaconf import OmegaConf

from sota_imagenet.arg_parser import StrictConfig, DataStage
from sota_imagenet.dali_dataloader import DaliDataManager
from sota_imagenet.callbacks import WeightDistributionTB


@hydra.main(config_path="./configs", config_name="base")
def main(cfg: StrictConfig):

    start_time = time.time()
    # setup distributed args
    cfg.distributed = cfg.world_size > 1
    # Only want master rank logging to tensorboard
    cfg.is_master = cfg.local_rank == 0

    # save hashid and git diff for reproduceability. current dir is already in logs because of Hydra
    kwargs = {"universal_newlines": True, "stdout": subprocess.PIPE}
    with open("commit_hash.txt", "w") as f:
        f.write(subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs).stdout)
    with open("diff.txt", "w") as f:
        f.write(subprocess.run(["git", "diff"], **kwargs).stdout)

    # setup loguru logger
    if cfg.is_master:
        config = {
            "handlers": [
                {"sink": sys.stdout, "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
                {"sink": f"logs.txt", "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
            ],
        }
        logger.configure(**config)
    else:
        logger.configure(handlers=[])

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Pytorch-tools version: {pt.__version__}")
    logger.info(f"Torch version: {torch.__version__}")

    torch.backends.cudnn.benchmark = True
    if cfg.random_seed is not None:
        pt.utils.misc.set_random_seed(cfg.random_seed)

    if cfg.distributed:
        logger.info("Distributed initializing process group")
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=cfg.world_size)

    logger.info("Loading model")
    model = hydra.utils.call(cfg.model)

    if cfg.weight_standardization:
        model = pt.modules.weight_standartization.conv_to_ws_conv(model)

    # correctly initialize weights
    if cfg.init_gamma is not None:
        pt.utils.misc.initialize(model, cfg.init_gamma)

    model = model.cuda()

    # default mom in PyTorch causes underperformance
    pt.utils.misc.patch_bn_mom(model, cfg.bn_momentum)

    if cfg.log.print_model:
        logger.info(model)

    criterion = hydra.utils.call(cfg.criterion).cuda()
    # maybe filter bn | bias | something else from weight decay
    if cfg.filter_from_wd is not None:
        opt_params = pt.utils.misc.filter_from_weight_decay(model, skip_list=cfg.filter_from_wd)
    else:
        opt_params = [{"params": list(model.parameters())}]

    # if criterion has it's own params, also optimize them
    opt_params[0]["params"].extend(list(criterion.parameters()))

    # start with 0 lr. Scheduler will change this later
    optimizer = hydra.utils.call(cfg.optim, opt_params)

    # need to log number of parameters after creating criterion because it may change in the process
    # for example because of MLP layer
    logger.info(f"Model params: {pt.utils.misc.count_parameters(model)[0]/1e6:.2f}M")

    if cfg.run.resume:
        resume_path = hydra.utils.to_absolute_path(cfg.run.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage.cuda(cfg.local_rank))
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"Loader model checkpoint from {resume_path}")
        if cfg.run.load_start_epoch:
            cfg.run.start_epoch = checkpoint["epoch"]
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info("Loader optimizer state")
        except:  # may raise an error if another optimzer was used
            logger.info("Failed to load state dict into optimizer. It wasn't saved or optimizer has changed")

    # Important to create EMA Callback after cuda() and AMP but before DDP wrapper
    ema_clb = pt_clb.ModelEma(model, cfg.run.ema_decay) if cfg.run.ema_decay else NoClbk()
    if cfg.distributed:
        model = DDP(model, device_ids=[cfg.local_rank])

    # nesting dataclasses in List is not currently supported. so do it manually
    cfg.run.stages = [DataStage(**stg) for stg in cfg.run.stages]
    logger.info(cfg.run.stages)

    # need to convert stages to dict, so that PhasesScheduler can consume it
    lr_stages = []
    for stage in cfg.run.stages:
        if stage.lr is None:
            continue
        lr_stages.append(dict(ep=(stage.start, stage.end), lr=stage.lr, mode=stage.lr_mode))
    logger.info(f"Learning rate stages: {lr_stages}")

    # common callbacks
    callbacks = [
        pt_clb.BatchMetrics([pt.metrics.Accuracy(), pt.metrics.Accuracy(5)]),
        pt_clb.PhasesScheduler(lr_stages),
        pt_clb.FileLogger(),
        # need to have CheckpointSaver before EMA so moving it here. current dir is already inside logs because of hydra
        pt_clb.CheckpointSaver(os.getcwd(), save_name="model.chpn", include_optimizer=cfg.log.save_optim),
        ema_clb,  # ModelEMA MUST go after checkpoint saver to work, otherwise it would save main model instead of EMA
        # callbacks below are only for master process. this is handled by `rank_zero_only` decorator
        pt_clb.Timer(),
        pt_clb.ConsoleLogger(),
        pt_clb.TensorBoard(os.getcwd(), log_every=50),
        WeightDistributionTB() if cfg.log.histogram else NoClbk(),
    ]
    # here we can add any custom callback. MixUp / CutMix is also defined here
    callbacks += [hydra.utils.call(clb_cfg) for clb_cfg in cfg.run.extra_callbacks]

    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion,
        callbacks=callbacks,
        use_fp16=cfg.run.fp16,
        accumulate_steps=cfg.run.accumulate_steps,
    )

    # data phases are parsed from start and schedule phases are parsed from the end
    # it allows mixtures like this: [{ep:0, bs:16, sz:128}, {ep:0, lr:1, mom:0.9}]
    data_manager = DaliDataManager(cfg)

    if cfg.run.evaluate:
        data_manager.set_stage(0)
        runner.callbacks.on_begin()
        runner.evaluate(data_manager.val_loader)
        return runner.state.loss_meter.avg, runner.state.metric_meters

    for idx in range(len(data_manager)):
        data_manager.set_stage(idx)
        runner.fit(
            data_manager.loader,
            steps_per_epoch=(None, 10)[cfg.debug],
            val_loader=data_manager.val_loader,
            val_steps=(None, 20)[cfg.debug],
            epochs=data_manager.end_epoch,
            start_epoch=data_manager.start_epoch,
        )
    # print number of params again for easier copy-paste
    logger.info(f"Model params: {pt.utils.misc.count_parameters(model)[0]/1e6:.2f}M")

    # metrics here are already reduced by runner. no need to anything additionally
    metrics = runner.state.val_metrics
    logger.info(f"Acc@1 {metrics['Acc@1'].avg:.3f} Acc@5 {metrics['Acc@5'].avg:.3f}")
    m = (time.time() - start_time) / 60
    logger.info(f"Total time: {int(m / 60)}h {m % 60:.1f}m")

    if cfg.is_master:  # additionally save the final model
        torch.save(model.state_dict(), "model_last.chpn")


if __name__ == "__main__":
    # TODO: find out how to return anything from hydra app. right now it always returns None
    main()
