import configargparse as argparse

import pytorch_tools as pt
import pytorch_tools.models as models


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
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    add_arg("--model_params", type=eval, default={}, help="Additional model params as kwargs")
    add_arg(
        "--weight_standardization", action="store_true", help="Change convs to WS Convs. See paper for details",
    )
    add_arg("--ema_decay", type=float, default=0, help="If not zero, enables EMA decay for model weights")
    add_arg("--sigmoid_trick", action="store_true", help="If True sets last bias to speedup convergence")

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
    add_arg("--resize_method", type=str, default="linear", choices=["linear", "cubic"], help="Interpolation type")
    add_arg("--classes_divisor", type=int, default=1, help="Used for reduction of number of classes")
    add_arg("--use_tfrecords", action="store_true", help="Flag to read data from tfrecords instead of files")
    add_arg("--rect_validation", action="store_true", help="Flag to perform validation on rectangles")

    ## CRITERION
    add_arg("--smooth", action="store_true", help="Use label smoothing")
    add_arg("--criterion", type=str, default="cce", help="Criterion to use")
    add_arg("--criterion_params", type=eval, default={}, help="Additional loss params as kwargs")
    add_arg("--hard_pct", type=float, default=0.0, help="If large that zero, wraps loss in HardNegativeMiner")

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
    add_arg("--accumulate_steps", type=int, default=1, help="Num steps for gradient accumulation")

    ## LOGGING
    add_arg("--logdir", default="logs", type=str, help="where logs go")
    add_arg(
        "-n", "--name", type=str, default="", dest="name", help="Name of this run. If empty it would be a timestamp",
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
