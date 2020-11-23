import argparse
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Dummy model inference script.")

    parser.add_argument("--half", action="store_true", help="Use half precision.")
    parser.add_argument("--fuse", action="store_true", help="Fuse model before inference.")
    parser.add_argument("--jit", action="store_true", help="Jit script model")

    parser.add_argument("--benchmark", action="store_true", help="torch.backends.cudnn.benchmark = True")

    return parser.parse_args()


class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def load_models(args):
    model = (
        nn.Sequential(SimpleBlock(3, 32), SimpleBlock(32, 64), *[SimpleBlock(64, 64) for _ in range(64)])
        .requires_grad_(False)
        .eval()
    )
    if args.fuse:
        model = fuse_model(model)

    if args.half:
        model = model.half()

    return model


def process_cube(cube, model):
    return model(cube)


def check_model_speed(args, model):
    input_tensor = torch.randn((16, 3, 256, 256), dtype=args.dtype, device=args.device,)

    process_cube(input_tensor, model)
    num_itrs = 20

    start = time.perf_counter()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    with torch.no_grad(), torch.cuda.amp.autocast(args.half):
        for _ in tqdm(range(num_itrs)):
            process_cube(input_tensor, model)
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    elapsed_time = time.perf_counter() - start
    logger.info(
        f"Results for simple model: \n\tElapsed time: {elapsed_time:.3f}. \n\t"
        f"Time for one iteration: {elapsed_time / num_itrs:.3f} sec.\n\t"
        f"FPS: {input_tensor.size(0) * num_itrs / elapsed_time:.3f}."
    )


def main():
    args = get_args()

    if args.benchmark:
        torch.backends.cudnn.benchmark = True

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.half = args.half if torch.cuda.is_available() else False
    args.dtype = torch.float16 if args.half else torch.float32

    print(args)

    model = load_models(args).to(args.device)
    if args.jit:
        model = torch.jit.script(model)
    check_model_speed(args, model)


if __name__ == "__main__":
    main()
