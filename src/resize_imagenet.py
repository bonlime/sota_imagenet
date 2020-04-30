from PIL import Image
from pathlib import Path
import argparse
from multiprocessing.pool import Pool
from functools import partial
import tqdm


def resize_img(fn, path=None, dest=None, sz=None):
    im = Image.open(fn)
    w, h = im.size
    ratio = max(h / sz, w / sz)
    new_w, new_h = int(w / ratio), int(h / ratio)
    new_fn = dest / fn.relative_to(path)
    new_fn.parent.parent.mkdir(exist_ok=True)
    new_fn.parent.mkdir(exist_ok=True)
    if new_fn.exists() and Image.open(new_fn).size == (new_w, new_h):
        return  # don't overwrite if exists
    try:
        im = im.resize((new_w, new_h), resample=Image.LANCZOS)
    except OSError:
        print("Corrupted: ", fn)  # some files are corrupted
    im.save(new_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to folder with data")
    parser.add_argument("--sz", type=int, default=512, help="Longest size")
    args = parser.parse_args()
    PATH = Path(args.data_dir)
    DEST = PATH.with_name(PATH.name + "_" + str(args.sz))
    DEST.mkdir(exist_ok=True)
    files = PATH.glob("*/*/*.JPG")
    files = list(files)
    # list(map(partial(resize_img, path=PATH, dest=DEST, sz=args.sz), files))
    with Pool() as p:
        with tqdm.tqdm(total=len(files)) as pbar:
            for _ in p.imap_unordered(partial(resize_img, path=PATH, dest=DEST, sz=args.sz), files):
                pbar.update()


if __name__ == "__main__":
    main()
