from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing

cpus = multiprocessing.cpu_count()
cpus = min(48, cpus)


PATH = Path("/mnt/storage/datasets/ImageNet/raw-data")  # .home()/'data/imagenet'
# DEST = Path('/mnt/ram')
DEST = Path("/mnt/storage/datasets/ImageNet/")  # .home()/'data/imagenet-sz'
# szs = (int(128*1.25), int(256*1.25))
szs = (int(256 * 1.25),)


def resize_img(p, im, fn, sz):
    w, h = im.size
    ratio = min(h / sz, w / sz)
    im = im.resize((int(w / ratio), int(h / ratio)), resample=Image.LINEAR)  #  Image.BICUBIC)
    # import pdb; pdb.set_trace()
    new_fn = DEST / str(sz) / fn.relative_to(PATH)
    new_fn.parent.mkdir(exist_ok=True)
    im.save(new_fn)


def resizes(p, fn):
    im = Image.open(fn)
    for sz in szs:
        resize_img(p, im, fn, sz)


def resize_imgs(p):
    files = p.glob("*/*.JPEG")
    # list(map(partial(resizes, p), files))
    with ProcessPoolExecutor(cpus) as e:
        e.map(partial(resizes, p), files)


for sz in szs:
    ssz = str(sz)
    (DEST / ssz).mkdir(exist_ok=True)
    for ds in ("val", "train"):
        (DEST / ssz / ds).mkdir(exist_ok=True)

for ds in ("val", "train"):
    resize_imgs(PATH / ds)

