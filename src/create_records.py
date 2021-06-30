"""Converts ImageNet data to TFRecords file format with Example protos. Run with `python3 src/create_records.py $IMAGENET_DIR/raw-data`

The raw ImageNet data set is expected to reside in JPEG files located in the following directory structure.

  data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
  data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
  ...

where 'n01440764' is the unique synset label associated with these images.

The training data set consists of 1000 sub-directories (i.e. labels) each containing 1200 JPEG images for a total of 1.2M JPEG images.
The evaluation data set consists of 1000 sub-directories (i.e. labels) each containing 50 JPEG images for a total of 50K JPEG images.

This TensorFlow script converts the training and evaluation data into a sharded data set. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/class/label: integer specifying the index in a classification layer. The label ranges from [1, 1000] where 0 is not used.

Running this script using 16 cores should take ~25 mins
"""

import cv2
import sys
import shutil
import random
import subprocess
import tensorflow as tf
from pathlib import Path
from loguru import logger
from typing import List, Dict
from dataclasses import dataclass
from multiprocessing import Pool
from configargparse import ArgumentParser

logger.configure(handlers=[{"sink": sys.stdout, "format": "{time:[MM-DD HH:mm:ss]} - {message}"}])
random.seed(42)  # want deterministic shuffle for filenames


@dataclass
class WorkerTask:
    # all files that go into shard
    filenames: List[Path]
    # name of output tfrecord file
    out_name: Path
    # name of output index file
    out_index_name: Path
    # dict to turn folder name into int label
    synset_to_label: Dict[str, int]


def get_args():
    p = ArgumentParser()
    p.add_argument("root_data_dir", type=Path, help="folder to process")
    p.add_argument("--train_shards", type=int, default=128, help="Number of shards in training TFRecord files")
    p.add_argument("--val_shards", type=int, default=16, help="Number of shards in validation TFRecord files")
    p.add_argument("--skip_train", action="store_true", help="if given, only convert val images")

    return p.parse_args()


def _int64_feature(value: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _is_broken(filename: Path) -> bool:
    """Some images are CMYK, PNG instead of jpeg. For such images need to read and encode them again to avoid errors later
    File list from: https://github.com/cytsai/ilsvrc-cmyk-image-list"""
    # fmt: off
    blacklist = [
        'n01739381_1309', 'n02077923_14822', 'n02447366_23489', 'n02492035_15739', 'n02747177_10752', 'n03018349_4028', 
        'n03062245_4620', 'n03347037_9675',  'n03467068_12171', 'n03529860_11437', 'n03544143_17228', 'n03633091_5218',
        'n03710637_5125', 'n03961711_5286', 'n04033995_2932', 'n04258138_17003', 'n04264628_27969', 'n04336792_7448',
        'n04371774_5854', 'n04596742_4225', 'n07583066_647', 'n13037406_4650', 'n02105855_2933'
    ]
    # fmt: on
    return filename.stem in blacklist


def _single_worker_func(task: WorkerTask):
    with tf.io.TFRecordWriter(str(task.out_name)) as writer:
        for filename in task.filenames:
            if _is_broken(filename):
                img = cv2.imread(str(filename))
                # saving with 95% compression. it is good enough, but images would be slightly different from original ones
                _, img_bytes = cv2.imencode(".jpeg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                img_bytes = bytes(img_bytes)
            else:  # if images is not broken, skip decode. makes this script much faster
                img_bytes = filename.open("rb").read()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image/class/label": _int64_feature(task.synset_to_label[filename.parent.name]),
                        "image/filename": _bytes_feature(bytes(filename.name, "utf-8")),
                        "image/encoded": _bytes_feature(img_bytes),
                    }
                )
            )
            writer.write(example.SerializeToString())
    # create DALI index to be used in dataloader
    subprocess.call(["tfrecord2idx", str(task.out_name), str(task.out_index_name)])
    logger.info(f"Finished {task.out_name.stem}")


def _process_folder(data_dir: Path, n_shards: int, synset_to_label: Dict[str, int]):
    filenames = sorted(data_dir.glob("*/*.JPEG"))
    # DALI expects TFRecords to be pre-shuffled. do it once here, instead of doing it inside loaders
    random.shuffle(filenames)
    num_images = len(filenames)
    images_per_shard = num_images // n_shards
    shard_ranges = [range(i * images_per_shard, (i + 1) * images_per_shard) for i in range(n_shards)]
    # make sure last images are put into last shard
    shard_ranges[-1] = range((n_shards - 1) * images_per_shard, num_images)
    out_name = data_dir.parent / (data_dir.name + "_records")
    out_index_name = data_dir.parent / (data_dir.name + "_indexes")
    shutil.rmtree(out_name, ignore_errors=True)
    shutil.rmtree(out_index_name, ignore_errors=True)
    out_name.mkdir(exist_ok=True)
    out_index_name.mkdir(exist_ok=True)
    tasks = []
    for idx in range(n_shards):
        task = WorkerTask(
            filenames[shard_ranges[idx].start : shard_ranges[idx].stop],
            out_name=out_name / f"{data_dir.name}-{idx}-{n_shards}.tfrecord",
            out_index_name=out_index_name / f"{data_dir.name}-{idx}-{n_shards}.idx",
            synset_to_label=synset_to_label,
        )
        tasks.append(task)

    with Pool() as pool:
        pool.map(_single_worker_func, tasks)


def main():
    args = get_args()
    logger.info(args)

    try:
        subprocess.call(["tfrecord2idx"])
    except FileNotFoundError:
        raise ImportError("Install NVIDIA DALI to be able to create indexes for tfrecords")

    assert args.root_data_dir.exists(), "Root data dir doesn't exist!"
    assert (args.root_data_dir / "train").exists(), "Train data dir doesn't exist!"
    assert (args.root_data_dir / "val").exists(), "Val data dir doesn't exist!"

    sorted_synsets = sorted((args.root_data_dir / "train").iterdir())
    synset_to_label = {s.name: i for s, i in zip(sorted_synsets, range(len(sorted_synsets)))}
    val_sorted_synsets = sorted((args.root_data_dir / "val").iterdir())
    val_synset_to_label = {s.name: i for s, i in zip(sorted_synsets, range(len(val_sorted_synsets)))}
    assert val_synset_to_label == synset_to_label, "Train and val dirs should contain the same number of classes"

    _process_folder(args.root_data_dir / "val", args.val_shards, synset_to_label)
    if not args.skip_train:
        _process_folder(args.root_data_dir / "train", args.train_shards, synset_to_label)


if __name__ == "__main__":
    main()
