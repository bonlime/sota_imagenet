# same as run_train but runs only on last 2 gpus
# $1 is config file

docker run --rm -it --shm-size 8G --gpus '"device=2,3"' \
  --userns=host \
  -v $IMAGENET_DIR:/workdir/data/ \
  -v `pwd`/logs/:/workdir/logs \
  -v `pwd`/configs/:/workdir/configs \
  bonlime/imagenet:latest \
 python3 -m torch.distributed.launch --nproc_per_node=2 \
 train.py -c $@
