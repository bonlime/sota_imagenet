# $1 is config file

docker run --rm -it --shm-size 8G --gpus '"device=0,1,2,3"' \
  -v $IMAGENET_DIR:/workdir/data/ \
  -v `pwd`/logs/:/workdir/logs \
  -v `pwd`/configs/:/workdir/configs \
  bonlime/imagenet:latest \
 python3 -m torch.distributed.launch --nproc_per_node=4 \
 train.py -c $@
