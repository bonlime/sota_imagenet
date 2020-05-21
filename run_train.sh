docker run --rm -it --shm-size 8G --gpus '"device=1,2,3"' \
  -v $IMAGENET_DIR:/workdir/data/ \
  -v `pwd`/logs/:/workdir/logs \
  -v `pwd`/configs/:/workdir/configs \
  bonlime/imagenet:latest \
  python3 -m torch.distributed.launch --nproc_per_node=3 \
  train.py -c configs/BResNet50_encoder.yaml \
  --resume logs/bresnet50_encoder_20200518_120748/model.chpn
