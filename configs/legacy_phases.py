"""File with different training phases for Imagenet"""
# Try no wd for bias
# label smoothing. Changes the ideal output from +oo to something finite
# Can be implemented as label smoothing + KLDivLoss. But need to check that it's the same as xent

# # MobilenetV2 phases
# LOADED_PHASES = [
#   {'ep':0,  'sz':224, 'bs':96},
#   {'ep':0,  'lr':0.045, 'mom':0.9},
# ]
# for ep in range(1,110):
#     LOADED_PHASES.append({'ep':ep, 'lr':0.045*0.99**ep})
# FAILS WITH augmentation
# PROBABLY CAN REPEAT THIS ERROR BY SWITCHING DATA LOADER
# Original phases for 1 machine
# 91.46 in 14.12h
# The only modification was lr/2 instead of lr during the first stage. It's needed to avoid nan loss at the beginning
# Don't know why it didn't work
# lr = 0.9
# bs = [512, 224, 128] # largest batch size that fits in memory for each image size
# bs_scale = [x/bs[0] for x in bs]
# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':0, 'lr': lr/10, 'mom':0.9},
#   {'ep':1,  'sz':224, 'bs':bs[2]},
# {'ep':1, 'lr': lr/10, 'mom':0.9}
# ]
# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,1), 'lr': (0, lr), 'mom':0.9},
#   {'ep':(1,5),  'lr':(lr, lr*2)}, # lr warmup is better with --init-bn0
#   {'ep':5, 'lr':lr}, # trying one cycle
#   {'ep':14, 'sz':224, 'bs':bs[1], 'lr': lr*bs_scale[1]},
#   {'ep':16, 'lr':lr/10*bs_scale[1]},
#   {'ep':27, 'lr':lr/100*bs_scale[1]},
#   {'ep':32, 'sz':288, 'bs':bs[2], 'rect_val':True, 'min_area': 0.5},
#   {'ep':32, 'lr':lr/100*bs_scale[2]},
#   {'ep':(33, 35), 'lr':lr/1000*bs_scale[2]},
# ]

# phases for ADAM
# lr = 0.7*1e-2
# bs = [512, 224, 128]
# bs_scale = [x/bs[0] for x in bs]
# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,8),  'lr':(0,lr*2), 'mode':'cos'},
#   {'ep':(8,26), 'lr':(lr*2,lr/2), 'mode':'cos'}, # trying one cycle
#   {'ep':26, 'sz':192, 'bs':bs[1]},
#   {'ep':(26, 34), 'lr':(lr*bs_scale[1], lr/5*bs_scale[1])},
#   {'ep':34, 'sz':224, 'bs':bs[2]},
#   {'ep':(34, 37), 'lr':(lr/5*bs_scale[2], lr/25*bs_scale[2])},
#   {'ep':37, 'sz':256, 'bs':bs[2], 'rect_val':True, 'min_area':0.2},
#   {'ep':(37, 40), 'lr':(lr/25*bs_scale[2], lr/125*bs_scale[2])},
# ]


# lr = 0.9
# bs = [512, 224, 128]
# bs_scale = [x/bs[0] for x in bs]
# LOADED_PHASES = [
#  {'ep':0,  'sz':128, 'bs':bs[0]},
#  {'ep':(0,8),  'lr':(lr/4,lr*2), 'mom':0.9, 'mode':'cos'},
#  {'ep':(8,26), 'lr':(lr*2,lr/2), 'mode':'cos'}, # trying one cycle
#  {'ep':26, 'sz':192, 'bs':bs[1]},
#  {'ep':(26, 34), 'lr':(lr*bs_scale[1], lr/5*bs_scale[1])},
#  {'ep':34, 'sz':224, 'bs':bs[2]},
#  {'ep':(34, 37), 'lr':(lr/5*bs_scale[2], lr/25*bs_scale[2])},
#  {'ep':37, 'sz':256, 'bs':bs[2], 'rect_val':True, 'min_area':0.2},
#  {'ep':(37, 40), 'lr':(lr/25*bs_scale[2], lr/125*bs_scale[2])},
# ]


# 91.9 in 8.53h with
# not enough training! xent is still decreasing when i drop bs and when stop training
# min_area=0.5 causes a VERY rapid decrease of xent at the end but it didn't affect val accuracy
# lr = 0.9
# bs = [512, 224, 128] # largest batch size that fits in memory for each image size
# bs_scale = [x/bs[0] for x in bs]
# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,8),  'lr':(lr/2,lr*2), 'mom':0.9, 'mode':'cos'}, # lr warmup is better with --init-bn0
#   {'ep':(8,24), 'lr':(lr*2,lr/2), 'mode':'cos'}, # trying one cycle
#   {'ep':24, 'sz':192, 'bs':bs[1]},
#   {'ep':(24, 30), 'lr':(lr*bs_scale[1], lr/5*bs_scale[1])},
#   {'ep':30, 'sz':224, 'bs':bs[2]},
#   {'ep':(30, 33), 'lr':(lr/5*bs_scale[2], lr/25*bs_scale[2])},
#   {'ep':33, 'sz':256, 'bs':bs[2], 'rect_val':True, 'min_area':0.5},
#   {'ep':(33, 34), 'lr':(lr/25*bs_scale[2], lr/125*bs_scale[2])},
# ]


# default Imagenet settings
# lr = 0.4
# LOADED_PHASES = [
#     {"ep": 0, "sz": 224, "bs": 256},
#     {"ep": [0, 5], "lr": [0, lr], "mom": 0.9},
#     {"ep": 5, "lr":  lr},
#     {"ep": 30, "lr": lr / 10},
#     {"ep": 60, "lr": lr / 100},
#     {"ep": [80, 90], "lr": [lr / 1000, lr / 1000]}
# ]

# default cosine Imagenet settings
# lr = 0.5
# LOADED_PHASES = [
#     {"ep": 0, "sz": 224, "bs": 256},
#     {"ep": [0, 5], "lr": [0, lr], "mom": 0.9},
#     {"ep": 5, "lr":  lr},
#     {"ep": 30, "lr": lr / 10},
#     {"ep": 60, "lr": lr / 100},
#     {"ep": [80, 90], "lr": [lr / 1000, lr / 1000]}
# ]


# 10.11.19 phases with reset
# lr = 0.7
# bs = [512, 224, 128] # largest batch size that fits in memory for each image size
# bs_scale = [x/bs[0] for x in bs]
# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,2), 'lr': (lr/50, lr/2), 'mom':0.9},
#   {'ep':(2,10),  'lr':(lr/2,lr*2), 'mode':'cos'},
#   {'ep':(10,30), 'lr':(lr*2,lr/2), 'mode':'cos'}, # trying one cycle
#   {'ep':30, 'sz':192, 'bs':bs[1]},
#   {'ep':(30, 38), 'lr':(lr*bs_scale[1], lr/50*bs_scale[1]), 'mode':'cos'},
#   # we got a good starting weights, can reset now
#   {'ep':38,  'sz':128, 'bs':bs[0]},
#   {'ep':(38, 50), 'lr': (lr/50, lr*2), 'mode':'cos'},
#   {'ep':(50, 76),'lr': (lr*2, lr/2), 'mode': 'cos'},
#   {'ep':76, 'sz':192, 'bs':bs[1]},
#   {'ep':(76, 106), 'lr':(lr*bs_scale[1], lr/100*bs_scale[1]), 'mode':'cos'},
#   {'ep':106, 'sz':224, 'bs':bs[2]},
#   {'ep':(106, 150), 'lr':(lr/100*bs_scale[2], lr/2000*bs_scale[2]), 'mode':'cos'},
# ]


# 1.11.19
# lr = 0.7
# bs = [512, 224, 128] # largest batch size that fits in memory for each image size
# bs_scale = [x/bs[0] for x in bs]
# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,2), 'lr': (lr/50, lr/2), 'mom':0.9},
#   {'ep':(2,10),  'lr':(lr/2,lr*2), 'mode':'cos'}, # lr warmup is better with --init-bn0
#   {'ep':(10,26), 'lr':(lr*2,lr/2), 'mode':'cos'}, # trying one cycle
#   {'ep':26, 'sz':192, 'bs':bs[1]},
#   {'ep':(26, 34), 'lr':(lr*bs_scale[1], lr/10*bs_scale[1])},
#   {'ep':38, 'sz':224, 'bs':bs[2]},
#   {'ep':(38, 40), 'lr':(lr/10*bs_scale[2], lr/200*bs_scale[2])},
# {'ep':28, 'sz':256, 'bs':bs[2]},
# {'ep':(28, 32), 'lr':(lr/100*bs_scale[2], lr/1000*bs_scale[2])},
# ]

# 91.5% in 9.62h
# dropped lr too early, could continue training on 128 till convergence
# 1.11.19: 91.39 in 2.65h on 4 V100
# lr = 0.9
# bs = [512, 224, 128] # largest batch size that fits in memory for each image size
# bs_scale = [x/bs[0] for x in bs]
# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,10),  'lr':(lr/2,lr*2), 'mom':0.9, 'mode':'cos'}, # lr warmup is better with --init-bn0
#   {'ep':(10,20), 'lr':(lr*2,lr/2), 'mode':'cos'}, # trying one cycle
#   {'ep':20, 'sz':192, 'bs':bs[1]},
#   {'ep':(20, 24), 'lr':(lr*bs_scale[1], lr/10*bs_scale[1])},
#   {'ep':24, 'sz':224, 'bs':bs[2]},
#   {'ep':(24, 32), 'lr':(lr/10*bs_scale[2], lr/100*bs_scale[2])},
# {'ep':28, 'sz':256, 'bs':bs[2]},
# {'ep':(28, 32), 'lr':(lr/100*bs_scale[2], lr/1000*bs_scale[2])},
# ]
# trying to repeat the experiment above with new code base 14.11.19
# lr = 0.5 # 0.125 * 4 gpus
# bs = [512, 224, 128] # largest batch size that fits in memory for each image size
# bs_scale = [x/bs[0] for x in bs]

# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,10),  'lr':(lr/2,lr*2), 'mom':0.9}, # lr warmup is better with --init-bn0
#   {'ep':(10,20), 'lr':(lr*2,lr/2)}, # trying one cycle
#   {'ep':20, 'sz':192, 'bs':bs[1]},
#   {'ep':(20, 24), 'lr':(lr*bs_scale[1], lr/10*bs_scale[1])},
#   {'ep':24, 'sz':224, 'bs':bs[2]},
#   {'ep':(24, 28), 'lr':(lr/10*bs_scale[2], lr/100*bs_scale[2])},
#   {'ep':28, 'sz':256, 'bs':bs[2]},
#   {'ep':(28, 32), 'lr':(lr/100*bs_scale[2], lr/1000*bs_scale[2])},
# ]


# INITIAL Phases. not enough for convergence
# lr = 0.9
# bs = [512, 224, 128]
# bs_scale = [x/bs[0] for x in bs]
# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,7),  'lr':(lr,lr*2), 'mom':(0.9, 0.8), 'mode':'cos'}, # lr warmup is better with --init-bn0
#   {'ep':(7,13), 'lr':(lr*2,lr/4), 'mom':(0.8, 0.9),'mode':'cos'}, # trying one cycle
#   {'ep':13, 'sz':224, 'bs':bs[1]},
#   {'ep':(13,22),'lr':(lr*bs_scale[1],lr/10*bs_scale[1]), 'mom':(0.9,0.9),'mode':'cos'},
#   {'ep':(22,25),'lr':(lr/10*bs_scale[1],lr/100*bs_scale[1]), 'mom':(0.9,0.9),'mode':'cos'},
#   {'ep':25, 'sz':288, 'bs':bs[2], 'rect_val':True}, # 'min_scale':0.5,
#   {'ep':(25,28),'lr':(lr/100*bs_scale[2],lr/1000*bs_scale[2]), 'mom':(0.9,0.9), 'mode':'cos'}
# ]


# NEW PHASES Starting from 17.11.19 #######
# cite from of the last Google papers
# For full ImageNet training, we use RMSProp optimizer
# with decay 0.9 and momentum 0.9. Batch norm is added
# after every convolution layer with momentum 0.99, and
# weight decay is 1e-5. Dropout rate 0.2 is applied to the last
# layer. Following [7], learning rate is increased from 0 to
# 0.256 in the first 5 epochs, and then decayed by 0.97 every
# 2.4 epochs. We use batch size 4K and Inception preprocessing with image size 224×224. For COCO training, we plug
# our learned model into SSD detector [22] and use the same
# settings as [29], including input size 320 × 320

# Acc@1 72.794 Acc@5 91.067
# very bad convergence in general
# lr = 0.4 # 0.1 * 4 gpus
# bs = [768, 384, 256] # largest batch size that fits in memory for each image size
# bs_scale = [x/bs[0] for x in bs]

# LOADED_PHASES = [
#   {'ep':0,  'sz':128, 'bs':bs[0]},
#   {'ep':(0,10),  'lr':(0,lr*2), 'mom':0.9},
#   {'ep':(10,40), 'lr':(lr*2,lr/1e3), 'mode': 'cos'}, # trying one cycle
#   {'ep':40, 'sz':192, 'bs':bs[1]},
#   {'ep':(40, 70), 'lr':(lr/10*bs_scale[1], lr/1e3*bs_scale[1]), 'mode':'cos'},
#   {'ep':70, 'sz':224, 'bs':bs[2]},
#   {'ep':(70, 100), 'lr':(lr/100*bs_scale[2], lr/1e3*bs_scale[2]), 'mode':'cos'},
# ]


# slightly faster, slightly worse Acc@1 73.464 Acc@5 91.425 Total time: 4h 51.4m
# lr = 0.4 * 2 # 2 comes from bs being 2 times larger
# bs = [512, 256] # largest batch size that fits in memory for each image size

# LOADED_PHASES = [
#     {"ep": 0, "sz": 128, "bs": bs[0]},
#     {"ep": [0, 5], "lr": [0, lr], "mom": 0.9},
#     {"ep": 5, "lr":  lr},
#     {"ep": 30, "sz": 192, "bs": bs[1]},
#     {"ep": 30, "lr": lr / 20},
#     {"ep": 60, "sz": 224, "bs": bs[1]},
#     {"ep": 60, "lr": lr / 200},
#     {"ep": [80, 100], "lr": [lr / 2000, lr / 2000]}
# ]

# LOADED_PHASES = [{"ep": 0, "sz": 224, "bs": 256}, {"ep": [0, 5], "lr": [0, 0.007]}]

# Need to test
# lr = 0.1
# bs = 256
# LOADED_PHASES = [
#     {"ep": 0, "sz": 224, "bs": bs},
#     {"ep": [0, 5], "lr": [0, lr], "mom": 0.9},
#     {"ep": [5, 60], "lr":  [lr, lr/100], 'mode':'cos'},
#     {"ep": 30, "sz": 192, "bs": bs},
#     {"ep": 60, "sz": 224, "bs": bs},
#     {"ep": [60, 100], "lr":  [lr/100, 0], 'mode':'cos'},
# ]

#  python3 -m torch.distributed.launch --nproc_per_node=4 train.py -a resnet34 --opt_level O1 --load-phases --no-bn-wd -n resnet34_1phase --optim fused_sgd --wd 1e-5 --lookahead --model-params "{'deep_stem':True, 'antialias':True}"
# ~ 75.19
# lr = 0.5
# bs = 256
# LOADED_PHASES = [
#     {"ep": 0, "sz": 128, "bs": bs, "ctwist": True, "mixup": 0.2},
#     {"ep": [0, 50], "lr": [lr, 0], "mom": 0.9, "mode": "cos"},
#     {"ep": 50, "sz": 192, "bs": bs},
#     {"ep": [50, 100], "lr": [lr, 0], "mode": "cos"},
#     {"ep": 100, "sz": 224, "bs": bs},
#     {"ep": [100, 150], "lr": [lr, 0], "mode": "cos"},
#     {"ep": 150, "sz": 224, "bs": bs, "min_area": 0.4, "ctwist": False, "mixup": 0.0},
#     {"ep": [150, 170], "lr": [1e-3, 1e-3]},
# ]

# lr = 0.5
# bs = 256
# LOADED_PHASES = [
#     {"ep": 0, "sz": 128, "bs": bs},
#     {"ep": 2, "sz": 224, "bs": bs},
#     {"ep": [0, 5], "lr": [0, lr], "mom": 0.9},
#     {"ep": 4, "sz": 224, "bs": bs, 'min_area': 0.2, 'cutmix': 0., 'ctwist': False},
# ]


#  python3 -m torch.distributed.launch --nproc_per_node=4 train_new.py -a resnet34 --opt_level O1 --load-phases --no-bn-wd -n resnet34_1phase --optim fused_sgd --wd 1e-5 --lookahead --smooth --mixup 0.2 --ctwist --model-params "{'deep_stem':True, 'antialias':True}"
# 
lr = 0.5
bs = 256
LOADED_PHASES = [
    {"ep": 0, "sz": 128, "bs": bs},
    {"ep": [0, 5], "lr": [0, lr], "mom": 0.9},
    {"ep": [5, 200], "lr": [lr, 0], "mode": "cos"},
    {"ep": 60, "sz": 192, "bs": bs},
    {"ep": 120, "sz": 224, "bs": bs},
    {"ep": 180, "sz": 224, "bs": bs, 'min_area': 0.2, 'cutmix': 0., 'ctwist': False},
]


# testing Novograd
# lr = 0.4
# LOADED_PHASES = [
#     {"ep": 0, "sz": 224, "bs": 256},
#     {"ep": [0, 5], "lr": [0, lr], "mom": 0.9},
#     {"ep": [5, 150], "lr":  [lr, 0], 'mode':'cos'},
# ]
