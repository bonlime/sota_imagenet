
# TODO
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

# FAILS WITH AUGMENTAION 
# PROBABLY CAN REPEAT THIS ERROR BY SWITCHING DATA LOADER
# Original phases for 1 machine
# 91.46 in 14.12h 
# The only modification was lr/2 instead of lr during the first stage. It's needed to avoid nan loss at the beginnning
# Don't know why it didn't work 
lr = 1.0
bs = [512, 224, 128] # largest batch size that fits in memory for each image size
bs_scale = [x/bs[0] for x in bs]
LOADED_PHASES = [
  {'ep':0,  'sz':128, 'bs':bs[0]},
  {'ep':(0,5),  'lr':(lr/2,lr*2), 'mom':0.9}, # lr warmup is better with --init-bn0
  {'ep':5, 'lr':lr}, # trying one cycle
  {'ep':14, 'sz':224, 'bs':bs[1], 'lr': lr*bs_scale[1]},
  {'ep':16, 'lr':lr/10*bs_scale[1]},
  {'ep':27, 'lr':lr/100*bs_scale[1]},
  {'ep':32, 'sz':288, 'bs':bs[2], 'rect_val':True, 'min_area': 0.5},
  {'ep':32, 'lr':lr/100*bs_scale[2]},
  {'ep':(33, 35), 'lr':lr/1000*bs_scale[2]},
]




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


# 91.5% in 9.62h
# droped lr too early, could continue training on 128 till convergence
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
#   {'ep':(24, 28), 'lr':(lr/10*bs_scale[2], lr/100*bs_scale[2])},
#   {'ep':28, 'sz':256, 'bs':bs[2], 'rect_val':True}, 
#   {'ep':(28, 32), 'lr':(lr/100*bs_scale[2], lr/1000*bs_scale[2])},
# ]


# bs = [512, 224, 128]
# INITIAL Phases. not enough for convergence
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