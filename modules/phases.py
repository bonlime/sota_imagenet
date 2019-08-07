

# # MobilenetV2 phases
# LOADED_PHASES = [
#   {'ep':0,  'sz':224, 'bs':96},
#   {'ep':0,  'lr':0.045, 'mom':0.9},
# ]
# for ep in range(1,110):
#     LOADED_PHASES.append({'ep':ep, 'lr':0.045*0.99**ep})



lr = 0.9
bs = [512, 224, 128] # largest batch size that fits in memory for each image size
bs_scale = [x/bs[0] for x in bs]
LOADED_PHASES = [
  {'ep':0,  'sz':128, 'bs':bs[0]},
  {'ep':(0,10),  'lr':(lr/2,lr*2), 'mom':0.9, 'mode':'cos'}, # lr warmup is better with --init-bn0
  {'ep':(10,20), 'lr':(lr*2,lr/2), 'mode':'cos'}, # trying one cycle
  {'ep':20, 'sz':192, 'bs':bs[1]},
  {'ep':(20, 24), 'lr':(lr*bs_scale[1], lr/10*bs_scale[1])},
  {'ep':24, 'sz':224, 'bs':bs[2]},
  {'ep':(24, 28), 'lr':(lr/10*bs_scale[2], lr/100*bs_scale[2])},
  {'ep':28, 'sz':256, 'bs':bs[2], 'rect_val':True}, 
  {'ep':(28, 32), 'lr':(lr/100*bs_scale[2], lr/1000*bs_scale[2])},
]


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