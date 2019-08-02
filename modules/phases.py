lr = 1.0
bs = [512, 224, 128] # largest batch size that fits in memory for each image size
bs_scale = [x/bs[0] for x in bs]
LOADED_PHASES = [
  {'ep':0,  'sz':128, 'bs':bs[0]},
  {'ep':(0,10),  'lr':(lr/2,lr*2), 'mom':0.9, 'mode':'cos'}, # lr warmup is better with --init-bn0
  {'ep':(10,16), 'lr':(lr*2,lr/4), 'mode':'cos'}, # trying one cycle
  {'ep':16, 'sz':192, 'bs':bs[1]},
  {'ep':(16, 24), 'lr':(lr/4, lr*bs_scale[1]), 'mode':'cos'},
  {'ep':(24, 32), 'lr':(lr*bs_scale[1], lr/10*bs_scale[1]), 'mode':'cos'},
  {'ep':32, 'sz':224, 'bs':bs[2]},
  {'ep':(32, 36), 'lr':(lr/10*bs_scale[1], lr/100*bs_scale[2]), 'mode':'cos'},
  {'ep':36, 'sz':256, 'bs':bs[2], 'rect_val':True}, 
  {'ep':(36, 40), 'lr':(lr/100*bs_scale[2], lr/1000*bs_scale[2]), 'mode':'cos'},
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