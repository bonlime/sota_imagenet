lr = 1.0
bs = [512, 224, 128] # largest batch size that fits in memory for each image size
bs_scale = [x/bs[0] for x in bs]
PHASES = [
  {'ep':0,  'sz':128, 'bs':bs[0], 'dali_val':True},
  {'ep':(0,7),  'lr':(lr,lr*2), 'mom':(0.9, 0.8)}, # lr warmup is better with --init-bn0
  {'ep':(7,13), 'lr':(lr*2,lr/4), 'mom':(0.8, 0.9)}, # trying one cycle
  {'ep':13, 'sz':224, 'bs':bs[1], 'dali_val':True},
  {'ep':(13,22),'lr':(lr*bs_scale[1],lr/10*bs_scale[1]), 'mom':(0.9,0.9)},
  {'ep':(22,25),'lr':(lr/10*bs_scale[1],lr/100*bs_scale[1]), 'mom':(0.9,0.9)},
  {'ep':25, 'sz':288, 'bs':bs[2], 'rect_val':True}, # 'min_scale':0.5, 
  {'ep':(25,28),'lr':(lr/100*bs_scale[2],lr/1000*bs_scale[2]), 'mom':(0.9,0.9)}
]