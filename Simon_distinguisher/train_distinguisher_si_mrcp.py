import deep_net_si_mrcp as tn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tn.train_simon_distinguisher(50, num_rounds=9, diff=(0x0000, 0x0040), group_size=8, depth=10)