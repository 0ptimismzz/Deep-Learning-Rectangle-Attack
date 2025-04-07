import deep_net_sp_mrcp as tn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tn.train_speck_distinguisher(50, num_rounds=7, diff=(0x40, 0), group_size=2, depth=10)




