
from zzc_dataset import get_seq_random_walk_random_global_jumps
import numpy as np
import random


meta = np.load('matlab/data/test.npz', encoding='latin1', allow_pickle=True)
f0 = random.randint(0, 10000)  # 随机选择初始点f0
seq, jump = get_seq_random_walk_random_global_jumps(meta, f0, 300)


# 验证一下游走出来的序列是什么样的
val_label = np.zeros(meta['dual_vertex'].shape[0])
val_label[seq] = 1
np.savetxt('matlab/val_20.seg', val_label, fmt='%d', delimiter='\t')