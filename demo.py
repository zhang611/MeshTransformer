
from my_dataset import get_seq_random_walk_random_global_jumps
import numpy as np
import random


meta = np.load('matlab/data/test.npz', encoding='latin1', allow_pickle=True)
f0 = random.randint(0, 10000)  # 随机选择初始点f0
seq, jump = get_seq_random_walk_random_global_jumps(meta, f0, 300)


# 验证一下游走出来的序列是什么样的
val_label = np.zeros(meta['dual_vertex'].shape[0])
val_label[seq] = 1
np.savetxt('matlab/val_20.seg', val_label, fmt='%d', delimiter='\t')


# 加载npz查看
import numpy as np
path = r'datasets_processed/psb/psb_teddy/1_not_changed_.npz'
mesh_data = np.load(path, encoding='latin1', allow_pickle=True)


def getPositionEncoding(seq_len, dim, n=10000):
    PE = np.zeros(shape=(seq_len, dim))
    for pos in range(seq_len):
        for i in range(int(dim / 2)):
            denominator = np.power(n, 2 * i / dim)
            PE[pos, 2 * i] = np.sin(pos / denominator)
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)

    return PE

# PE = getPositionEncoding(seq_len=300, dim=256, n=10000)
# print(PE.shape)
# print(PE)