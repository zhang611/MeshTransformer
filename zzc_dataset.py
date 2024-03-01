from torch.utils.data import Dataset
import torch
import numpy as np
import os
import glob
import sys
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

'''
训练需要的
一次吐出一条序列的特征，输入神经网络
还需要对应的标签
20个模型，16个训练集，4个测试集
首先在16个模型里面随机选一个模型，然后随机选一个初始点，游走产生一条300个面的序列索引
用索引得到特征和标签

'''


def get_seq_random_walk_random_global_jumps(mesh_extra, f0, seq_len):
    """mesh_extra是保存结果的，f0是随机的初始点，seq_len是300序列长度"""
    nbrs = mesh_extra['ring']  # 用顶点索引定义的边(27648, 3)  备选点就是这三个相邻的面
    nbrs = np.int64(nbrs)

    # for i in range(mesh_extra['center_edges'].shape[0]):
    #     mesh_extra['center_edges'][i] = list(mesh_extra['center_edges'][i])  # 集合转列表,TODO 放到数据处理阶段去，太慢了

    for i in range(nbrs.shape[0]):
        nbrs[i] = list(nbrs[i])

    n_vertices = mesh_extra['center'].shape[0]  # 重心的数量 27648
    seq = np.zeros((seq_len + 1,), dtype=np.int32)  # 初始化序列(301,)
    jumps = np.zeros((seq_len + 1,), dtype=np.bool)  # (301,)全false
    visited = np.zeros((n_vertices + 1,), dtype=np.bool)  # (27648,)全false,用过的点是true
    visited[-1] = True  # 用过的点变成true
    visited[f0] = True
    seq[0] = f0
    jumps[0] = [True]  # 跳了就是jump，第一个是随机选的所以就是jump
    backward_steps = 1  # 后退步数为1
    jump_prob = 1 / 100  # jump概率0.01
    for i in range(1, seq_len + 1):
        this_nbrs = nbrs[seq[i - 1]]  # (15,)  前一个顶点所联通的其他顶点的索引，也就是这一步随机游走的选择,[1,5,9]
        nodes_to_consider = [n for n in this_nbrs if
                             not visited[n]]  # 备选顶点 [5, 6, 11, 14, 17] 这句代码，首先-1会被去掉因为visited[-1] = True，其次用过的点也是true
        jump_now = np.random.binomial(1, jump_prob)  # 从二项分布中抽取随机样本，抛硬币，抛一次，1的概率0.01，绝大数情况都不跳
        if len(nodes_to_consider) and not jump_now:  # 正常情况都走这，没有可走的点了，并且选择跳才不走这
            to_add = np.random.choice(nodes_to_consider)  # 从被选点里面随机选一个
            jump = False
            backward_steps = 1
        else:  # 没有被选的点了，并且jump_now为1，选择跳
            if i > backward_steps and not jump_now:  # 可以回退并且不跳，你这都跳来跳去了，这流行不重要啊
                to_add = seq[i - backward_steps - 1]  # 折回来，序列里有重复的面片
                backward_steps += 2
            else:  # 没有被选点了，且无法回退，且选择跳跃
                backward_steps = 1
                to_add = np.random.randint(n_vertices)  # 随机选一个点跳过去，不管重复与否
                jump = True
                visited[...] = 0  # 既然都跳跃了，那就刷新一下，重来，否则可能连续跳，三个点表示所有维度
                visited[-1] = True
        visited[to_add] = 1  # 标记使用过了
        seq[i] = to_add  # 顶点的索引加入序列
        jumps[i] = jump  # 标记在哪里跳了，代码里面用不上，就是看看跳的多不多，这里就跳了一次

    return seq, jumps


class PSBDataset(Dataset):
    def __init__(self, npz_dir):
        self.pathname_expansion = npz_dir + '/*.npz'
        self.filenames = glob.glob(self.pathname_expansion)

    def __len__(self):
        return len(self.filenames)  # 模型的数量

    def __getitem__(self, indices):
        filename = self.filenames[indices - 1]  # 1到16,每次随机选一个模型
        meta = np.load(filename, encoding='latin1', allow_pickle=True)  # 加载处理好的模型字典

        f0 = random.randint(0, meta["faces"].shape[0])  # 随机选择初始点f0
        seq, jump = get_seq_random_walk_random_global_jumps(meta, f0, 300)

        # 测地距离和二面角的特征
        ring = meta['ring']
        ring = ring.astype(np.int32)
        ring_geo = meta['geodesic']
        ring_dih = meta['dihedral']
        feature_geo = np.zeros(seq.shape, dtype=float)
        feature_dih = np.zeros(seq.shape, dtype=float)

        for i, num in enumerate(seq):  # [7648, 1794, 4065, 5140 ...]
            if i == (len(seq)-1):
                break
            if jump[i + 1]:      # True 就是代表调过了
                feature_geo[i] = -1
                feature_dih[i] = -1
            else:
                index = [j for j, x in enumerate(ring[num]) if x == seq[i + 1]][0]
                feature_geo[i] = ring_geo[num][index]
                feature_dih[i] = ring_dih[num][index]


        center = meta['center']  # (27648, 3)
        label = meta['labels']    # (27648, )

        seq = seq[:-1]
        label_seq = label[seq]  # (300,)
        feature_center = center[seq]  # (300, 3)
        feature_geo = feature_geo[:-1].reshape(300, 1)
        feature_dih = feature_dih[:-1].reshape(300, 1)

        feature_seq = np.concatenate((feature_center, feature_geo, feature_dih), axis=1)

        return feature_seq, label_seq


class TestDataset(Dataset):
    """测试集专用，需要遍历所有面片，得到最后的结果"""

    def __init__(self, npz_dir):
        self.pathname_expansion = npz_dir + '/*.npz'
        self.filenames = glob.glob(self.pathname_expansion)

    def __len__(self):
        return len(self.filenames)  # 模型的数量

    def __getitem__(self, indices):
        filename = self.filenames[indices - 1]  # 1到16,每次随机选一个模型
        meta = np.load(filename, encoding='latin1', allow_pickle=True)  # 加载处理好的模型字典

        f0 = random.randint(0, meta["faces"].shape[0])  # 随机选择初始点f0
        seq, jump = get_seq_random_walk_random_global_jumps(meta, f0, 300)
        seq = seq[:-1]

        # feature = meta['face_feature']    # (27648, 628)
        feature = meta['center']  # (27648, 628)
        label = meta['labels']  # (27648, )

        feature_seq = feature[seq]  # (300, 3)
        label_seq = label[seq]  # (300,)

        return feature_seq, label_seq


if __name__ == '__main__':
    path = 'datasets_processed/psb/psb_teddy'
    dataset = PSBDataset(path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for data, label in dataloader:
        pass
