from torch.utils.data import Dataset
import torch
import numpy as np
import os

'''
0.把一个类中所有的面片都集合在一起，每次从中取2000个面片作为一组

1.每次输出一个的面片、label,输出形式是np.array
面片形状(N,628),label形状(N,)

2.如何区分不同的类别
在调用时传入参数index指明当前训练/测试的类别
参数partition指明是训练/测试

3.load_data_and_label函数
函数接受两个目录，分别指明训练集和标签，同时需要指明模式和类别索引
返回两个列表，列表中的元素均为np.array对象

4.还没有处理260~279的无效输入

'''








class PsbDataset(Dataset):
    def __init__(self, off_path, data_path, label_path, partition='train', index=0):
        self.data, self.label, self.mesh = load_data_and_label(data_path, label_path, partition, index)
        self.index = index
        self.partition = partition
        self.off_path = off_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.tensor(data, dtype=torch.float32)

        label = self.label[idx]
        label = torch.tensor(label, dtype=torch.long)

        index_in_all = idx + 1 + self.index * 20
        if self.partition == 'test':
            index_in_all += 12
        # off_dir=os.path.join(self.off_path, str(index_in_all)+'_normalized.off')
        # v,f=read_off(off_dir)
        # position=position_encoding(v,f)
        # position=torch.Tensor(position)
        # print(position.shape,data.shape)
        # position=torch.unsqueeze(position,dim=0)
        # data=torch.cat([data,position],dim=1)
        '''
        if self.partition=='test':
            index_in_all+=12
        '''
        return data, label
