import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import Tensor
from torch import nn, optim

import zzc_dataset
import zzc_net


def train():
    # 加载数据集
    dataset = zzc_dataset.PSBDataset('matlab/data/PSB/teddy')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 创建模型
    model = zzc_net.MLP()
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda(0)  # 内部自动转化为onehot，标签一定要是LongTensor类型也就是int64
    optimizer = optim.SGD(model.parameters(), lr=0.003)

    for epoch in range(100 + 1):
        train_loss = 0.0
        model.train()
        correct = 0.0
        seq_len = 300
        for data, label in dataloader:     # 一个batch的面片data(4, 301, 628) label(4, 301)
            # data, label = data.cuda(), label.cuda()

            # 如果要用MLP输入就要是(1200, 628)，用RNN再考虑顺序
            data = data.view(-1, 628)
            data = torch.tensor(data, dtype=torch.float32)
            label = label.view(-1)
            label = label.to(dtype=torch.int64)

            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            preds = pred.max(dim=1)[1]   # (1204,)  预测结果变成lable形式
            train_loss += loss.item()
            correct += (preds == label).sum().item()   # 预测准确的面片数
        print('[Epoch %d] [Train: loss: %.6f, acc: %.6f]' % (epoch, train_loss, correct/1204))
    print('Training Finished!')


if __name__ == '__main__':
    train()



