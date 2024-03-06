import argparse
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import my_dataset
import net
import train

"""
训练和测试分开，因为完整测试比较慢
"""


def get_model_by_name(name):   # 通过名字获得模型
    mesh_data = np.load(name[0], encoding='latin1', allow_pickle=True)
    model = {'faces': mesh_data['faces'], 'labels': mesh_data['labels']}
    return model


def calc_accuracy_test(args):
    """
    算出整个模型的准确率，足够多的walker数量，大概率能游走到
    不行再补一下，再不行就要考虑下采样了
    是否可以每个面片都当一次起始点，全部游走一遍，这样就不会漏了，所有的结果再平均一下,不可以，因为前一半的预测要丢掉
    """
    path = args.data_path + '/' + 'test'
    test_dataset = my_dataset.PSBDataset(args, path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = "cpu"
    src_pad_idx = 0  # 不定长序列中用0填充
    trg_pad_idx = 0
    src_vocab_size = 1024  # 10  输入序列的词汇表大小
    trg_vocab_size = 5  # 10  就五个类别，五个词够了
    model = net.Transformer(args, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device)
    model_path = args.model_path

    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    skip = int(args.seq_len * 0.5)
    models = {}

    n_iters = 100
    for _ in tqdm(range(n_iters)):
        for name, data, label in test_loader:
            data = data.to(torch.float32)  # (1, 32*300, 6)
            label = label.to(dtype=torch.int64)  # (1, 9600)

            data = data.view((-1, 300, 6))  # (32,300,6)
            label = label.view(-1, 300)  # (32, 300)

            all_seq = data[:, :, -1]  # (32, 300)最后一个特征是面片索引
            data = data[:, :, :-1]    # (32, 300, 5)

            pred = model(data, label)
            pred = pred[:, skip:]  # (32, 150, 5) 一个模型上的16条序列

            if name not in models.keys():
                models[name] = get_model_by_name(name)
                models[name]['pred'] = np.zeros((models[name]['faces'].shape[0], args.out_c))     # (面数，分割类别数)
                models[name]['pred_count'] = 1e-6 * np.ones((models[name]['faces'].shape[0],))    # 防止除0

            label = label[:, skip:]
            label = label.reshape(-1)
            pred = pred.reshape(-1, 5)

            # 所有的面片索引

            all_seq = all_seq[:, skip:].reshape(-1).to(torch.int32)   # 4800=32*150

            all_seq = all_seq.numpy()
            pred = pred.detach().numpy()
            for w_step in range(all_seq.shape[0]):   # 4800 都过一遍，这个很慢
                models[name]['pred'][all_seq[w_step]] += pred[w_step]              # 把对应预测的结果加上去
                models[name]['pred_count'][all_seq[w_step]] += 1                   # 加了多少次要记录下来

    for name in models:
        pred = models[name]['pred']
        pred_count = models[name]['pred_count']
        labels = models[name]['labels']
        num = models[name]['faces'].shape[0]

        fin_pred = np.argmax(pred, axis=1)     # 我并不需要知道走到了多少次，最大的那个索引知道就可以了  没走到的点预测要是-1啊

        num_skip_faces = 0  # 有多少点没有走到
        for idx, val in enumerate(pred_count):
            if val < 1:
                num_skip_faces = num_skip_faces + 1
                fin_pred[idx] = -1

        correct = (fin_pred == labels).sum().item()
        full_test_acc = correct * 1.0 / num
        acc = correct * 1.0 / (num-num_skip_faces)

        print(f"{name}的准确率为{full_test_acc}")
        print(f"{name}的去除没走到面的准确率为{acc}")
        print(f"一共有{num}个面片，有{num_skip_faces}个面没有游走到，占比{(num_skip_faces/num):.4f}")
        print("===============================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh Transformer')  # 使用命令行参数
    parser.add_argument('--use_cuda', type=bool, default=False)
    # 数据集相关

    parser.add_argument('--model_path', type=str, default='checkpoints/psb_teddy1/models/model.pth')
    parser.add_argument('--data_path', type=str, default='datasets_processed/psb/psb_teddy')
    parser.add_argument('--n_walks_per_model', type=int, default=32)  #

    parser.add_argument('--full_test', type=bool, default=True)  # 是否把面片索引作为特征加上去

    parser.add_argument('--exp_name', type=str, default='psb_teddy')
    parser.add_argument('--save_path', type=str, default='result/')  # 计划放验证模式输出的整个模型标签
    parser.add_argument('--off_path', type=str, default='E:/3DModelData/PSB/Teddy/', help='off path')

    # 神经网络相关
    parser.add_argument('--feature_num', type=int, default=5, help='Number of features')  # 消融实验
    parser.add_argument('--out_c', type=int, default=5, help='classes of this shape')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum ')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--pre_train', type=bool, default=False,
                        help='use Pretrained model')  # 继续训练

    # transformer 参数
    parser.add_argument('--src_pad_idx', type=int, default=0, help='原序列不足用0填充')
    parser.add_argument('--trg_pad_idx', type=int, default=0, help='标签序列不足用0填充')
    parser.add_argument('--src_vocab_size', type=int, default=1024, help='输入序列的词汇表大小')
    parser.add_argument('--trg_vocab_size', type=int, default=5, help='用类别数')

    # 序列相关
    parser.add_argument('--seq_len', type=int, default=300)  # 做消融实验，不一定是300，我的模型挺大的

    # parser.add_argument('--index', type=int, default=0, help='which class to train')
    # parser.add_argument('--attention_layers', type=int, default=0)
    # parser.add_argument('--input_layer', type=tuple, default=(628, 1024, 256, 16))
    # parser.add_argument('--offset', type=bool, default=True)
    args = parser.parse_args()

    calc_accuracy_test(args)
