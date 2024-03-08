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
    path = args.data_path + '/' + 'full_test'
    test_dataset = my_dataset.PSBDataset(args, path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = "cpu"
    model = net.Transformer(args).to(args.device)

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()

    skip = int(args.seq_len * 0.5)
    models = {}

    n_iters = 10
    for _ in tqdm(range(n_iters)):
        for name, data, label in test_loader:
            data = data.to(torch.float32)  # (1, 32*300, 6)
            label = label.to(dtype=torch.int64)  # (1, 9600)

            data = data.view((-1, 300, 6))  # (32,300,6)
            label = label.view(-1, 300)  # (32, 300)

            all_seq = data[:, :, -1]  # (32, 300)最后一个特征是面片索引
            data = data[:, :, :-1]    # (32, 300, 5)

            data, label = data.to(args.device), label.to(args.device)
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

            all_seq = all_seq[:, skip:].reshape(-1).to(torch.int32)   # 4800=32*150  一次看32条序列

            all_seq = all_seq.numpy()
            pred = pred.cpu()
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

        print("===============================================================================")
        print(f"{name}的准确率为{full_test_acc}")
        print(f"{name}的去除没走到面的准确率为{acc}")
        print(f"一共有{num}个面片，有{num_skip_faces}个面没有游走到，占比{(num_skip_faces/num):.4f}")
        print("===============================================================================")

        # 看一下有多少面没有游走到
        # val_label = np.zeros(num)  # 初始化所有面的标签，都为0
        # for idx, val in enumerate(pred_count):
        #     if val > 0.5:
        #         val_label[idx] = 1
        # np.savetxt('result/test/10times_20.seg', val_label, fmt='%d', delimiter='\t')

        # 保存结果
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        save_dir = args.save_path + '/' + str('ant') + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        f = open(save_dir + str(20) + '.seg', 'w')
        pres = fin_pred.tolist()
        str_out = [str(x_i) for x_i in pres]
        strs_out = '\n'.join(str_out)
        f.write(strs_out)
        f.close()

        # 图割使用
        f = open(save_dir + str(20) + '.prob', 'w')
        # prob = pred.cpu().numpy()
        # prob = pred.squeeze(0)
        # prob = pred.tolist()
        for i in pred:   # 一行一行的写
            for j in i:
                f.write(str(j))
                f.write(' ')
            f.write('\n')
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh Transformer')  # 使用命令行参数
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cuda")
    # 数据集相关

    parser.add_argument('--model_path', type=str, default='checkpoints/psb_ant/models/model.pth')
    parser.add_argument('--data_path', type=str, default='datasets_processed/psb/psb_ant')
    parser.add_argument('--n_walks_per_model', type=int, default=32)  #

    parser.add_argument('--full_test', type=bool, default=True)  # 是否把面片索引作为特征加上去

    parser.add_argument('--exp_name', type=str, default='psb_ant')
    parser.add_argument('--save_path', type=str, default='result')  # 计划放验证模式输出的整个模型标签
    parser.add_argument('--off_path', type=str, default='E:/3DModelData/PSB/Ant/', help='off path')

    # 神经网络相关
    parser.add_argument('--feature_num', type=int, default=5, help='Number of features')
    parser.add_argument('--out_c', type=int, default=5, help='classes of this shape')
    parser.add_argument('--batch_size', type=int, default=4)


    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--forward_expansion', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=300)

    parser.add_argument('--pre_train', type=bool, default=False,
                        help='use Pretrained model')  # 继续训练

    # transformer 参数
    parser.add_argument('--src_pad_idx', type=int, default=0, help='原序列不足用0填充')
    parser.add_argument('--trg_pad_idx', type=int, default=0, help='标签序列不足用0填充')
    parser.add_argument('--src_vocab_size', type=int, default=1024, help='输入序列的词汇表大小')
    parser.add_argument('--trg_vocab_size', type=int, default=5, help='用类别数')

    # 序列相关
    parser.add_argument('--seq_len', type=int, default=300)  # 做消融实验，不一定是300，我的模型挺大的

    args = parser.parse_args()

    calc_accuracy_test(args)
