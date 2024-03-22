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
import new_net
import train

"""
训练和测试分开，因为完整测试比较慢
"""


def get_model_by_name(name):  # 通过名字获得模型
    mesh_data = np.load(name[0], encoding='latin1', allow_pickle=True)
    model = {'faces': mesh_data['faces'], 'labels': mesh_data['labels'], 'ring': mesh_data['ring']}
    return model


def postprocess(models, scale):
    """平均周围面片的预测结果，直接修改models字典里面模型的pred"""
    for model_name, model in models.items():
        pred_orig = model['pred'].copy()  # (14698, 5)
        av_pred = np.zeros_like(pred_orig)  # 初始化平均结果
        for v in range(model['faces'].shape[0]):  # 看每个顶点周围几个顶点的结果，平均
            this_pred = pred_orig[v]     # 当前顶点的预测结果
            nbrs_ids = model['ring'][v]  # 周围一圈顶点的索引,从0开始，一个个来
            nbrs_ids = np.int64(nbrs_ids)
            nbrs_ids = np.array([n for n in nbrs_ids if n != -1])  # 没游走到的点是-1去掉
            if nbrs_ids.size:
                # first_ring_pred = (pred_orig[nbrs_ids].T / model['pred_count'][nbrs_ids]).T  # 除过了
                # nbrs_pred = np.mean(first_ring_pred, axis=0) * scale
                nbrs_pred = np.mean(pred_orig[nbrs_ids], axis=0) * scale
                av_pred[v] = this_pred + nbrs_pred
            else:
                av_pred[v] = this_pred
        model['pred'] = av_pred  # 预测结果变成平均的结果


def calc_accuracy_test(args):
    """
    算出整个模型的准确率，足够多的walker数量
    前一半的预测要丢掉
    """
    data_path = 'datasets_processed/' + os.path.split(args.exp_name)[1].split('_')[0] + '/' + args.exp_name
    path = data_path + '/' + 'full_test'
    test_dataset = my_dataset.PSBDataset(args, path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = "cpu"
    # model = net.Transformer(args).to(args.device)
    model = new_net.Transformer(args).to(args.device)

    model_path = 'checkpoints/%s/models/model.pth' % args.exp_name
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    skip = int(args.seq_len * 0.5)  # 前面一半不要
    models = {}

    n_iters = 100
    for _ in tqdm(range(n_iters)):
        for name, data, label in test_loader:
            data = data.to(torch.float32)  # (1, 32*300, 6)
            label = label.to(dtype=torch.int64)  # (1, 9600)

            data = data.view((-1, 300, 6))  # (32,300,6)
            label = label.view(-1, 300)  # (32, 300)

            all_seq = data[:, :, -1]  # (32, 300)最后一个特征是面片索引
            data = data[:, :, :-1]  # (32, 300, 5)

            data, label = data.to(args.device), label.to(args.device)
            # pred = model(data, label)
            pred = model(data)  # (32, 300, 5)
            pred = pred[:, skip:]  # (32, 150, 5) 一个模型上的16条序列

            if name not in models.keys():
                models[name] = get_model_by_name(name)
                models[name]['pred'] = np.zeros((models[name]['faces'].shape[0], args.out_c))  # (面数，分割类别数)
                models[name]['pred_count'] = 1e-6 * np.ones((models[name]['faces'].shape[0],))  # 走过的次数，防止除0

            label = label[:, skip:]
            label = label.reshape(-1)
            pred = pred.reshape(-1, 5)  # (4800, 5)

            # 游走到的面片的索引

            all_seq = all_seq[:, skip:].reshape(-1).to(torch.int32)  # 4800=32*150  一次看32条序列

            all_seq = all_seq.numpy()
            pred = pred.cpu()
            pred = pred.detach().numpy()
            for w_step in range(all_seq.shape[0]):  # 4800 都过一遍，这个很慢
                models[name]['pred'][all_seq[w_step]] += pred[w_step]  # 把对应预测的结果加上去
                models[name]['pred_count'][all_seq[w_step]] += 1  # 加了多少次要记录下来
            # models[name]['pred'] = (models[name]['pred'].T / models[name]['pred_count']).T   # 提前把次数除掉准确率会降低

    # 最直接的结果
    for name in models:
        pred = models[name]['pred']  # (14698, 5) 这是所有结果累加的
        pred_count = models[name]['pred_count']
        labels = models[name]['labels']
        num = models[name]['faces'].shape[0]  # 14698
        # pred = (pred.T / pred_count).T  # 把次数给除掉，反正后面也要归一化，不是很有必要把次数除掉
        pred_norm = np.linalg.norm(pred, axis=1, keepdims=True)  # 计算矩阵的范数
        pred = pred / pred_norm  # 归一化
        # 先这样，后面再改，之前是-1到0之间，越接近0越准，现在是0到1之间，接近1的准
        pred = pred + 1
        pred_norm2 = np.linalg.norm(pred, axis=1, keepdims=True)  # 计算矩阵的范数
        pred = pred / pred_norm2  # 归一化

        fin_pred = np.argmax(pred, axis=1)  # 算结果不需要平均，最大的那个索引知道就可以了，但是prob有问题

        num_skip_faces = 0  # 有多少点没有走到
        for idx, val in enumerate(pred_count):
            if val < 1:
                num_skip_faces = num_skip_faces + 1
                fin_pred[idx] = -1

        correct = (fin_pred == labels).sum().item()
        full_test_acc = correct * 1.0 / num
        acc = correct * 1.0 / (num - num_skip_faces)

        print("===============================================================================")
        print(f"{name}的准确率为{full_test_acc}")
        print(f"{name}的去除没走到面的准确率为{acc}")
        print(f"一共有{num}个面片，有{num_skip_faces}个面没有游走到，占比{(num_skip_faces / num):.4f}")
        print("===============================================================================")

        # 保存结果
        # save_dir = args.save_path + '/' + args.exp_name + '/'
        save_dir = args.save_path + '/' + 'test' + '/'
        saveResult(save_dir, fin_pred, pred)

        # 后处理

    # 后处理后的结果
    postprocess(models, 2)
    for name in models:
        pp_pred = models[name]['pred']  # (14698, 5)
        labels = models[name]['labels']
        num = models[name]['faces'].shape[0]  # 14698
        pp_pred_norm = np.linalg.norm(pp_pred, axis=1, keepdims=True)  # 计算矩阵的范数
        pp_pred = pp_pred / pp_pred_norm  # 归一化
        # 先这样，后面再改，之前是-1到0之间，越接近0越准，现在是0到1之间，接近1的准
        pp_pred = pp_pred + 1
        pp_pred_norm2 = np.linalg.norm(pp_pred, axis=1, keepdims=True)  # 计算矩阵的范数
        pp_pred = pp_pred / pp_pred_norm2  # 归一化

        pp_fin_pred = np.argmax(pp_pred, axis=1)  # 算结果不需要平均，最大的那个索引知道就可以了，但是prob有问题
        pp_correct = (pp_fin_pred == labels).sum().item()
        pp_full_test_acc = pp_correct * 1.0 / num

        print("===============================================================================")
        print(f"{name}后处理后的准确率为{pp_full_test_acc}")
        print("===============================================================================")
        path = 'result/test2/'
        saveResult(path, pp_fin_pred, pp_pred)


def saveResult(save_dir, seg, pred):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f = open(save_dir + str(20) + '.seg', 'w')
    pres = seg.tolist()  # 单值标签
    str_out = [str(x_i) for x_i in pres]
    strs_out = '\n'.join(str_out)  # 变成一行行的
    f.write(strs_out)
    f.close()

    # 图割使用
    f = open(save_dir + str(20) + '.prob', 'w')
    for i in pred:  # 一行一行的写
        for j in i:
            f.write(str(j))
            f.write(' ')
        f.write('\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh Transformer')  # 使用命令行参数
    parser.add_argument('--device', type=str, default="cuda")  # cuda or cpu
    parser.add_argument('--pattern', type=str, default='test')  # 把面片索引作为特征加上去

    # 实验相关
    parser.add_argument('--exp_name', type=str, default='psb_airplane')
    parser.add_argument('--save_path', type=str, default='result')  # 计划放验证模式输出的整个模型标签

    parser.add_argument('--n_walks_per_model', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=300)  # 做消融实验，不一定是300，我的模型挺大的

    # 网络相关
    parser.add_argument('--feature_num', type=int, default=5, help='Number of features')
    parser.add_argument('--out_c', type=int, default=5, help='classes of this shape')

    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--forward_expansion', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=int, default=0)

    # transformer 参数
    parser.add_argument('--src_pad_idx', type=int, default=0, help='原序列不足用0填充')
    parser.add_argument('--trg_pad_idx', type=int, default=0, help='标签序列不足用0填充')
    parser.add_argument('--src_vocab_size', type=int, default=1024, help='输入序列的词汇表大小')
    parser.add_argument('--trg_vocab_size', type=int, default=5, help='用类别数')

    # 序列相关

    args = parser.parse_args()

    calc_accuracy_test(args)
