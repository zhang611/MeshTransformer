import argparse
import os
import time

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch import Tensor
from torch import nn, optim

import zzc_dataset
import zzc_net
from util import IOStream


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp zzc_dataprepare.py checkpoints' + '/' + args.exp_name + '/' + 'zzc_dataprepare.py.backup')  # 备份文件


def train(args, io):
    # 加载数据集
    # dataset = zzc_dataset.PSBDataset('matlab/data/PSB/teddy')
    # path = 'datasets_processed/psb/psb_teddy'
    path1 = args.data_path
    dataset = zzc_dataset.PSBDataset(path1)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    path2 = args.data_path + '/' + 'test'
    test_dataset = zzc_dataset.PSBDataset(path2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)  # 都是4

    device = torch.device("cuda" if args.use_cuda else "cpu")

    # 创建模型
    # model = zzc_net.MLP(args, out_c=args.out_c).to(device)
    src_pad_idx = 0   # 不定长序列中用0填充
    trg_pad_idx = 0
    src_vocab_size = 1024        # 10  输入序列的词汇表大小
    trg_vocab_size = 5           # 10  就五个类别，五个词够了
    model = zzc_net.Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device)

    if args.pre_train:
        model.load_state_dict(torch.load(args.model_path))
        print('load model successfully')

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 15, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)  # 内部自动转化为onehot，标签一定要是LongTensor类型也就是int64

    best_test_acc = 0.0    # 是否保存神经网络比较的时候用
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss = 0.0    # 记录每一轮训练的累计loss
        count = 0.0         # 一个batch看4个模型，四条序列，每次加4
        model.train()
        total_time = 0.0
        correct = 0.0       # 记录一轮的累计准确率
        num = 0             # 序列的面片数量
        for data, label in dataloader:     # 一个batch的面片data(4, 300, 3) label(4, 300)

            # 如果要用MLP输入就要是(1200, 628)，用RNN再考虑顺序
            # data = data.view(-1, 3)
            # data = torch.tensor(data, dtype=torch.float32)
            data = data.to(torch.float32)
            # label = label.view(-1)
            label = label.to(dtype=torch.int64)
            data, label = data.to(device), label.to(device)

            opt.zero_grad()
            start_time = time.time()
            pred = model(data, label)   # (4, 300, 5)
            # loss = criterion(pred, label)
            pred = pred.view(-1, 5)
            label = label.view(-1)
            loss = criterion(pred, label)
            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)

            preds = pred.max(dim=1)[1]   # (1204,)  预测结果变成lable形式
            count += 1 * args.batch_size  # 看的模型数，16，也就是在16个模型上游走了16条序列
            train_loss += loss.item() * args.batch_size   # 一轮累计的loss
            num += data.shape[0] * data.shape[1]           # 1200
            correct += (preds == label).sum().item()   # 预测准确的面片数,算的是这一轮的累计准确率

        test_acc = correct * 1.0 / num    # 准确面片数、序列面片数（4*300），没有加权面积
        res["train_loss"].append((train_loss * 1.0 / count))
        res["train_acc"].append(test_acc)
        outstr = ('[Epoch %d] [Train: loss: %.6f, acc: %.6f, time: %.6f]' %
                  (epoch, train_loss * 1.0 / count, test_acc, total_time))
        io.cprint(outstr, end_char='')
        # print('[Epoch %d] [Train: loss: %.6f, acc: %.6f]' % (epoch, train_loss, test_acc))  # 别的地方打印了

        ############################
        # 验证
        ############################
        test_loss = 0.0
        count = 0.0
        correct = 0.0
        model.eval()
        num = 0

        with torch.no_grad():
            for data, label in test_loader:
                # data = data.view(-1, 3)
                # data = torch.tensor(data, dtype=torch.float32)
                data = data.to(torch.float32)
                # label = label.view(-1)
                label = label.to(dtype=torch.int64)
                data, label = data.to(device), label.to(device)

                opt.zero_grad()
                pred = model(data, label)
                pred = pred.view(-1, 5)
                label = label.view(-1)
                loss = criterion(pred, label)
                preds = pred.max(dim=1)[1]
                test_loss += loss
                count += 1 * args.batch_size
                test_loss += loss.item() * args.batch_size
                num += data.shape[0] * data.shape[1]
                correct += (preds == label).sum().item()

            test_acc = correct * 1.0 / num
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.pth' % args.exp_name)
                best_epoch = epoch

            res["val_loss"].append((test_loss * 1.0 / count))
            res["val_acc"].append(test_acc)
            outstr = ' [Val: loss: %.6f, acc: %.6f] [Best acc: %.6f (Epoch %d)]' % (
                test_loss * 1.0 / count, test_acc, best_test_acc, best_epoch)
            # outstr = 'Test %d, test acc: %.6f' % (epoch,test_acc)
            io.cprint(outstr)

    print('Training Finished!')


def test(args, io):
    """
    测试的时候需要算出整个模型的准确率，大数准则还是怎么办
    是否可以，每个面片都当一次起始点，全部游走一遍，这样就不会漏了，所有的结果再平均一下
    """
    path = args.data_path + '/' + 'test'
    test_dataset = zzc_dataset.PSBDataset(path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    # model = zzc_net.MLP(args, out_c=args.out_c).to(device)
    src_pad_idx = 0   # 不定长序列中用0填充
    trg_pad_idx = 0
    src_vocab_size = 1024        # 10  输入序列的词汇表大小
    trg_vocab_size = 5           # 10  就五个类别，五个词够了
    model = zzc_net.Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data = data.view(-1, 3)
        data = torch.tensor(data, dtype=torch.float32)
        label = label.view(-1)
        label = label.to(dtype=torch.int64)
        data, label = data.to(device), label.to(device)

        pred = model(data)
        preds = pred.max(dim=1)[1]
        pass






if __name__ == '__main__':
    # 超参数
    parser = argparse.ArgumentParser(description='Mesh Segmentation')  # 使用命令行参数
    parser.add_argument('--use_cuda', type=bool, default=False)
    # 实验目录相关
    parser.add_argument('--data_path', type=str, default='datasets_processed/psb/psb_teddy')
    parser.add_argument('--exp_name', type=str, default='psb_teddy')
    parser.add_argument('--save_path', type=str, default='result/')  # 计划放验证模式输出的整个模型标签
    parser.add_argument('--model_path', type=str, default='checkpoints/Teddy/models/model.pth',
                        help='Pretrained model path')  # 验证模式使用
    parser.add_argument('--off_path', type=str, default='E:/3DModelData/PSB/Teddy/', help='off path')

    # 神经网络相关
    parser.add_argument('--out_c', type=int, default=5, help='classes of this shape')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--eval', type=bool, default=False)  # 是否是验证模式，一般就是训练
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum ')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--pre_train', type=bool, default=False,
                        help='use Pretrained model')  # 继续训练

    # 序列相关
    parser.add_argument('--seq_len', type=int, default=300)

    # parser.add_argument('--index', type=int, default=0, help='which class to train')
    # parser.add_argument('--attention_layers', type=int, default=0)
    # parser.add_argument('--input_layer', type=tuple, default=(628, 1024, 256, 16))
    # parser.add_argument('--offset', type=bool, default=True)
    args = parser.parse_args()

    _init_()
    res = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
        # plot_history(res)
    else:
        test(args, io)




