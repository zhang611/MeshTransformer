import argparse
import os
import time
import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import my_dataset
import net
from utils import IOStream


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    for file in os.listdir():
        if file.endswith('py'):
            pass
            # os.system(f'cp {file} checkpoints' + '/' + args.exp_name + '/' + f'{file}.backup')


def train(args, io):
    # 加载数据集
    path_train = args.data_path
    dataset = my_dataset.PSBDataset(args, path_train)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  # 一次看四个模型

    path_test = args.data_path + '/' + 'test'
    test_dataset = my_dataset.PSBDataset(args, path_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)  # 都是4

    device = args.device

    # 创建模型
    # model = zzc_net.MLP(args, out_c=args.out_c).to(device)

    src_vocab_size = args.src_vocab_size
    trg_vocab_size = args.trg_vocab_size
    src_pad_idx = args.src_pad_idx
    trg_pad_idx = args.trg_pad_idx
    model = net.Transformer(args, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device)

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

    best_test_acc = 0.0    # 是否保存神经网络比较的时候用 TODO：是不是多看几条序列决定是否保存
    best_epoch = 0
    print("Start Training")
    print(datetime.datetime.now())
    for epoch in range(args.epochs):
        train_loss = 0.0    # 记录每一轮训练的累计loss
        count = 0.0         # 一个batch看4个模型，4条序列，每次加16
        model.train()
        total_time = 0.0
        correct = 0.0       # 记录一轮的累计准确率
        num = 0             # 使用的面片数量，算准确率使用
        for data, label in dataloader:
            data = data.to(torch.float32)         # (4, 4*300, 5) 4个模型，每个模型取4条长300的序列，5个特征
            label = label.to(dtype=torch.int64)   # (4, 1200)
            data, label = data.to(device), label.to(device)

            # 进入网络之前要把(4, 1200, 5)->(16, 300, 5)网络是根据序列来的
            data = data.view((-1, 300, 5))
            label = label.view(-1, 300)
            opt.zero_grad()
            start_time = time.time()
            # 网络的输入(16, 300, 5)  标签(16, 300)
            pred = model(data, label)   # 输出 (16, 300, 5)
            # loss = criterion(pred, label)
            pred = pred.view(-1, 5)   # (16*300, 5)  方便后面算loss，算准确率
            label = label.view(-1)    # (16*300, 5)
            loss = criterion(pred, label)
            loss.backward()
            opt.step()    # 更新优化器
            end_time = time.time()   # 一次iter花的时间
            total_time += (end_time - start_time)

            preds = pred.max(dim=1)[1]   # (1204,)  one-hot变回数值标签，方便算准确率
            count += 1 * args.batch_size * args.n_walks_per_model  # 4*4=16，一次看十六条序列
            train_loss += loss.item() * args.batch_size   # 一轮累计的loss
            num += data.shape[0] * data.shape[1]          # 看的面片数量，4*4*300=4800
            correct += (preds == label).sum().item()   # 预测准确的面片数,算的是这一轮的累计准确率

        test_acc = correct * 1.0 / num    # 没有加权面积
        res["train_loss"].append((train_loss * 1.0 / count))
        res["train_acc"].append(test_acc)     # 这个res最后画图用的
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
                data = data.to(torch.float32)
                label = label.to(dtype=torch.int64)
                data, label = data.to(device), label.to(device)

                data = data.view((-1, 300, 5))
                label = label.view(-1, 300)

                opt.zero_grad()
                pred = model(data, label)
                pred = pred.view(-1, 5)   # (4800, 5)
                label = label.view(-1)    # (4800)
                loss = criterion(pred, label)
                preds = pred.max(dim=1)[1]
                test_loss += loss
                count += 1 * args.batch_size * args.n_walks_per_model
                test_loss += loss.item() * args.batch_size
                num += data.shape[0] * data.shape[1]
                correct += (preds == label).sum().item()

            test_acc = correct * 1.0 / num
            if test_acc >= best_test_acc:   # 用测试集上效果最好，是不是可以把序列数变大一点
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
    print(datetime.datetime.now())


if __name__ == '__main__':
    ############################################################
    # 调参
    ############################################################
    parser = argparse.ArgumentParser(description='Mesh Transformer')  # 使用命令行参数
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--full_test', type=bool, default=False)  # 是否把面片索引作为特征加上去,训练模式


    # 数据集相关
    parser.add_argument('--data_path', type=str, default='datasets_processed/psb/psb_ant')
    parser.add_argument('--exp_name', type=str, default='psb_ant')
    parser.add_argument('--model_path', type=str, default='checkpoints/psb_ant/models/model.pth',
                        help='Pretrained model path')  # 验证模式使用
    parser.add_argument('--off_path', type=str, default='E:/3DModelData/PSB/Ant/', help='off path')

    # 神经网络相关
    parser.add_argument('--feature_num', type=int, default=5, help='Number of features')  # 消融实验
    parser.add_argument('--out_c', type=int, default=5, help='classes of this shape')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=4)    # 一次看四个模型
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
    parser.add_argument('--n_walks_per_model', type=int, default=4)  # 4*4=16，每次16条序列

    # parser.add_argument('--index', type=int, default=0, help='which class to train')
    # parser.add_argument('--attention_layers', type=int, default=0)
    # parser.add_argument('--input_layer', type=tuple, default=(628, 1024, 256, 16))
    # parser.add_argument('--offset', type=bool, default=True)
    args = parser.parse_args()

    ############################################################
    # 调参结束
    ############################################################
    _init_()
    res = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    io = IOStream('checkpoints/' + args.exp_name + '/0_run.log')
    io.cprint(str(args))

    if args.use_cuda and torch.cuda.is_available():
        args.device = "cuda"
        io.cprint('Using GPU')
    else:
        args.device = "cpu"
        io.cprint('Using CPU')

    train(args, io)
