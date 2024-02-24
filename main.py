from __future__ import print_function
import os
import argparse
import time
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset import PsbDataset
from model import Pct_seg
from util import cal_loss, IOStream
# import model_new

res = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')  # 备份文件
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    train_loader = DataLoader(    # 训练集的12个模型的所有面片都加载进来，dataset是12个元素的列表，第一个元素（27648,628）
        PsbDataset(partition='train', off_path=args.off_path, data_path=args.data_path, label_path=args.label_path,
                   index=args.index), batch_size=args.batch_size, num_workers=0, shuffle=True)
    valid_loader = DataLoader(
        PsbDataset(partition='test', off_path=args.off_path, data_path=args.data_path, label_path=args.label_path,
                   index=args.index), batch_size=args.batch_size, num_workers=0, shuffle=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct_seg(args, out_c=args.out_c).to(device)
    if args.pre_train:
        model.load_state_dict(torch.load(args.model_path))
        print('load model successfully')
        # print(str(model))
        # model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 15, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.1)

    criterion = torch.nn.CrossEntropyLoss()
    best_test_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        # scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        # idx = 0
        total_time = 0.0
        correct = 0.0
        num = 0
        for data, label in train_loader:  # 8号模型，data(1, 25118, 628) lable(1, 25118)
            data, label = data.to(device), label.to(device)
            # print('data:',data.shape) #(2,3000,628)
            # print('label:',label.shape) #(2,3000)
            # print(idx_in_all.item())

            opt.zero_grad()
            start_time = time.time()
            pred = model(data)  # (1, 25118, 5)
            # pred = model(data,idx_in_all.item())
            # print('pred',pred.shape) #(2,3000,8)
            pred = pred.permute(0, 2, 1)  # (1, 5, 25118)
            loss = criterion(pred, label)

            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            # print('pred',pred.shape)
            preds = pred.max(dim=1)[1]   # (1, 25118)  预测结果变成lable形式
            # print('preds',preds.shape) # (2,3000)
            count += 1 * args.batch_size      # 一个batch就是一个模型，12345这样增加
            train_loss += loss.item() * args.batch_size
            num += data.shape[0] * data.shape[1]       # 25118 一个模型的面片数，1*25118
            correct += (preds == label).sum().item()   # 预测准确的面片数

            # print(correct)
            # train_true.append(label.cpu().numpy())
            # train_pred.append(preds.detach().cpu().numpy())

        # print ('train total time is',total_time)
        # train_true = np.concatenate(train_true)
        # train_pred = np.concatenate(train_pred)
        test_acc = correct * 1.0 / num
        res["train_loss"].append((train_loss * 1.0 / count))
        res["train_acc"].append(test_acc)
        outstr = '[Epoch %d] [Train: loss: %.6f, acc: %.6f]' % (epoch,
                                                                train_loss * 1.0 / count,
                                                                test_acc)

        io.cprint(outstr, end_char='')
        # print(outstr, end='')
        '''
        if test_acc >= best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'checkpoints/%s/models/model.pth' % args.exp_name)
        '''

        ####################
        # Validation
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        correct = 0.0
        num = 0
        total_time = 0.0
        # filenum = 1
        filenum = 1  # 验证集模型编号

        with torch.no_grad():
            for data, label in valid_loader:
                data, label = data.to(device), label.to(device)
                # print(idx,type(idx)) tensor
                # print(data.shape) (1,9408,628)
                # print('test_index',idx_in_all.item())
                opt.zero_grad()
                start_time = time.time()
                # pred = model(data,idx_in_all.item())
                pred = model(data)
                # print(pred.shape)
                pred = pred.permute(0, 2, 1)
                # end_time = time.time()
                # total_time += (end_time - start_time)
                loss = criterion(pred, label)
                preds = pred.max(dim=1)[1]
                # print('pred:'+str(pred.shape)+', label:'+str(label.shape)+', preds:'+str(preds.shape))
                test_loss += loss
                count += 1 * args.batch_size
                test_loss += loss.item() * args.batch_size

                num += data.shape[0] * data.shape[1]

                correct += (preds == label).sum().item()

                if epoch > 5:   # 每5轮保存模型
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    save_dir = args.save_path + str(epoch) + '/'
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    f = open(save_dir + str(filenum) + '.seg', 'w')
                    pres = preds.cpu().numpy()
                    pres = pres.squeeze(0)
                    pres = pres.tolist()
                    str_out = [str(x_i) for x_i in pres]
                    strs_out = '\n'.join(str_out)
                    f.write(strs_out)
                    f.close()

                    f = open(save_dir + str(filenum) + '.prob', 'w')
                    prob = pred.cpu().numpy()
                    prob = prob.squeeze(0)
                    # prob = pred.tolist()
                    for i in prob:
                        for j in i:
                            f.write(str(j))
                            f.write(' ')
                        f.write('\n')
                    f.close()

                    filenum = filenum + 1
            # print ('test total time is', total_time)
            # test_true = np.concatenate(test_true)
            # test_pred = np.concatenate(test_pred)
            test_acc = correct * 1.0 / num
            # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

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
            # print(type(res))
            # res = res.cuda().data.cpu().numpy()
            # plot_history(res)

    # plot_history(res)

    # torch.save(model.state_dict(), 'checkpoints/%s/models/model.pth' % args.exp_name)


def test(args, io):
    train_loader = DataLoader(
        PsbDataset(partition='train', off_path=args.off_path, data_path=args.data_path, label_path=args.label_path,
                   index=args.index),
        batch_size=1, num_workers=1)
    test_loader = DataLoader(
        PsbDataset(partition='test', off_path=args.off_path, data_path=args.data_path, label_path=args.label_path,
                   index=args.index),
        batch_size=1, num_workers=1)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct_seg(args, out_c=args.out_c).to(device)
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    print(outstr)


def plot_history(res):
    plt.subplot(211)
    val_loss = res['val_loss']
    print(type(val_loss))
    plt.plot(val_loss, label=['val_loss'])
    plt.plot(res['train_loss'], label=['train_loss'])
    plt.legend(loc='upper right')
    plt.savefig('F:/zym/zym/checkpoints/loss.jpg')
    # plt.show()

    plt.subplot(212)
    plt.plot(res['val_acc'], label=['val_acc'])
    plt.plot(res['train_acc'], label=['train_acc'])
    plt.legend(loc='lower right')
    plt.savefig('F:/zym/zym/checkpoints/acc.jpg')
    plt.show()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Mesh Segmentation')

    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    # [0]Human-8 [1]Cup-2   [2]Glasses-3 [3]Airplane-5 [4]Ant-5 [5]Chair-4 [6]Octopus-2 [7]Table-2 [8]Teddy-5
    # [9]Hand-6 [10]Plier-3 [11]Fish-3 [12]Bird-5 [14]Armadillo-11 [18]Vase-5 [19]Fourleg-6

    # 需要调的超参
    parser.add_argument('--data_path', type=str, default='E:/3DModelData/PSB/Teddy/Features/', help='path of dataset')
    parser.add_argument('--label_path', type=str, default='E:/3DModelData/PSB/Teddy/', help='path of label')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N')
    parser.add_argument('--exp_name', type=str, default='Teddy_test', metavar='N', help='log_dir')
    parser.add_argument('--index', type=int, default=0, help='which class to train')
    parser.add_argument('--out_c', type=int, default=5, help='classes of this shape')
    parser.add_argument('--attention_layers', type=int, default=0)
    parser.add_argument('--input_layer', type=tuple, default=(628, 1024, 256, 16))
    parser.add_argument('--save_path', type=str, default='G:/zhangzhichao/zym/result')
    parser.add_argument('--offset', type=bool, default=True)

    parser.add_argument('--model_path', type=str, default='G:/zhangzhichao/zym/checkpoints/Teddy/models/model.pth',
                        metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--off_path', type=str, default='E:/3DModelData/PSB/Teddy/',
                        help='off path')
    parser.add_argument('--pre_train', type=bool, default=False,
                        help='use Pretrained model')

    args = parser.parse_args()

    _init_()

    res = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        # torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
        # plot_history(res)
    else:
        test(args, io)
