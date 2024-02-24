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


def read_off(off_path):
    """ 读取.off文件，输出顶点v、面片f的列表 """
    f = open(off_path, 'r')
    f.readline()
    num_v, num_f, num_edges = map(int, f.readline().split())
    v_data = []
    for view_id in range(num_v):
        v_data.append(list(map(float, f.readline().split())))
    v_data = np.array(v_data)
    f_data = []
    for face_id in range(num_f):
        f_data.append(list(map(int, f.readline().split()[1:])))
    f_data = np.array(f_data)
    f.close()

    return v_data, f_data


def prepare_edges(mesh):
    """得到mesh的边"""
    vertices = mesh['vertices']
    faces = mesh['faces']
    mesh['edges'] = [set() for _ in range(vertices.shape[0])]
    for i in range(faces.shape[0]):
        for v in faces[i]:
            mesh['edges'][v] |= set(faces[i])
    for i in range(vertices.shape[0]):
        if i in mesh['edges'][i]:
            mesh['edges'][i].remove(i)
        mesh['edges'][i] = list(mesh['edges'][i])
    max_vertex_degree = np.max([len(e) for e in mesh['edges']])
    for i in range(vertices.shape[0]):
        if len(mesh['edges'][i]) < max_vertex_degree:
            mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
    mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)
    return mesh['edges']


def load_data_and_label(data_dir, label_dir, partition, index):
    data_list, label_list = [], []
    mesh = {}
    data_class = range(1, 19)  # 18个类
    # except_idx = [100, 101, 235, 245, 246, 272, 338, 344, 351, 352, 353, 370, 380, 390] # chairLarge
    # train_idx = list(set(range(1,21))-set(except_idx))
    # train_idx = range(13,14)
    test_idx = range(13, 19)  # 6个测试
    train_idx = list(set(data_class) - set(test_idx))

    data_idx = train_idx
    if partition == 'test':
        data_idx = test_idx
    for i in data_idx:
        print(i)
        DATA_DIR = os.path.join(data_dir, str(i) + '.txt')    # 数据文件
        LABEL_DIR = os.path.join(label_dir, str(i) + '.seg')  # 标签文件

        data = np.loadtxt(DATA_DIR)     # (N,628)
        label = np.loadtxt(LABEL_DIR)   # (N,)
        data_list.append(data)          # 累加所有数据，一行一个面片
        label_list.append(label)

        # mesh字典里面有，特征，标签，顶点，面片，边
        mesh[i] = {}
        mesh[i]['features'] = data
        mesh[i]['labels'] = label
        [mesh[i]['vertices'], mesh[i]['faces']] = read_off(os.path.join(label_dir, str(i) + '.off'))
        filename = os.path.join(label_dir, 'edges/' + str(i) + '.txt')
        if not os.path.exists(os.path.join(label_dir, 'edges')):
            os.mkdir(os.path.join(label_dir, 'edges'))
        if not os.path.exists(filename):
            mesh[i]['edges'] = prepare_edges(mesh[i])
            np.savetxt(filename, mesh[i]['edges'], fmt='%d')
        # else:
        #    mesh[i]['edges'] = np.loadtxt(filename,dtype='int')

    return data_list, label_list, mesh


def position_encodng(v, f):
    """后期计算序列的测地线距离作为PE，是一个lxl的矩阵，要下采样适应特征数"""
    l = len(f)           # 面片数 29734,3  估计是点的索引
    position = []        # 位置编码矩阵
    for i in range(l):
        a, b, c = f[i]

        center = v[0].copy()
        center[0] = (v[a - 1][0] + v[b - 1][0] + v[c - 1][0]) / 3  # 三个顶点的坐标求平均
        center[1] = (v[a - 1][1] + v[b - 1][1] + v[c - 1][1]) / 3
        center[2] = (v[a - 1][2] + v[b - 1][2] + v[c - 1][2]) / 3
        position.append(center)
    position = np.array(position)
    # (N,3)
    return position


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
