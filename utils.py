import torch
import torch.nn.functional as F
import numpy as np


# import matplotlib.pyplot as plot


def get_label_num(path):
    # 输出模型的类别数
    # path = r'datasets_processed/HumanBodySegmentation/HumanBodySegmentation_train/123_not_changed.npz'
    mesh_data = np.load(path, encoding='latin1', allow_pickle=True)
    max_num_class = mesh_data['labels'].max()
    min_num_class = mesh_data['labels'].min()
    label_num = max_num_class - min_num_class + 1
    return label_num



class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text, end_char='\n'):
        if end_char != '\n':
            print(text, end='')
            self.f.write(text)
        else:
            print(text)
            self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    计算两点之间的欧几里得距离。
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


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
        DATA_DIR = os.path.join(data_dir, str(i) + '.txt')  # 数据文件
        LABEL_DIR = os.path.join(label_dir, str(i) + '.seg')  # 标签文件

        data = np.loadtxt(DATA_DIR)  # (N,628)
        label = np.loadtxt(LABEL_DIR)  # (N,)
        data_list.append(data)  # 累加所有数据，一行一个面片
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


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    xyz = xyz.contiguous()

    fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long()  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)
    # new_xyz = xyz[:]
    # new_points = points[:]

    idx = knn_point(nsample, xyz, new_xyz)
    #idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, -1)
    new_points = torch.cat([grouped_points_norm, new_points.view(B, S, 1, -1).repeat(1, 1, nsample, 1)], dim=-1)
    return new_xyz, new_points
