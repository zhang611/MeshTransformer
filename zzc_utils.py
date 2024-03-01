import numpy as np
import matplotlib.pyplot as plot


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text, end_char='\n'):
        if end_char != '\n':
            print(text, end='')
            self.f.write(text)
        else:
            print(text)
            self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def getPositionEncoding(seq_len, dim, n=10000):
    PE = np.zeros(shape=(seq_len, dim))
    for pos in range(seq_len):
        for i in range(int(dim / 2)):
            denominator = np.power(n, 2 * i / dim)
            PE[pos, 2 * i] = np.sin(pos / denominator)
            PE[pos, 2 * i + 1] = np.cos(pos / denominator)

    return PE

PE = getPositionEncoding(seq_len=300, dim=256, n=10000)
print(PE.shape)
print(PE)


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







# 加载npz查看
import numpy as np
path = r'datasets_processed/psb/psb_teddy/1_not_changed_.npz'
mesh_data = np.load(path, encoding='latin1', allow_pickle=True)

