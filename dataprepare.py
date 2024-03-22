import glob
import os
from tqdm import tqdm
import numpy as np
import open3d
import trimesh
from easydict import EasyDict
from utils import get_label_num


def prepare_edges_and_kdtree(mesh):
    """obj文件里只有顶点和面，通过顶点和面得到边"""
    vertices = mesh['vertices']  # (13826, 3)
    faces = mesh['faces']  # (27648, 3)
    mesh['edges'] = [set() for _ in range(vertices.shape[0])]  # 初始化顶点数个空集合
    for i in range(faces.shape[0]):  # 一个面一个面的看
        for v in faces[i]:  # 一个面三个点  [41, 40, 33]
            mesh['edges'][v] |= set(faces[i])  # 把401 40 33 放到 41 索引下，其他同理
    for i in range(vertices.shape[0]):
        if i in mesh['edges'][i]:
            mesh['edges'][i].remove(i)
        mesh['edges'][i] = list(mesh['edges'][i])
    max_vertex_degree = np.max([len(e) for e in mesh['edges']])
    for i in range(vertices.shape[0]):
        if len(mesh['edges'][i]) < max_vertex_degree:
            mesh['edges'][i] += [-1] * (max_vertex_degree - len(mesh['edges'][i]))
    mesh['edges'] = np.array(mesh['edges'], dtype=np.int32)

    mesh['kdtree_query'] = []
    t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    n_nbrs = min(10, vertices.shape[0] - 2)
    for n in range(vertices.shape[0]):
        d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
        i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
        if len(i_nbrs_cleared) > n_nbrs - 1:
            i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
        mesh['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
    mesh['kdtree_query'] = np.array(mesh['kdtree_query'])
    assert mesh['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(
        mesh['kdtree_query'].shape[1])


def get_center(mesh):
    """获得面的重心坐标"""
    vertices = mesh['vertices']  # (13826, 3)
    faces = mesh['faces']  # (27648, 3)
    position = []
    for i in range(len(faces)):
        a, b, c = faces[i]  # 41 40 33
        center = vertices[0].copy()  # 初始化
        # TODO 就是a不用a-1得到的和matlab算的一样，再验证一下这个,也可能是matlab有问题
        center[0] = (vertices[a][0] + vertices[b][0] + vertices[c][0]) / 3  # x 坐标
        center[1] = (vertices[a][1] + vertices[b][1] + vertices[c][1]) / 3  # y
        center[2] = (vertices[a][2] + vertices[b][2] + vertices[c][2]) / 3  # z
        position.append(center)
    mesh['center'] = np.array(position)


def add_fields_and_dump_model(mesh_data, needed, out_fn, dump_model=True):
    m = {}  # 存放模型的字典
    for k, v in mesh_data.items():
        if k in needed:
            m[k] = v  # 先把前面搞好的6个写进去，顶点,面,标签,ring,dih,geo

    for field in needed:  # 初始化其他键的值
        if field not in m.keys():
            if field == 'edges':  # 通过顶点和面得到边
                prepare_edges_and_kdtree(m)  # 边的信息有两个，edges和kdtree_query
            if field == 'center':
                get_center(m)

    if dump_model:
        np.savez(out_fn, **m)  # 一个obj变成一个npz，神经网络就用这个

    return m


def load_mesh(model_fn):
    """加载mesh"""
    mesh_ = trimesh.load_mesh(model_fn, process=False)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
    mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)
    return mesh


def prepare_directory(dataset_name, model_name, pathname_expansion=None, p_out=None, f_idx=0):  # f_idx保证标签的索引0开始
    key_needed = ['vertices', 'faces', 'labels', 'center',
                  'ring', 'geodesic', 'dihedral',
                  'edges', 'kdtree']  # mesh字典的键，9个

    if not os.path.isdir(p_out):
        os.makedirs(p_out)

    filenames = glob.glob(pathname_expansion)  # 模型 off 路径列表
    filenames.sort()  # 先排序,字符串排序，1后面是10,排不排序也无所谓了
    for file in tqdm(filenames, disable=False):  # disable 是否输出详细信息
        model_num = os.path.split(file)[1].split('.')[0]  # 几号模型
        out_fn = os.path.join(p_out, model_num)  # 'datasets_processed/psb/teddy/1'

        # 加载顶点和面
        mesh = load_mesh(file)
        mesh_data = EasyDict({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.triangles)})

        # 加载 ring 相关
        ring_root = "matlab/data"
        ring_path = os.path.join(ring_root, dataset_name, model_name, model_num + "_ring.txt")
        ring = np.loadtxt(ring_path, delimiter='\t')
        ring = np.int64(ring - 1)  # 统一面片索引从0开始，并且是int64格式
        mesh_data['ring'] = ring

        # 加载 测地距离，都是matlab算好的  TODO：有没有必要放大一些，现在是0.001附近
        geo_path = os.path.join(ring_root, dataset_name, model_name, model_num + "_ring_geo.txt")
        ring_geo = np.loadtxt(geo_path, delimiter='\t')
        mesh_data['geodesic'] = ring_geo

        # 加载 二面角  TODO 是不是要放大，这个角度到底是什么角度
        dih_path = os.path.join(ring_root, dataset_name, model_name, model_num + "_ring_dih.txt")
        ring_dih = np.loadtxt(dih_path, delimiter='\t')
        mesh_data['dihedral'] = ring_dih

        # 加载标签
        label_path = file.split(".")[0] + ".seg"  # 后缀改成seg就是标签
        labels = np.loadtxt(label_path)
        labels = np.int64(labels - f_idx)
        mesh_data['labels'] = labels

        str_to_add = f'_{model_name}_original'
        out_fc_full = out_fn + str_to_add  # 最后的npz输出路径
        add_fields_and_dump_model(mesh_data, key_needed, out_fc_full)


def prepare_psb(dataset, subfolder):
    datasets_root = "datasets_raw/psb"  # 数据集所在的根目录
    path_in = os.path.join(datasets_root, subfolder)  # teddy模型所在目录
    # path_in = "E:\\3DModelData/PSB/" + sub            # 模型所在位置  r"E:\3DModelData\PSB\teddy"
    out_root = "datasets_processed"  # 输出根目录
    p_out = os.path.join(out_root, dataset, subfolder)  # 输出目录
    pathname_expansion = os.path.join(path_in, "*.off")
    prepare_directory(dataset, subfolder, pathname_expansion, p_out, f_idx=0)


def prepare_hbs(dataset, subfolder):
    sub = subfolder[22:]
    path_in = "E:\\3DModelData/HumanBodySegmentation/" + sub  # 模型所在位置
    p_out = 'datasets_processed/' + dataset + '/' + subfolder  # 'datasets_processed/HumanBodySegmentation/test'
    prepare_directory(dataset, subfolder, pathname_expansion=path_in + '/' + '*.off', p_out=p_out)


def prepare_one_dataset(dataset_name):
    if dataset_name == 'psb':
        # model_list = ["airplane", "ant", "armadillo", "bearing", "bird", "bust", "chair",
        #               "cup", "fish", "fourLeg", "glasses", "hand", "human", "mech",
        #               "octopus", "plier", "table", "teddy", "vase"]

        model_list = ["teddy"]
        for model in model_list:
            prepare_psb(dataset_name, model)

    if dataset_name == 'HumanBodySegmentation':
        prepare_hbs(dataset_name, 'test')
        prepare_hbs(dataset_name, 'HumanBodySegmentation_train')
        # prepare(dataset_name, 'train')

    if dataset_name == 'shrec11':
        print('To do later')

    # Semantic Segmentations
    if dataset_name == 'human_seg':
        print('To do later')

    if dataset_name == 'coseg':
        print('To do later')
        # prepare_seg_from_meshcnn('coseg', 'coseg_aliens')
        # prepare_seg_from_meshcnn('coseg', 'coseg_chairs')
        # prepare_seg_from_meshcnn('coseg', 'coseg_vases')


"""
psb 的标签就是从0开始的，human_body的标签从1开始的，统一一下
"""
if __name__ == '__main__':
    np.random.seed(1)
    dataset_name = 'psb'  # 'HumanBodySegmentation'
    prepare_one_dataset(dataset_name)

    path = "datasets_processed/psb/teddy/1_teddy_original.npz"
    print("类别数为：", get_label_num(path))
