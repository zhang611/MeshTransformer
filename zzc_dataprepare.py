import glob
import os
import sys

import numpy as np
import open3d
import trimesh
from easydict import EasyDict

import torch
import meshio
import dualmesh
"""
mesh模型的所有属性都保存在easydict里面
整体都是基于面的，标签只需要面标签

1.预处理，读入模型，并且加载或者计算出所有我需要的东西，一个模型保存一个npz
我需要的
顶点:vertices
面:faces
面特征:face_feature
面标签:label
面的重心:face_center
重心的面:center_face
重心的边:center_edges

"""


def load_mesh(model_fn):
    """加载mesh"""
    mesh_ = trimesh.load_mesh(model_fn, process=False)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
    mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)
    return mesh


if __name__ == '__main__':
    # 获得模型的点和面
    model_path = r'E:\3DModelData\PSB\Teddy\1.off'
    teddy = load_mesh(model_path)     # TriangleMesh数据，13826个点，27648个面
    mesh_data = EasyDict({'vertices': np.asarray(teddy.vertices), 'faces': np.asarray(teddy.triangles)})

    # 获得模型的面标签
    label_path = r'E:\3DModelData\PSB\Teddy\1.seg'
    label = np.loadtxt(label_path)
    mesh_data['label'] = label

    # 获得模型的面特征
    feature_path = r'E:\3DModelData\PSB\Teddy\Features\1.txt'
    face_feature = np.loadtxt(feature_path)
    mesh_data['face_feature'] = face_feature   # (27648, 628)

    # mesh的对偶图，得到面的重心和多边形网格
    model_path = r'E:\3DModelData\PSB\Teddy\1.off'
    mesh = meshio.read(model_path)
    dual_mesh = dualmesh.get_dual(mesh, order=True)
    mesh_data['face_center'] = dual_mesh.points
    mesh_data['center_face'] = dual_mesh.cells_dict['polygon']    # 多边形，不是三边型

    # 重心的边
    mesh_data['center_edges'] = [set() for _ in range(mesh_data['face_center'].shape[0])]  # 27648个
    for i in range(mesh_data['center_face'].shape[0]):  # 13826次,遍历所有对偶面
        for j in range(len(mesh_data['center_face'][i])):       # mesh_data['center_face'][0] = [1,0,5,2,4,3]
            if j == (len(mesh_data['center_face'][i]) - 1):
                mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][j-1])  # 前一个加进来
                mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][0])  # 后一个
            else:
                mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][j-1])  # 前一个加进来
                mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][j+1])  # 后一个

    # 保存为npz
    np.savez('PSBdata/teddy1.npz', **mesh_data)

    # 加载模型
    # mesh_data = np.load(npz_fn, encoding='latin1', allow_pickle=True)




    # 获得模型的点和面
    model_path = r'E:\3DModelData\PSB\Teddy\20.off'
    teddy = load_mesh(model_path)     # TriangleMesh数据，13826个点，27648个面
    mesh_data = EasyDict({'vertices': np.asarray(teddy.vertices), 'faces': np.asarray(teddy.triangles)})

    # 获得模型的面标签
    label_path = r'E:\3DModelData\PSB\Teddy\20.seg'
    labels = np.loadtxt(label_path)
    mesh_data['labels'] = labels

    # 获得模型的面特征
    feature_path = r'E:\3DModelData\PSB\Teddy\Features\1.txt'
    face_feature = np.loadtxt(feature_path)
    mesh_data['face_feature'] = face_feature   # (27648, 628)

    # matlab 算的一些东西
    SDF_Face = np.loadtxt('matlab/SDF_Face.txt', delimiter='\t')
    dual_vertex = np.loadtxt('matlab/dual_vertex.txt', delimiter='\t')
    dual_vertex = dual_vertex.T  # (19092,3)
    normalf = np.loadtxt('matlab/normalf.txt', delimiter='\t')
    normalf = normalf.T
    geodesicDistances = np.loadtxt('matlab/geodesicDistances.txt', delimiter='\t')  # 这两个对称矩阵，不用转置，这个两个多g
    DihedralAngles = np.loadtxt('matlab/DihedralAngles.txt', delimiter='\t')

    ring = np.loadtxt('ring.txt', delimiter='\t')


    # 放入mesh_data
    mesh_data['SDF_Face'] = SDF_Face
    mesh_data['dual_vertex'] = dual_vertex
    mesh_data['normalf'] = normalf
    mesh_data['geodesicDistances'] = geodesicDistances
    mesh_data['DihedralAngles'] = DihedralAngles
    mesh_data['ring'] = ring



    # 用matlab算，不用这个了
    # mesh的对偶图，得到面的重心和多边形网格
    mesh = meshio.read(model_path)
    dual_mesh = dualmesh.get_dual(mesh, order=True)
    mesh_data['face_center'] = dual_mesh.points
    mesh_data['center_face'] = dual_mesh.cells_dict['polygon']    # 多边形，不是三边型

    # 重心的边
    mesh_data['center_edges'] = [set() for _ in range(mesh_data['face_center'].shape[0])]  # 27648个
    for i in range(mesh_data['center_face'].shape[0]):  # 13826次,遍历所有对偶面
        for j in range(len(mesh_data['center_face'][i])):       # mesh_data['center_face'][0] = [1,0,5,2,4,3]
            if j == (len(mesh_data['center_face'][i]) - 1):
                mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][j-1])  # 前一个加进来
                mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][0])  # 后一个
            else:
                mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][j-1])  # 前一个加进来
                mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][j+1])  # 后一个

    # 保存为npz
    np.savez('matlab/data/test.npz', **mesh_data)
    np.savez('matlab/data/PSB/teddy/1.npz', **mesh_data)

    # 还可以把不需要的键去掉













