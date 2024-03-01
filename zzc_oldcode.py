# import torch
# import meshio
# import dualmesh


# # 获得模型的点和面
# # model_path = r'E:\3DModelData\PSB\Teddy\1.off'
# model_path = r'1.off'
# teddy = load_mesh(model_path)  # TriangleMesh数据，13826个点，27648个面
# mesh_data = EasyDict({'vertices': np.asarray(teddy.vertices), 'faces': np.asarray(teddy.triangles)})
#
# # 获得模型的面标签
# # label_path = r'E:\3DModelData\PSB\Teddy\1.seg'
# label_path = r'1.seg'
# label = np.loadtxt(label_path)
# mesh_data['label'] = label
#
# # 获得模型的面特征
# feature_path = r'E:\3DModelData\PSB\Teddy\Features\1.txt'
# face_feature = np.loadtxt(feature_path)
# mesh_data['face_feature'] = face_feature  # (27648, 628)
#
# # mesh的对偶图，得到面的重心和多边形网格
# model_path = r'E:\3DModelData\PSB\Teddy\1.off'
# mesh = meshio.read(model_path)
# dual_mesh = dualmesh.get_dual(mesh, order=True)
# mesh_data['face_center'] = dual_mesh.points
# mesh_data['center_face'] = dual_mesh.cells_dict['polygon']  # 多边形，不是三边型
#
# # 重心的边
# mesh_data['center_edges'] = [set() for _ in range(mesh_data['face_center'].shape[0])]  # 27648个
# for i in range(mesh_data['center_face'].shape[0]):  # 13826次,遍历所有对偶面
#     for j in range(len(mesh_data['center_face'][i])):  # mesh_data['center_face'][0] = [1,0,5,2,4,3]
#         if j == (len(mesh_data['center_face'][i]) - 1):
#             mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(
#                 mesh_data['center_face'][i][j - 1])  # 前一个加进来
#             mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][0])  # 后一个
#         else:
#             mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(
#                 mesh_data['center_face'][i][j - 1])  # 前一个加进来
#             mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][j + 1])  # 后一个
#
# # 加载模型
# # mesh_data = np.load(npz_fn, encoding='latin1', allow_pickle=True)
#
# # matlab 算的一些东西
# dual_vertex = np.loadtxt('matlab/dual_vertex.txt', delimiter='\t')
# dual_vertex = dual_vertex.T  # (19092,3)
# geodesicDistances = np.loadtxt('matlab/geodesicDistances.txt', delimiter='\t')  # 这两个对称矩阵，不用转置，这个两个多g
# DihedralAngles = np.loadtxt('matlab/DihedralAngles.txt', delimiter='\t')
# SDF_Face = np.loadtxt('matlab/SDF_Face.txt', delimiter='\t')
# normalf = np.loadtxt('matlab/normalf.txt', delimiter='\t')
# normalf = normalf.T
#
# ring = np.loadtxt('ring.txt', delimiter='\t')  # 这个是不是就是周围的一环，就是备选游走点？
#
# # 放入mesh_data
# mesh_data['SDF_Face'] = SDF_Face
# mesh_data['dual_vertex'] = dual_vertex
# mesh_data['normalf'] = normalf
# mesh_data['geodesicDistances'] = geodesicDistances
# mesh_data['DihedralAngles'] = DihedralAngles
# mesh_data['ring'] = ring
#
# # 用matlab算，不用这个了
# # mesh的对偶图，得到面的重心和多边形网格
# mesh = meshio.read(model_path)
# dual_mesh = dualmesh.get_dual(mesh, order=True)
# mesh_data['face_center'] = dual_mesh.points
# mesh_data['center_face'] = dual_mesh.cells_dict['polygon']  # 多边形，不是三边型
#
# # 重心的边
# mesh_data['center_edges'] = [set() for _ in range(mesh_data['face_center'].shape[0])]  # 27648个
# for i in range(mesh_data['center_face'].shape[0]):  # 13826次,遍历所有对偶面
#     for j in range(len(mesh_data['center_face'][i])):  # mesh_data['center_face'][0] = [1,0,5,2,4,3]
#         if j == (len(mesh_data['center_face'][i]) - 1):
#             mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(
#                 mesh_data['center_face'][i][j - 1])  # 前一个加进来
#             mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][0])  # 后一个
#         else:
#             mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(
#                 mesh_data['center_face'][i][j - 1])  # 前一个加进来
#             mesh_data['center_edges'][mesh_data['center_face'][i][j]].add(mesh_data['center_face'][i][j + 1])  # 后一个
#
# # 保存为npz
# np.savez('matlab/data/test.npz', **mesh_data)



# x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)   # (2, 9) 输入语言
# x = torch.randint(1, 100, size=(4, 300, 3)).to(device)   # (4, 300, 3)
# trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)  # (2, 8)  标签，另一种语言
# trg = torch.randint(1, 6, size=(4, 300)).to(device)  # (4, 300)
# out = model(x, trg[:, :-1])  # (2, 7, 10)  最后一个位置通常作为预测的目标