%% 初始化
close all
clear
clc

%% 
% 先下采样到2k，都有代码，先不下采样了，先完成
% 下采样标签 
% 算特征描述符 ok
% 获得一环面矩阵 ok
% 面重心，面向量，作为特征 ok
% 测地距离
% 二面角矩阵 ok

%% 处理一个模型
% 导入模型
[vertex, face] = read_mesh('E:/3DModelData/PSB/Teddy/1.off');


% 计算对偶图
% adjacency_matrix 邻接矩阵，面与面之间的关系，dual_vertex 就是面的质心
[adjacency_matrix, dual_vertex] = compute_dual_graph(face,vertex);

% 测地距离，gpt给的代码，这个是欧氏距离近似，局部近似全局，近似了两次
geodesicDistances = calculateGeodesicDistances(dual_vertex, adjacency_matrix);



% 计算二面角，DihedralAngles矩阵保存结果
fileNameOfModel = 'E:/3DModelData/PSB/Teddy/1.off';
[DihedralAngles, Index] = szy_Compute_Dihedral_Angles(fileNameOfModel);




%% 暂时不用
% 计算三角形的法线，顶点法线和面法线
[normal,normalf] = compute_normal(vertex,face);

% SDF特征
SDF_Face = szy_Compute_SDF_AllFace_vf(vertex, face, 1);



%% 保存为txt
dlmwrite('SDF_Face.txt', SDF_Face, 'delimiter', '\t');
dlmwrite('dual_vertex.txt', dual_vertex, 'delimiter', '\t');
dlmwrite('normalf.txt', normalf, 'delimiter', '\t');


dlmwrite('geodesicDistances.txt', geodesicDistances, 'delimiter', '\t');

DihedralAngles = full(DihedralAngles);
dlmwrite('DihedralAngles.txt', DihedralAngles, 'delimiter', '\t');


%%
% 可视化walker

[vertex, face] = read_mesh('E:/3DModelData/PSB/Teddy/20.off');  % 读入模型
res = computeTriangleBarycentric(vertex, face);                 % 计算三角形重心
% 用重心的顺序找到最近的面的顺序索引
mesh_ids = szy_FindClosestFaceByPoint('E:/3DModelData/PSB/Teddy/20.off', res);   
seg_origin = load('./val_20.seg') + 1;                % 读入上色标签
mesh_label = seg_origin(mesh_ids);                    % 获得对应面的标签
figure();                                             % 画布
szy_PlotMesh_Discrete_vf(vertex, face, mesh_label);   % 画图



% 保存
folder = './look/20';  % 写入文件夹
dlmwrite([char(folder), '.seg'], (mesh_label - 1)');  %写入标签
% 写入obj和colorbar
szy_WriteMeshWithFaceColor_Discrete(vertex, face, [char(folder), '.seg.obj'], mesh_label-1);




%% 拓扑计算

% 获得面的连接关系列表ring
fring = compute_face_ring(face);  % 就是邻接关系矩阵
temp = cell2mat(fring);
ring = reshape(temp, 3, []);
ring = transpose(ring);
% 保存
save_path = ['matlab/data/PSB/teddy/',num2str(1),'_ring.txt' ];
dlmwrite(save_path, ring, 'delimiter', '\t');


% 批处理
for i = 1 : 1 : 20
    fn = ['E:/3DModelData/PSB/Bearing/',num2str(i),'.off'];
    % fn = ['E:/3DModelData/HumanBodySegmentation/train/',num2str(i),'.off'];
    [vertex, face] = read_mesh(fn);
    fring = compute_face_ring(face);  % 就是邻接关系矩阵
    temp = cell2mat(fring);
    ring = reshape(temp, 3, []);
    ring = transpose(ring);
    % 保存
    save_path = ['matlab/data/PSB/bearing/',num2str(i),'_ring.txt' ];
    % save_path = ['matlab/data/HumanBodySegmentation/train/',num2str(i),'_ring.txt' ];
    dlmwrite(save_path, ring, 'delimiter', '\t');
end



%% 测地距离
[vertex, face] = read_mesh('E:/3DModelData/PSB/Teddy/1.off');
[adjacency_matrix, dual_vertex] = compute_dual_graph(face,vertex);
% 测地距离，gpt给的代码，这个是欧氏距离近似，局部近似全局，近似了两次
geodesicDistances = calculateGeodesicDistances(dual_vertex, adjacency_matrix);

ring_geo = repmat(ring,1,1);
for i = 1 : size(ring,1)   % 一行
    for j = 1 : 3
        x = ring(i,j);    % 获得面片索引
        ring_geo(i,j) = geodesicDistances(i,x);
    end
end

% 批处理
for i = 1 : 1 : 20
    fn = ['E:/3DModelData/PSB/Bearing/',num2str(i),'.off'];
    % fn = ['E:/3DModelData/HumanBodySegmentation/train/',num2str(i),'.off'];
    [vertex, face] = read_mesh(fn);
    [adjacency_matrix, dual_vertex] = compute_dual_graph(face,vertex);
    geodesicDistances = calculateGeodesicDistances(dual_vertex, adjacency_matrix);

    fring = compute_face_ring(face);  % 就是邻接关系矩阵
    temp = cell2mat(fring);
    ring = reshape(temp, 3, []);
    ring = transpose(ring);

    ring_geo = repmat(ring,1,1);
    for j = 1 : size(ring,1)   % 一行
        for k = 1 : 3
            x = ring(j,k);    % 获得面片索引
            ring_geo(j,k) = geodesicDistances(j,x);
        end
    end
    % 保存
    save_path = ['matlab/data/PSB/bearing/',num2str(i),'_ring_geo.txt' ];
    % save_path = ['matlab/data/HumanBodySegmentation/train/',num2str(i),'_ring_geo.txt' ];
    dlmwrite(save_path, ring_geo, 'delimiter', '\t');
end






%% 二面角
fileNameOfModel = 'E:/3DModelData/PSB/Teddy/1.off';
[DihedralAngles, Index] = szy_Compute_Dihedral_Angles(fileNameOfModel);


[vertex, face] = read_mesh(fileNameOfModel);
[adjacency_matrix, dual_vertex] = compute_dual_graph(face,vertex);
DihedralAngles = full(DihedralAngles);

fring = compute_face_ring(face);  % 就是邻接关系矩阵
temp = cell2mat(fring);
ring = reshape(temp, 3, []);
ring = transpose(ring);
ring_dih = repmat(ring,1,1);
for i = 1 : size(ring,1)   % 一行
    for j = 1 : 3
        x = ring(i,j);    % 获得面片索引
        ring_dih(i,j) = DihedralAngles(i,x);
    end
end


    close all;
    clear;
    clc;
% 批处理
for i = 1 : 1 : 20
    % fn = ['E:/3DModelData/PSB/Bearing/',num2str(i),'.off'];
    fn = ['E:/3DModelData/HumanBodySegmentation/train/',num2str(i),'.off'];
    [vertex, face] = read_mesh(fn);
    [adjacency_matrix, dual_vertex] = compute_dual_graph(face,vertex);
    [DihedralAngles, Index] = szy_Compute_Dihedral_Angles(fn);
    DihedralAngles = full(DihedralAngles);

    fring = compute_face_ring(face);  % 就是邻接关系矩阵
    temp = cell2mat(fring);
    ring = reshape(temp, 3, []);
    ring = transpose(ring);
    ring_dih = repmat(ring,1,1);
    for j = 1 : size(ring,1)   % 一行
        for k = 1 : 3
            x = ring(j,k);    % 获得面片索引
            ring_dih(j,k) = DihedralAngles(j,x);
        end
    end
    % 保存 
    save_path = ['matlab/data/PSB/bearing/',num2str(i),'_ring_dih.txt' ];
    dlmwrite(save_path, ring_dih, 'delimiter', '\t');
end







%%
% 备用的一些代码


% 计算面个面周围的一环面，一环就三个，不用算对偶图，应该用这个就可以
fring = compute_face_ring(face);  % 有这个就可以游走了，不强求要对偶图


% Quadric Error Metrics模型简化，SIGGRAPH 97 二十多年了
nface = 2000;  % 下采样从19k到2k
[new_vertex,new_face] = perform_mesh_simplification(vertex,face,nface,'20');

% 计算三角形重心
res = computeTriangleBarycentric(vertex, face);  %要补充这个函数，看看用dual_vertex是否可替代



% 两两顶点之间的测地距离矩阵
DistanceMatrix = szy_GetGeodesicDistanceMatrix_dijkstra_vf(vertex, face);



% 计算628维的特征
SDF_Face = szy_Compute_SDF_AllFace_vf(vertex, face, 1);
GC_Face = szy_Compute_GaussianCurvatureAllFace_vf(vertex, face);
%szy_ComputeMeshSegmentationFeatures(modelFileName, featureFileName);
% 2010年希腊人628维特征向量
% 628维特征向量按顺序分别为the curvature (64维), PCA (48维), geodesic shape contexts (270维), 
% geodesic distance features (15维), shape diameter (72维), distance from medial surface (24维), 
% spin images (100维), shape context class probabilities (35维) 
% 这个测地距离特征为什么就15维，能不能用这个作为位置编码


%加载标签
segName = ['E:/3DModelData/PSB/Teddy/', int2str(20), '.seg'];
seg_origin = load(segName) + 1;

% 怎么映射过来？自己映射？
% 原来19092个面，QEM之后2k个面



























