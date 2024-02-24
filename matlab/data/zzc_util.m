% 一些工具
%% 初始化
close all
clear
clc
%% 算一个模型的准确率
[vertex, face] = read_mesh('E:/3DModelData/PSB/Teddy/20.off');    % 读取模型的顶点和面
Areas = szy_GetAreaOfFaces_vf(vertex, face);                      % 计算模型面的面积
seg_origin = load('E:/3DModelData/PSB/Teddy/20.seg') + 1;         % 读取GT标签
test_seg = load('./PLable/teddy1/19.seg') + 1;                    % 读取我的预测标签
successRatio = Get_SuccessRatio(test_seg,seg_origin,Areas');      % 参数：预测标签，gt，面积

%% 算所有结果的准确率
successRatio = {};
for i = 1:20
    offName = ['E:/3DModelData/PSB/Teddy/', int2str(i), '.off'];
    segName = ['E:/3DModelData/PSB/Teddy/', int2str(i), '.seg'];
    [vertex, face] = read_mesh(offName);
    Areas = szy_GetAreaOfFaces_vf(vertex, face);
    seg_origin = load(segName) + 1;
    test_seg_name = ['./PLable/teddy2/', int2str(i-1),'.seg'];
    test_seg = load(test_seg_name) + 1;
    successRatio{i} = Get_SuccessRatio(test_seg,seg_origin,Areas');
end


%% 可视化并保存
[vertex, face] = read_mesh('E:/3DModelData/PSB/Teddy/19.off');  % 读入模型
res = computeTriangleBarycentric(vertex, face);                 % 计算三角形重心
% 用重心的顺序找到最近的面的顺序索引
mesh_ids = szy_FindClosestFaceByPoint('E:/3DModelData/PSB/Teddy/19.off', res);   
seg_origin = load('E:/3DModelData/PSB/Teddy/19.seg') + 1;      % 读入上色标签
mesh_label = seg_origin(mesh_ids);                    % 获得对应面的标签
figure();                                             % 画布
szy_PlotMesh_Discrete_vf(vertex, face, mesh_label);   % 画图

% 保存
folder = './zzcTEST./19';  % 写入文件夹
dlmwrite([char(folder), '.seg'], (mesh_label - 1)');  %写入标签
% 写入obj和colorbar
szy_WriteMeshWithFaceColor_Discrete(vertex, face, [char(folder), '.seg.obj'], mesh_label-1);

%% 图割+算准确率+可视化
% 图割
[vertex, face] = read_mesh('E:/3DModelData/PSB/Teddy/19.off');
pred = load('./PLable/teddy2/18.prob');
segResult = szy_GraphCut_vf(vertex,face,0.1,pred',false);
% 算准确率
Areas = szy_GetAreaOfFaces_vf(vertex, face);                % 算面积
seg_origin = load('E:/3DModelData/PSB/Teddy/19.seg') + 1;   % 读取GT标签
successRatio = Get_SuccessRatio(segResult,seg_origin,Areas'); 
% 可视化
res = computeTriangleBarycentric(vertex, face);
mesh_ids = szy_FindClosestFaceByPoint('E:/3DModelData/PSB/Teddy/19.off', res);
mesh_label = segResult(mesh_ids);
figure();                                             
szy_PlotMesh_Discrete_vf(vertex, face, mesh_label);
% 保存
folder = './result./teddyGC./20';  % 写入文件夹
dlmwrite([char(folder), '.seg'], (mesh_label - 1)');  %写入标签
% 写入obj和colorbar
szy_WriteMeshWithFaceColor_Discrete(vertex, face, [char(folder), '.seg.obj'], mesh_label-1);

