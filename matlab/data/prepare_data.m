% 准备数据
%%
%准备工作
close all
clear
clc

%PSB中所有的模型名
categoryName = {'Vase', 'Teddy', 'Table', 'Plier', 'Octopus', 'Mech',...
    'Human', 'Hand', 'Glasses', 'Fourleg', 'Fish', 'Cup', 'Chair', 'Bust',...
    'Bird', 'Bearing', 'Armadillo', 'Ant', 'Airplane'};
% 对应模型的类别数
categoryClassNum = {1,5,2,1,2.5,1,1,6,3,1,1,1,4,1,1,1,1,5,1};

category_now = 2;   %目前是哪个模型，teddy
classNum = categoryClassNum{category_now}; % 当前模型的部件数，5


%%
%基本设置
%Teddy模型1-20号，1-4做有标签,4-16无标签，16-20测试集

% 数据集根目录
UserPath = 'E:/3DModelData/PSB/';  
% 待处理的模型'E:/3DModelData/PSB/Teddy/*.off'
dataFileDir = [UserPath,categoryName{category_now},'/*.off'];
% Teddy模型的根目录 'E:/3DModelData/PSB/Teddy/'
indir = [UserPath,categoryName{category_now},'/'];
% 特征的根目录 'E:/3DModelData/PSB/Teddy/Features/'
dataDir = [UserPath,categoryName{category_now},'/Features/'];
% 得到所有模型名字
model = dir(dataFileDir);  % struct存所有模型
model = {model.name};      % 只要name变成元胞数组
model = sort_nat(model);   % 排序



%%
%处理读入的20个模型，分别计算他们的特征向量、面片面积，以及读入他们的groundth-truth
start = 1;                    % 下标
AllFaceFeatures = {};         % 所有面片的特征向量，神经网络的输入
seg = {};                     % 所有面片的标签，ground-truth
Areas = {};                   % 所有面片的面积，算准确率使用

for i = start:(start+19)
    Name = [indir, model{1, i}];       % 要处理模型的名字
    disp(['Processing ', Name, ' ...']);
    [vertex, face] = read_mesh(Name);  % 读入一个模型
    
    % 计算每个面片的面积
    Areas{i} = szy_GetAreaOfFaces_vf(vertex, face);  

    % 拼接希腊人的628维特征向量、WKS和SIHKS特征
    Feature = load([dataDir, erase(model{1, i}, '.off'), '.txt']);
    WKS_face = load([dataDir, erase(model{1, i}, '.off'), '.WKS.txt']);
    SIHKS_face = load([dataDir, erase(model{1, i}, '.off'), '.SIHKS.txt']);
    %此时每个面片的特征向量一共747维
    AllFaceFeatures{i} = [ Feature WKS_face SIHKS_face ];

    % 读入每个模型的ground-truth,并+1处理
    seg{i} = load([indir, int2str(i), '.seg']) + 1;
end
clear Feature;
clear WKS_face;
clear SIHKS_face;
%%
% 准备mat文件
% AllFaceFeatures 是二十个模型的特征
% seg 是二十个模型的标签
train_data = [];
train_lable = [];
test_data = [];
test_lable = [];

for i = 1:16
    train_data = [train_data;AllFaceFeatures{i}];
    train_lable = [train_lable;seg{i}];   
end

for i = 16:20
    test_data = [test_data;AllFaceFeatures{i}];
    test_lable = [test_lable;seg{i}];   
end


save PSBdata/Teddy  train_data


%%

tr_dat = [];
tr_lb = [];

%构造tr_dat  全监督训练集
for i = start:(start+11)
    if (i == 1)||(i == 2)||(i == 3)||(i == 4)
        tr_dat = [tr_dat;AllFaceFeatures{i}];
        tr_lb = [tr_lb;seg{i}];
    end
end
tr_dat = tr_dat';
% mlp的架构[747,360,120,5]
structure = [size(tr_dat, 1) 360 120 numel(unique(tr_lb))];
X_Train = tr_dat';
Y_Train = categorical(tr_lb);

save data X_Train Y_Train tr_lb

%%
% 准备测试集，最后四个模型
test_dat = [];
test_lb = [];

%构造tr_dat  全监督训练集
for i = 17:20
    test_dat = [test_dat;AllFaceFeatures{i}];
    test_lb = [test_lb;seg{i}];
end

X_Test = test_dat;
Y_Test = test_lb;


save test X_Test Y_Test



%%
%设置MLP的layers，% MLP网络，复杂的用pytorch实现
layers = [featureInputLayer(structure(1))];
for i = 2:(size(structure, 2)-1)
    layers = [layers fullyConnectedLayer(structure(i)) reluLayer()];
end
layers = [layers fullyConnectedLayer(structure(size(structure, 2)))];
layers = [layers softmaxLayer() classificationLayer()];

        
%设置训练选项
training_options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 1024, ...
    'L2Regularization', 0.0001, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

feature_net = [];
feature_net.Layers = layers;
[feature_net, info] = trainNetwork(X_Train, Y_Train, feature_net.Layers, training_options);



%%




%% 可视化
% 读入顶点和面
[vertex, face] = read_mesh('./Teddy/17.obj');

% 计算三角形重心
res = computeTriangleBarycentric(vertex, face);
            
% 用重心的顺序找到最近的面的顺序索引
mesh_ids = szy_FindClosestFaceByPoint('./Teddy/17.obj', res);

% 读入标签,matlab从1开始，用不同标签改这里就行
% seg_origin = load('./Teddy_Labels/17.seg') + 1;
seg_origin = load('./result/16.seg') + 1;
% 获得对应面的标签
mesh_label = seg_origin(mesh_ids);

% 画图
figure();
szy_PlotMesh_Discrete_vf(vertex, face, mesh_label);

% 写入文件夹
folder = './result./demo';
% 写入标签
dlmwrite([char(folder), '.seg'], (mesh_label - 1)');  %写入标签
% 写入obj和colorbar
szy_WriteMeshWithFaceColor_Discrete(vertex, face, [char(folder), '.seg.obj'], mesh_label-1);

%% 计算准确率
Areas = szy_GetAreaOfFaces_vf(vertex, face);
% test = load('./Teddy_Labels/1_test.seg') + 1;
test = load('./result/16.seg') + 1;
successRatio = Get_SuccessRatio(test,seg_origin,Areas');  % 参数：预测标签，gt，面积

%% 图割
% 顶点，面表示这个模型，0.03，false默认不管他，预测矩阵就是没max的标签
test = load('./result/16.prob');
test = 1.0 / test;
Label = szy_GraphCut_vf(vertex,face,0.03,test,false);









