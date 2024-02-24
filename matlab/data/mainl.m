%半监督scribble分割
%需要手调的超参数如下：
%E_face中高斯平滑的sigma，根据W_GC的结果选择训练集序号,最终结果进行GC中的lamda
%网络训练过程中的epoch数（只要能差不多收敛就可以）
%%
%准备工作
close all
clear
clc
%准备PSB中所有的模型名
categoryName = {'Vase', 'Teddy', 'Table', 'Plier', 'Octopus', 'Mech',...
    'Human', 'Hand', 'Glasses', 'Fourleg', 'Fish', 'Cup', 'Chair', 'Bust',...
    'Bird', 'Bearing', 'Armadillo', 'Ant', 'Airplane'};
categoryClassNum = {1,5,2,1,2.5,1,1,6,3,1,1,1,4,1,1,1,1,5,1};
sigma = {0,2,1,0,3,0,0,2,5,0,0,0,0,0,0,0,0,0,0};
%目前是第几类
category_now = 2;
classNum = categoryClassNum{category_now};

%%
%基本设置

%选择Teddy模型1-20号，1-4做F，5-12做W，13-20做测试集
%UserPath = 'D:/WT/PSB/';
UserPath = 'E:/3DModelData/PSB/';
dataFileDir = [UserPath,categoryName{category_now},'/*.off'];
indir = [UserPath,categoryName{category_now},'/'];
dataDir = [UserPath,categoryName{category_now},'/Features/'];
model = dir(dataFileDir);
model = {model.name};

%%有时候读入的文件名是乱序的，通过这个函数进行排序
model = sort_nat(model);
%%
%处理读入的20个模型，分别计算他们的特征向量、面片面积，以及读入他们的groundth-truth
start = 1;                    %下标
AllFaceFeatures = {};         %所有面片的特征向量
seg = {};                     %所有面片的ground-truth
Areas = {};                   %所有面片的面积
FaceFeature = {};
for i = start:(start+19)
    Name = [indir, model{1, i}];
    disp(['Processing ', Name, ' ...']);
    [vertex, face] = read_mesh(Name);
    
    %计算每个面片的面积并读进来
    Areas{i} = szy_GetAreaOfFaces_vf(vertex, face);  
%     AGD{i} = szy_Compute_AGDAllVertex_vf(vertex,face);

    %采用希腊人生成的628维特征向量、WKS和SIHKS特征拼接起来
    Feature = load([dataDir, erase(model{1, i}, '.off'), '.txt']);
    WKS_face = load([dataDir, erase(model{1, i}, '.off'), '.WKS.txt']);
    SIHKS_face = load([dataDir, erase(model{1, i}, '.off'), '.SIHKS.txt']);

    %此时每个面片的特征向量一共747维
    AllFaceFeatures{i} = [ Feature WKS_face SIHKS_face ];

    %读入每个模型的ground-truth,先做+1处理，方便后续使用
    seg{i} = load([indir, int2str(i), '.seg']) + 1;
    angles = full(szy_Compute_Dihedral_Angles(Name));
end
clear Feature;
clear WKS_face;
clear SIHKS_face;
%% 保存工作区变量，下次从这执行就行
save angles;
%%
%读入12个模型的人工scribble文件
scribble_W = {};
%读取手工标注的scribble文件：1-12号模型
%scribble_W = load(['G:/zhangzhichao/matlab/code_paper_segmentation/Scribble_Teddy/1_scribble.txt']);


% 循环读取每个txt文件
for i = 1:12
    % 读取txt文件
    %filename = sprintf('file%d.txt', i);
    filename = sprintf('G:/zhangzhichao/matlab/code_paper_segmentation/Scribble_Teddy/%d_scribble.txt', i);
    temp_data = importdata(filename);
    
    % 将数据添加到矩阵中
     scribble_W{i} = temp_data(:);
end
clear temp_data;

Read_Scribble_txt;
%%
%特征提取网络尝试
tr_dat = [];
tr_lb = [];

%构造tr_dat
for i = start:(start+11)
    if (i == 1)||(i == 2)||(i == 3)||(i == 4)  % 1234有标签
        tr_dat = [tr_dat;AllFaceFeatures{i}];
        tr_lb = [tr_lb;seg{i}];
    end
end
tr_dat = tr_dat';
structure = [size(tr_dat, 1) 360 120 numel(unique(tr_lb))]; % 网络结构
X_Train = tr_dat';
Y_Train = categorical(tr_lb);

%设置MLP的layers
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
%构建合适的FaceFeature
FaceFeature = {};
for i = start:start+11
    modelFileName = [indir, model{1, i}];          %读入对应模型
    disp(['Processing ',modelFileName , ' ...']);
    [vertex, face] = read_mesh(modelFileName);  
    WKS_face = load([dataDir, erase(model{1, i}, '.off'), '.WKS.txt']);
    SIHKS_face = load([dataDir, erase(model{1, i}, '.off'), '.SIHKS.txt']);
    %用于欧氏距离计算的特征向量
    FaceFeature{i} = [WKS_face SIHKS_face];
end
%%
save zzc
%%
%调整模型的sigma参数
sumRatio_1 = 0;  % 正常的
sumRatio_2 = 0;  % 图割之后的
for i = (start):(start+11)
        modelFileName = [indir, model{1, i}];          %读入对应模型
        [vertex, face] = read_mesh(modelFileName);  
        %读入对应的手工scribble结果，并构造对应的scribble矩阵
        scribble = scribble_W{i};
        ScribbleMatrix = zeros(size(face,2),numel(unique(scribble(scribble~=0))));%facenum * classnum的矩阵
        for k = 1:numel(unique(scribble(scribble~=0)))
            index = find(scribble == k);
            ScribbleMatrix(index, k) = 1;
        end
        
        [E_face_to_scribble, E_face] = Get_face_to_scribble(face,scribble,test_feature{i},1);
        [~,test] = max(E_face,[],2);
%         temp1 = find(scribble~=0);
%         temp2 = temp1(test);
%         temp3 = scribble(temp2);
        
        successRatio = Get_SuccessRatio(test,seg{i},Areas{i}');
        sumRatio_1 = sumRatio_1 + successRatio;
        disp([':success Ratio  of E_face of ',int2str(i),' is ',num2str(successRatio)]);
        figure;
        szy_PlotMesh_Discrete_vf(vertex,face,test);
        
        E_face = 1./E_face;
        E_face = ReplaceInf(E_face);
        E_scribble = Get_E_Scribble(ScribbleMatrix);
        Prob_Matrix = E_scribble + E_face;
        Prob_Matrix = 1./Prob_Matrix;
        Prob_Matrix = ReplaceInf(Prob_Matrix);
        Label = szy_GraphCut_vf(vertex,face,0.03,Prob_Matrix',false );
        successRatio = Get_SuccessRatio(Label,seg{i},Areas{i}');
        sumRatio_2 = sumRatio_2 + successRatio;
        disp([':success Ratio  of E_face of ',int2str(i),' is ',num2str(successRatio)]);
        figure;
        szy_PlotMesh_Discrete_vf(vertex,face,Label);
end
sumRatio_1 = sumRatio_1 ./ 12;
sumRatio_2 = sumRatio_2 ./12;
disp(['Average ratio of is ', num2str(sumRatio_1)]);
disp(['Average ratio of is ', num2str(sumRatio_2)]);
%%
%生成W模型的face_level标签
times = 1;
Label_now = {};
time_max = 5;
net = [];
Prob ={};
while true
    for i = (start):(start+11)
        modelFileName = [indir, model{1, i}];          %读入对应模型
        [vertex, face] = read_mesh(modelFileName);
        clear E_GD_face;
        clear E_pairwise;
        clear AdjacentMatrix;
        clear E_net;
        clear E_scribble;
        clear pred_y;
        clear Prob_Matrix;
        clear tr_dat;
        clear tr_lb;
        clear X_Train;
        clear Y_Train;
        
        
        %读入对应的手工scribble结果，并构造对应的scribble矩阵
        scribble = scribble_W{i};
        ScribbleMatrix = zeros(size(face,2),numel(unique(scribble(scribble~=0))));%facenum * classnum的矩阵
        for k = 1:numel(unique(scribble(scribble~=0)))
            index = find(scribble == k);
            ScribbleMatrix(index, k) = 1;
        end
        AdjacentMatrix = Get_AdjacentMatrix(face);
        
        if(isempty(net))
            [~,E_face] = Get_face_to_scribble(face,scribble,FaceFeature{i},5);
            [~,test] = min(E_face,[],2);
            successRatio = Get_SuccessRatio(test,seg{i},Areas{i}');
            disp(['Times ',int2str(times),':success Ratio  of E_face of ',int2str(i),' is ',num2str(successRatio)]);
            figure;
            szy_PlotMesh_Discrete_vf(vertex,face,test);
            
%             %处理E_face
%             E_face = 1./E_face;
%             E_face = ReplaceInf(E_face);
        end
        %scribble对应的能量项，facenum*classnum的矩阵，矩阵值越小越对应真实标签
        E_scribble = Get_E_Scribble(ScribbleMatrix);
        E_scribble = double(E_scribble);
        clear ScribbleMatrix;
        
        if(~isempty(net))
            tt_dat = AllFaceFeatures{i};
            pred_y = predict(net, tt_dat);
            E_net = pred_y;
            [~,Init_Label] = max(E_net,[],2);
            successRatio = Get_SuccessRatio(Init_Label,seg{i},Areas{i}');
            disp(['Times ',int2str(times),':success Ratio  of Pseudo label of ',int2str(i),' is ',num2str(successRatio)]);
            E_net = -log(pred_y);
            E_net = ReplaceInf(E_net);
        else
            E_net = 0;
        end
        
        if(~isempty(net))
            Prob_Matrix = E_scribble + E_net;
        else
            Prob_Matrix = E_scribble + E_face;
        end
        [~,Init_Label] = min(Prob_Matrix,[],2);
        Prob{i} = Prob_Matrix;
        successRatio = Get_SuccessRatio(Init_Label,seg{i},Areas{i}');
        disp(['Times ',int2str(times),':success Ratio of Pseudo label with E_scribble + E_face of ',int2str(i),' is ',num2str(successRatio)]);
        
        Prob_Matrix = 1./Prob_Matrix;
        Prob_Matrix = ReplaceInf(Prob_Matrix);
        
        clear E_face;
        clear pred_y;
        clear E_scribble;
        clear E_net;
        %欧式距离对应的二元项
%         X = full(szy_Compute_Dihedral_Angles(modelFileName));
%         X = -X;
        
%         E_pairwise = Get_E_pairwise(AdjacentMatrix,AllFaceFeatures{i}',2); 
        %加入欧氏距离这个二元项后
%         Label_now{i} = szy_AlphaExpansionGraphCut(X,Prob_Matrix',Init_Label)';
        Label_now{i} = szy_GraphCut_vf(vertex,face,0.05,Prob_Matrix',false );
        successRatio = Get_SuccessRatio(Label_now{i},seg{i},Areas{i}');
        disp(['Times ',int2str(times),':success Ratio of Pseudo label with E_scribble + E_net + E_face of ',int2str(i),' is ',num2str(successRatio)]);
        figure;
        szy_PlotMesh_Discrete_vf(vertex,face,Label_now{i});
    end
   
    clear E_pairwise;

    tr_dat = [];
    tr_lb = [];
    for i = (start):(start+11)
        tr_dat = [tr_dat;AllFaceFeatures{i}];
        if (i == 2)||(i == 5)||(i == 6)||(i == 10)
            tr_lb = [tr_lb;seg{i}];
        else
            tr_lb = [tr_lb;Label_now{i}];
        end
    end
    
    if times == time_max
        break;
    end
    
    tr_dat = tr_dat';
    structure = [size(tr_dat, 1) 30 30 30 30 10 numel(unique(tr_lb))];
    X_Train = tr_dat';
    Y_Train = categorical(tr_lb);
    
    %设置MLP的layers
    layers = [featureInputLayer(structure(1))];
    for i = 2:(size(structure, 2)-1)
        layers = [layers fullyConnectedLayer(structure(i)) reluLayer()];
    end
    layers = [layers fullyConnectedLayer(structure(size(structure, 2)))];
    layers = [layers softmaxLayer() classificationLayer()];
    
    training_options = trainingOptions('adam', ...
        'InitialLearnRate', 0.001, ...
        'MaxEpochs', 15, ...
        'MiniBatchSize', 1024, ...
        'L2Regularization', 0.0001, ...
        'Verbose', true, ...
        'Plots', 'training-progress');
    
    net = [];
    net.Layers = layers;
    disp(['Training：  ',int2str(times)]);
    net = trainNetwork(X_Train, Y_Train, net.Layers, training_options);
    times = times+1;
end
%%
%导出前十二个模型的标签
sumRatio = 0;
for i = (start):(start+11)
    modelFileName = [indir, model{1, i}];          %读入对应模型
    [vertex, face] = read_mesh(modelFileName);
    segResult = Label_now{i};
    figure;
    szy_PlotMesh_Discrete_vf(vertex,face,segResult);
    segResultFileName = ['./Result_W/', int2str(i), '.seg'];
    dlmwrite(segResultFileName, segResult - 1);
    szy_WriteMeshWithFaceColor_Discrete(vertex, face, [segResultFileName, '.obj'], segResult);
    successRatio = Get_SuccessRatio(segResult,seg{i},Areas{i}');
    sumRatio = sumRatio + successRatio;
    disp(['W result of ',int2str(i),' with net_2 is ',num2str(successRatio)]);
end
sumRatio = sumRatio ./ 12;
disp(['Average ratio of is ', num2str(sumRatio)]);
%%
% W with GC
sumRatio = 0;
for i = (start):(start+11)
    modelFileName = [indir, model{1, i}];          %读入对应模型
    [vertex, face] = read_mesh(modelFileName);
    Prob_ = 1./Prob{i};
    Prob_ = ReplaceInf(Prob_);
    segResult = szy_GraphCut_vf(vertex,face,0.05,Prob_',false);
    Label_now{i} = segResult;
    figure;
    szy_PlotMesh_Discrete_vf(vertex,face,segResult);
    segResultFileName = ['./Result_W_GC/', int2str(i), '.seg'];
    dlmwrite(segResultFileName, segResult - 1);
    szy_WriteMeshWithFaceColor_Discrete(vertex, face, [segResultFileName, '.obj'], segResult);
    successRatio = Get_SuccessRatio(segResult,seg{i},Areas{i}');
    sumRatio = sumRatio + successRatio;
    disp(['W_GC result of ',int2str(i),' with net_2 is ',num2str(successRatio)]);
end
sumRatio = sumRatio ./ 12;
disp(['Average ratio of is ', num2str(sumRatio)]);
%%
%调试网络
tr_dat = [];
tr_lb = [];

%构造tr_dat
for i = start:(start+11)
    tr_dat = [tr_dat;AllFaceFeatures{i}];
    if (i == 2)||(i == 5)||(i == 6)||(i == 10)
        tr_lb = [tr_lb;seg{i}];
    else
        segPath = ['./Result_W_GC/', int2str(i), '.seg'];
        Seg = load(segPath)+1;
        tr_lb = [tr_lb;Seg];
    end
end
tr_dat = tr_dat';
structure = [size(tr_dat, 1) 30 30 10 numel(unique(tr_lb))];
X_Train = tr_dat';
Y_Train = categorical(tr_lb);

%设置MLP的layers
layers = [featureInputLayer(structure(1))];
for i = 2:(size(structure, 2)-1)
    layers = [layers fullyConnectedLayer(structure(i)) reluLayer()];
end
layers = [layers fullyConnectedLayer(structure(size(structure, 2)))];
layers = [layers softmaxLayer() classificationLayer()];

        
%设置训练选项
training_options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 1024, ...
    'L2Regularization', 0.0001, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

final_net = [];
final_net.Layers = layers;
final_net = trainNetwork(X_Train, Y_Train, final_net.Layers, training_options);
%%
%看看在训练集上的误差
sumRatio = 0;
for i = (start):(start+11)
    modelFileName = [indir, model{1, i}];          %读入对应模型
    [vertex, face] = read_mesh(modelFileName);
    segResult = classify(final_net,AllFaceFeatures{i});
    segResult = double(segResult);
    figure;
    szy_PlotMesh_Discrete_vf(vertex,face,segResult);
    segResultFileName = ['./Result_tt/', int2str(i), '.seg'];
    dlmwrite(segResultFileName, segResult - 1);
    szy_WriteMeshWithFaceColor_Discrete(vertex, face, [segResultFileName, '.obj'], segResult);
    successRatio = Get_SuccessRatio(segResult,seg{i},Areas{i}');
    sumRatio = sumRatio + successRatio;
    disp(['Train result of ',int2str(i),' with final_net is ',num2str(successRatio)]);
end
sumRatio = sumRatio ./ 12;
disp(['Average ratio of is ', num2str(sumRatio)]);
%%
sumRatio = 0;
for i = (start+12):(start+19)
    modelFileName = [indir, model{1, i}];          %读入对应模型
    [vertex, face] = read_mesh(modelFileName);
    segResult = classify(final_net,AllFaceFeatures{i});
    segResult = double(segResult);
    figure;
    szy_PlotMesh_Discrete_vf(vertex,face,segResult);
    segResultFileName = ['./Result_tt/', int2str(i), '.seg'];
    dlmwrite(segResultFileName, segResult - 1);
    szy_WriteMeshWithFaceColor_Discrete(vertex, face, [segResultFileName, '.obj'], segResult);
    successRatio = Get_SuccessRatio(segResult,seg{i},Areas{i}');
    sumRatio = sumRatio + successRatio;
    disp(['Test result of ',int2str(i),' with final_net is ',num2str(successRatio)]);
end
sumRatio = sumRatio ./ 8;
disp(['Average ratio of is ', num2str(sumRatio)]);
%%
sumRatio = 0;
for i = (start+12):(start+19)
    modelFileName = [indir, model{1, i}];          %读入对应模型
    [vertex, face] = read_mesh(modelFileName);
    pred_y = predict(final_net, AllFaceFeatures{i});
    segResult = szy_GraphCut_vf(vertex,face,0.1,pred_y',false);
    figure;
    szy_PlotMesh_Discrete_vf(vertex,face,segResult);
    segResultFileName = ['./Result_tt_GC/', int2str(i), '.seg'];
    dlmwrite(segResultFileName, segResult - 1);
    szy_WriteMeshWithFaceColor_Discrete(vertex, face, [segResultFileName, '.obj'], segResult);
    successRatio = Get_SuccessRatio(segResult,seg{i},Areas{i}');
    sumRatio = sumRatio + successRatio;
    disp(['Test result of ',int2str(i),' with final_net is ',num2str(successRatio)]);
end
sumRatio = sumRatio ./ 8;
disp(['Average ratio of is ', num2str(sumRatio)]);
%%
%12个模型全用GT训练一个网络
%调试网络
tr_dat = [];
tr_lb = [];

%构造tr_dat
for i = start:(start+11)
    tr_dat = [tr_dat;AllFaceFeatures{i}];
    tr_lb = [tr_lb;seg{i}];
end
tr_dat = tr_dat';
structure = [size(tr_dat, 1) 30 30 30 10 numel(unique(tr_lb))];
X_Train = tr_dat';
Y_Train = categorical(tr_lb);

%设置MLP的layers
layers = [featureInputLayer(structure(1))];
for i = 2:(size(structure, 2)-1)
    layers = [layers fullyConnectedLayer(structure(i)) reluLayer()];
end
layers = [layers fullyConnectedLayer(structure(size(structure, 2)))];
layers = [layers softmaxLayer() classificationLayer()];

        
%设置训练选项
training_options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 1024, ...
    'L2Regularization', 0.0001, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

comp_net_1 = [];
comp_net_1.Layers = layers;
comp_net_1 = trainNetwork(X_Train, Y_Train, comp_net_1.Layers, training_options);
%%
%测试集结果
sumRatio = 0;
for i = (start+12):(start+19)
    modelFileName = [indir, model{1, i}];          %读入对应模型
    [vertex, face] = read_mesh(modelFileName);
    segResult = classify(comp_net_1,AllFaceFeatures{i});
    segResult = double(segResult);
    figure;
    szy_PlotMesh_Discrete_vf(vertex,face,segResult);
    segResultFileName = ['./Result_tt/', int2str(i), '.seg'];
    dlmwrite(segResultFileName, segResult - 1);
    szy_WriteMeshWithFaceColor_Discrete(vertex, face, [segResultFileName, '.obj'], segResult);
    successRatio = Get_SuccessRatio(segResult,seg{i},Areas{i}');
    sumRatio = sumRatio + successRatio;
    disp(['Test result of ',int2str(i),' with final_net is ',num2str(successRatio)]);
end
sumRatio = sumRatio ./ 8;
disp(['Average ratio of is ', num2str(sumRatio)]);
%%
%用4个模型全用GT训练一个网络
%调试网络
tr_dat = [];
tr_lb = [];

%构造tr_dat
for i = start:(start+11)
    if (i == 2)||(i == 5)||(i == 6)||(i == 10)
        tr_dat = [tr_dat;AllFaceFeatures{i}];
        tr_lb = [tr_lb;seg{i}];
    end
end
tr_dat = tr_dat';
structure = [size(tr_dat, 1)  30 30 30 10 numel(unique(tr_lb))];
X_Train = tr_dat';
Y_Train = categorical(tr_lb);

%设置MLP的layers
layers = [featureInputLayer(structure(1))];
for i = 2:(size(structure, 2)-1)
    layers = [layers fullyConnectedLayer(structure(i)) reluLayer()];
end
layers = [layers fullyConnectedLayer(structure(size(structure, 2)))];
layers = [layers softmaxLayer() classificationLayer()];

        
%设置训练选项
training_options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1024, ...
    'L2Regularization', 0.0001, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

comp_net_2 = [];
comp_net_2.Layers = layers;
comp_net_2 = trainNetwork(X_Train, Y_Train, comp_net_2.Layers, training_options);
%%
%测试集结果
sumRatio = 0;
for i = (start+12):(start+19)
    modelFileName = [indir, model{1, i}];          %读入对应模型
    [vertex, face] = read_mesh(modelFileName);
    segResult = classify(comp_net_2,AllFaceFeatures{i});
    segResult = double(segResult);
    figure;
    szy_PlotMesh_Discrete_vf(vertex,face,segResult);
    segResultFileName = ['./Result_tt/', int2str(i), '.seg'];
    dlmwrite(segResultFileName, segResult - 1);
    szy_WriteMeshWithFaceColor_Discrete(vertex, face, [segResultFileName, '.obj'], segResult);
    successRatio = Get_SuccessRatio(segResult,seg{i},Areas{i}');
    sumRatio = sumRatio + successRatio;
    disp(['Test result of ',int2str(i),' with final_net is ',num2str(successRatio)]);
end
sumRatio = sumRatio ./ 8;
disp(['Average ratio of is ', num2str(sumRatio)]);