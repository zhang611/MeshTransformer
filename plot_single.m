num = 13;
num2 = 1;
epochs = 220;
dir = 'chairLarge\';
best_lambda = 0.1;
class = 'Irons\';

labels = load(['E:\3DModelData\COSEG\',class, num2str(num),'.seg']);
preds = load(['F:\zym\PSB_1500\results_1500\',dir,num2str(epochs),'\', num2str(num2), '.seg']);
[v,f] = read_mesh(['E:\3DModelData\COSEG\',class, num2str(num),'.off']);
prob = load(['F:\zym\PSB_1500\results_1500\',dir,num2str(epochs),'\', num2str(num2), '.prob']);
meshNum = size(f,2);

acc = labels == preds;   
disp(['acc_noCut: ',num2str(sum(acc)/meshNum)]);

labels_cut = szy_GraphCut_vf(v, f, best_lambda, prob, false);
labels_cut = labels_cut-1;

acc = labels == labels_cut;
disp(['acc_cut: ',num2str(sum(acc)/meshNum)]);

%title(num2str(meshNum))
subplot(1,3,1)
szy_PlotMesh_Discrete_vf(v, f, labels);
title('gt')
axis off

subplot(1,3,2)
szy_PlotMesh_Discrete_vf(v, f, preds);
title('seg-noCut')
axis off

subplot(1,3,3)
szy_PlotMesh_Discrete_vf(v, f, labels_cut);
axis off
title('seg-cut')