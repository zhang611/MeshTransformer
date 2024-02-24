% [0]Human-8   [1]Cup-2    [2]Glasses-3     [3]Airplane-5    [4]Ant-5    [5]Chair-4 
% [6]Octopus-2 [7]Table-2  [8]Teddy-5       [9]Hand-6        [10]Plier-3
% [11]Fish-3  [12]Bird-5 [14]Armadillo-11 [18]Vase-5 [19]Fourleg-6
clear
results = zeros(5,8);
best_res = zeros(5,8);
num = 15;
epochs = 311;
dir = 'human_0\';
best_lambda = 0;
best_accCut = 0;

label_dir = 'E:\3DModelData\COSEG\Fourleg\';
mesh_dir = 'E:\3DModelData\COSEG\Fourleg\';
res_dir = 'F:\zym\PSB_1500\results_1500\chairLarge\'
test_len = 6
for lambda = 0.01:0.02:0.5
    disp(num2str(lambda));
    for i = 1:test_len
        labels = load([label_dir, num2str(i+num-1),'.seg']);
        [v,f] = read_mesh([mesh_dir, num2str(i+num-1),'.off']);
        for j = 1:test_len
            preds = load([res_dir,num2str(epochs),'\', num2str(j), '.seg']);
            if size(labels,1)==size(preds,1)
                prob = load([res_dir,num2str(epochs),'\', num2str(j), '.prob']);
                meshNum = size(f,2);
                acc1 = labels == preds;  
                acc1 = sum(acc1)/meshNum;
                %disp(['acc_noCut: ',num2str(acc1)]);
            
                labels_cut = szy_GraphCut_vf(v, f, lambda, prob, false);
                labels_cut = labels_cut-1;

                acc2 = labels == labels_cut;
                acc2 = sum(acc2)/meshNum;
                %disp(['acc_cut: ',num2str(acc2)]);
            
                results(1,i) = meshNum;
                results(2,i) = i+num-1;
                results(3,i) = j;
                results(4,i) = acc1;
                results(5,i) = acc2;
            end
        end
    end
    avg1 = sum(results(4,:))/8;
    avg2 = sum(results(5,:))/8;
    if avg2 > best_accCut
        best_accCut = avg2;
        best_lambda = lambda;
        best_res = results;
    end
end    
disp(['acc_noCut: ',num2str(avg1)]);
disp(['acc_cut: ',num2str(best_accCut)]);
best_res = best_res';

            



%%%%%%%%%%%%%%%%%%%%%

% temp = zeros(1,100);
% best_acc = 0; best_lambda = 0;
% for i=1:100
%     labels_cut = szy_GraphCut_vf(v, f, i/100, prob, false);
%     labels_cut = labels_cut-1;
% 
%     acc = labels == labels_cut;
%     acc = sum(acc)/meshNum;
%     if acc > best_acc
%         best_acc = acc;
%         best_lambda = i/100;
%     end
%     temp(i) = acc;
%     %disp(['lambda=',num2str(i/100),', acc_cut: ',num2str(acc)]);
% end
% disp(['best_acc=',num2str(best_acc),', lambda=',num2str(best_lambda)])
% 
