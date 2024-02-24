clc,clear 
close all
%% 

offPath='F:\zym\PSB_new\off\';
offDir = dir([offPath '*.off']); % 遍历所有off格式文件
labels = load(['F:\zym\PSB_new\seg_consistent\',num2str(i+num-1),'.seg']);

for i = 1:length(offDir)/2          % 遍历结构体就可以一一处理图片了
    strs = [offPath offDir(i).name];
    pth = strsplit(strs,'.');
    filename=strsplit(offDir(i).name,'.');
    filename=char(filename(1));
    [V,F,UV,C,N] = readOFF(strs);
    vertex_GD_Matrix = szy_GetGeodesicDistanceMatrix_dijkstra_vf(V', F');
    save(filename,'vertex_GD_Matrix')
end

num_face = length(F);
d_face = zeros(num_face,num_face);
for i = 1:num_face
    i1 = F(i,1); i2 = F(i,2); i3 = F(i,3);
    d_face(i,:) = vertex_GD_Matrix(i1,:) + vertex_GD_Matrix(i2,:) + vertex_GD_Matrix(i3,:);
end
d_face = d_face/3;
    
    