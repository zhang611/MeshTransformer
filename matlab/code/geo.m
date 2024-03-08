% 三角形的顶点坐标
A = [0, 0, 0];
B = [1, 0, 0];
C = [0.5, 1, 0];

% 选择旋转边（这里选择边AB）
rotation_axis = B - A;

% 选择旋转角度（这里选择旋转30度）
rotation_angle = deg2rad(30);

% 构造旋转矩阵
rotation_matrix = make_rotation_matrix(rotation_axis, rotation_angle);

% 将三角形的每个顶点通过旋转矩阵进行变换
rotated_A = rotation_matrix * A';
rotated_B = rotation_matrix * B';
rotated_C = rotation_matrix * C';

% 绘制原始三角形和旋转后的三角形
figure;

% 绘制原始三角形
patch([A(1), B(1), C(1)], [A(2), B(2), C(2)], [A(3), B(3), C(3)], 'r', 'FaceAlpha', 0.5);
hold on;

% 绘制旋转后的三角形
patch([rotated_A(1), rotated_B(1), rotated_C(1)], [rotated_A(2), rotated_B(2), rotated_C(2)], [rotated_A(3), rotated_B(3), rotated_C(3)], 'b', 'FaceAlpha', 0.5);

axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Rotated Triangle');



