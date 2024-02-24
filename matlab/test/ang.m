% 三角形的顶点坐标（假设是相邻的两个三角形）
triangle1 = [0, 0, 0; 1, 0, 0; 0, 1, 0];
triangle2 = [1, 0, 0; 1, 1, 0; 0, 1, 0];

% 计算法向量
normal1 = cross(triangle1(2, :) - triangle1(1, :), triangle1(3, :) - triangle1(1, :));
normal2 = cross(triangle2(2, :) - triangle2(1, :), triangle2(3, :) - triangle2(1, :));

% 归一化法向量
normal1 = normal1 / norm(normal1);
normal2 = normal2 / norm(normal2);

% 计算夹角（弧度）
cosine_angle = dot(normal1, normal2);
angle_radians = acos(cosine_angle);

% 将弧度转换为度
angle_degrees = rad2deg(angle_radians);

disp(['相邻三角形的二面角：', num2str(angle_degrees), ' 度']);
