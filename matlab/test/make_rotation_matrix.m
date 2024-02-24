function R = make_rotation_matrix(axis, angle)
    % 构造绕任意轴旋转的旋转矩阵
    axis = axis / norm(axis);  % 将轴向量归一化
    ux = axis(1);
    uy = axis(2);
    uz = axis(3);

    cos_angle = cos(angle);
    sin_angle = sin(angle);

    R = [cos_angle + ux^2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle;
         uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy^2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle;
         uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz^2 * (1 - cos_angle)];
end