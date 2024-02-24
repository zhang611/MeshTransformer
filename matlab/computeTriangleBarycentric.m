function barycentric_cordinate = computeTriangleBarycentric(mesh_vertex, mesh_face)
    barycentric_cordinate = zeros(size(mesh_face, 2), 3);
    for i = 1:size(barycentric_cordinate, 1)
        points = mesh_face(:, i);
        point_a = mesh_vertex(:, points(1));
        point_b = mesh_vertex(:, points(2));
        point_c = mesh_vertex(:, points(3));
        point_center = (point_a + point_b + point_c) ./ 3;
        barycentric_cordinate(i, :) = point_center;
    end
end