function edgeLengths = calculateEdgeLengths(vertices, edges)
    % Calculate the lengths of edges based on vertex coordinates
    v1 = vertices(:, edges(:, 1));
    v2 = vertices(:, edges(:, 2));
    edgeLengths = sqrt(sum((v1 - v2).^2));
end