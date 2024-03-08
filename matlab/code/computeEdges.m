function edges = computeEdges(adjacencyMatrix)
    % Given an adjacency matrix, compute the edges of the mesh
    [i, j] = find(adjacencyMatrix);
    edges = [i, j];
end