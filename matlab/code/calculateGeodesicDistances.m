function geodesicDistances = calculateGeodesicDistances(vertices, adjacencyMatrix)
    % vertices: 3xN matrix of vertex coordinates
    % adjacencyMatrix: NxN adjacency matrix representing mesh connectivity
    
    % Compute edges from adjacency matrix
    edges = computeEdges(adjacencyMatrix);
    
    % Calculate edge lengths based on vertex coordinates
    edgeLengths = calculateEdgeLengths(vertices, edges);
    
    % Create a graph using the edges and edge lengths
    G = graph(edges(:, 1), edges(:, 2), edgeLengths);
    
    % Calculate geodesic distances between all pairs of vertices
    geodesicDistances = distances(G);
end