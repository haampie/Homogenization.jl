"""
Uniformly refine a mesh of triangles: each triangle
is split into four new triangles.
"""
function refine_uniformly(mesh::Tris{Tv,Ti}, graph::SparseGraph) where {Tv,Ti}
    Nn = length(mesh.nodes)
    Nt = length(mesh.elements)
    Ne = length(graph.adj)

    # Each edge is split 2, so Nn + Ne is the number of nodes
    nodes = Vector{SVector{2,Tv}}(Nn + Ne)

    # Each triangle is split in 4, so 4Nt triangles
    triangles = Vector{NTuple{3,Ti}}(4Nt)

    # Keep the old nodes in place
    copy!(nodes, mesh.nodes)
    
    # Add the new ones
    idx = Nn + 1
    @inbounds for i = 1 : length(graph.ptr) - 1, j = graph.ptr[i] : graph.ptr[i + 1] - 1
        nodes[idx] = (mesh.nodes[i] + mesh.nodes[graph.adj[j]]) / 2
        idx += 1
    end

    # Split each triangle in four smaller ones
    @inbounds for (i, t) in enumerate(mesh.elements)

        # Index of the nodes on the new edges
        a = Ti(edge_index(graph, t[1], t[2]) + Nn)
        b = Ti(edge_index(graph, t[1], t[3]) + Nn)
        c = Ti(edge_index(graph, t[2], t[3]) + Nn)

        # Split the triangle in 4 pieces
        idx = 4i - 3
        triangles[idx + 0] = sort_bitonic((t[1], a, b))
        triangles[idx + 1] = sort_bitonic((t[2], c, a))
        triangles[idx + 2] = sort_bitonic((t[3], b, c))
        triangles[idx + 3] = sort_bitonic((a   , c, b))
    end

    return Tris{Tv,Ti}(nodes, triangles)
end