"""
Uniformly refine a mesh of tetrahedrons: each tetrahedron is split into eight
new tetrahedrons.
"""
function refine_uniformly(mesh::Tets{Tv,Ti}, graph::SparseGraph) where {Tv,Ti}
    Nn = length(mesh.nodes)
    Nt = length(mesh.elements)
    Ne = length(graph.adj)

    ### Refine the grid.
    nodes = Vector{SVector{3,Tv}}(undef, Nn + Ne)
    copyto!(nodes, mesh.nodes)

    ## Split the edges
    @inbounds begin
        idx = Nn + 1
        for from = 1 : Nn, to = graph.ptr[from] : graph.ptr[from + 1] - 1
            nodes[idx] = (mesh.nodes[from] + mesh.nodes[graph.adj[to]]) / 2
            idx += 1
        end
    end

    ## Next, build new tetrahedrons...
    tets = Vector{NTuple{4,Ti}}(undef, 8Nt)
    parts = Vector{Ti}(undef, 10)

    tet_idx = 1
    offset = Ti(Nn)
    @inbounds for tet in mesh.elements

        # Collect the nodes
        parts[1] = tet[1]
        parts[2] = tet[2]
        parts[3] = tet[3]
        parts[4] = tet[4]

        # Find the mid-points (6 of them)
        idx = 5
        for i = 1 : 4, j = i + 1 : 4
            from, to = sort_bitonic((tet[i], tet[j]))
            parts[idx] = edge_index(graph, from, to) + offset
            idx += 1
        end

        # Generate new tets!
        for (a,b,c,d) in ((1,5,6,7), (5,2,8,9), (6,8,3,10), (7,9,10,4),
                          (5,6,7,9), (5,6,8,9), (6,7,9,10), (6,8,9,10))
            tets[tet_idx] = (parts[a], parts[b], parts[c], parts[d])
            tet_idx += 1
        end
    end

    return Tets{Tv,Ti}(nodes, tets)
end
