function reference_element(::Type{Tris{Tv,Ti}}) where {Tv,Ti}
    nodes = SVector{2,Tv}[(0,0),(1,0),(0,1)]
    elements = [(Base.OneTo(Ti(3))...)]
    Mesh(nodes, elements)
end

function reference_element(::Type{Tets{Tv,Ti}}) where {Tv,Ti}
    nodes = SVector{3,Tv}[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    elements = [(Base.OneTo(Ti(4))...)]
    Mesh(nodes, elements)
end

struct MultilevelReference{dim,N,Tv,Ti}
    levels::Vector{Mesh{dim,N,Tv,Ti}}
    interops::Vector{SparseMatrixCSC{Tv,Ti}}
end

"""
Return a multilevel structure of refined grids with interpolation operators.
"""
function refined_element(n::Int, m::Type{Mesh{dim,N,Tv,Ti}}) where {dim,N,Tv,Ti}
    levels = Vector{m}(n)
    interops = Vector{SparseMatrixCSC{Tv,Ti}}(n - 1)
    levels[1] = reference_element(m)

    for i = 1 : n - 1
        graph = edge_graph(levels[i])
        levels[i + 1] = refine_uniformly(levels[i], graph)
        interops[i] = interpolation_operator(levels[i], graph)
    end

    MultilevelReference(levels, interops)
end

function nodes_on_ref_faces(m::Tets{Tv}) where {Tv}
    return [
        find(x -> x[3] == 0, m.nodes),
        find(x -> x[2] == 0, m.nodes),
        find(x -> x[1] == 0, m.nodes),
        find(x -> x[1] + x[2] + x[3] ≥ 1 - 10 * eps(Tv), m.nodes)
    ]
end

function nodes_on_ref_edges(m::Tets{Tv,Ti}) where {Tv,Ti}
    ref_nodes = SVector{3,Tv}[(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
    nodes_per_edge = Vector{Vector{Ti}}(6)
    idx = 1
    for i = 1 : 4, j = i + 1 : 4
        # Find all nodes on the edge ref_nodes[i] → ref_nodes[j]
        a, b = ref_nodes[i], ref_nodes[j]
        unit = b - a
        unit /= norm(b - a)
        nodes_per_edge[idx] = find(x -> dot(unit, x - a) ≈ norm(x - a), m.nodes)
        idx += 1
    end

    nodes_per_edge
end