function reference_element(::Type{Tris{Tv,Ti}}) where {Tv,Ti}
    nodes = SVector{2,Tv}[(0,0),(0,1),(1,0)]
    elements = [(Base.OneTo(Ti(3))...)]
    Mesh(nodes, elements)
end

function reference_element(::Type{Tets{Tv,Ti}}) where {Tv,Ti}
    nodes = SVector{3,Tv}[(0,0,0),(0,0,1),(0,1,0),(1,0,0)]
    elements = [(Base.OneTo(Ti(4))...)]
    Mesh(nodes, elements)
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

    levels, interops
end