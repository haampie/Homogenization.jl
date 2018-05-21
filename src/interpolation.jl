"""
Return the interpolation operator assuming all linear basis elements and all
edges being split in two.
"""
function interpolation_operator(mesh::Mesh{dim,N,Tv,Ti}, graph::SparseGraph{Ti}) where {dim,N,Ti,Tv}
    # Interpolation operator
    Nn = length(mesh.nodes)
    Ne = length(graph.adj)

    nzval = Vector{Tv}(Nn + 2Ne)
    colptr = Vector{Int}(Nn + Ne + 1)
    rowval = Vector{Int}(Nn + 2Ne)

    # Nonzero values
    @inbounds for i = 1 : Nn
        nzval[i] = 1.0
    end

    @inbounds for i = Nn + 1 : Nn + 2Ne
        nzval[i] = 0.5
    end

    # Column pointer
    @inbounds for i = 1 : Nn + 1
        colptr[i] = i
    end

    @inbounds for i = Nn + 2 : Nn + Ne + 1
        colptr[i] = 2 + colptr[i - 1]
    end

    # Row values
    @inbounds for i = 1 : Nn
        rowval[i] = i
    end

    idx = Nn + 1
    @inbounds for i = 1 : Nn
        for j = graph.ptr[i] : graph.ptr[i + 1] - 1
            rowval[idx] = i
            rowval[idx + 1] = graph.adj[j]
            idx += 2
        end
    end

    # Note the transpose
    return SparseMatrixCSC(Nn, Nn + Ne, colptr, rowval, nzval)'
end