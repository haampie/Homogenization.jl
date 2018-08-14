"""
Sparse graph structure like SparseMatrixCSC without values.
"""
struct SparseGraph{Ti}
    ptr::Vector{Ti}
    adj::Vector{Ti}
end


"""
Given a sorted edge between nodes (n1 → n2), return the natural index of the edge.
Costs are O(log b) where b is the connectivity
"""
edge_index(graph::SparseGraph{Ti}, n1::Ti, n2::Ti) where {Ti <: Integer} =
    binary_search(graph.adj, n2, graph.ptr[n1], graph.ptr[n1 + 1] - one(Ti))

"""
Construct an edge graph for simplices
"""
function edge_graph(mesh::Mesh{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    Nn = length(mesh.nodes)
    ptr = zeros(Ti, Nn + 1)

    # Count edges per node
    @inbounds for element in mesh.elements, i = 1 : N, j = i + 1 : N
        from, = sort_bitonic((element[i], element[j]))
        ptr[from + Ti(1)] += Ti(1)
    end

    # Accumulate
    ptr[1] = 1
    @inbounds for i = 1 : Nn
        ptr[i + 1] += ptr[i]
    end

    # Build adjacency list
    adj = Vector{Ti}(undef, ptr[end] - 1)
    indices = copy(ptr)

    @inbounds for element in mesh.elements, i = 1 : N, j = i + 1 : N
        from, to = sort_bitonic((element[i], element[j]))
        # @assert from < to
        adj[indices[from]] = to
        indices[from] += 1
    end

    remove_duplicates!(sort_edges!(SparseGraph(ptr, adj)))
end

"""
Remove duplicate edges from an adjacency list with sorted edges
"""
function remove_duplicates!(g::SparseGraph)
    Nn = length(g.ptr) - 1
    slow = 0
    fast = 1

    @inbounds for next = 2 : Nn + 1
        last = g.ptr[next]

        # If there is an edge going out from `node` copy the first one to the
        # `slow` position and copy the remaining unique edges after it
        if fast < last

            # Copy the first 'slow' item
            slow += 1
            g.adj[slow] = g.adj[fast]
            fast += 1

            # From then on only copy distinct values
            while fast < last
                if g.adj[fast] != g.adj[slow]
                    slow += 1
                    g.adj[slow] = g.adj[fast]
                end
                fast += 1
            end
        end

        g.ptr[next] = slow + 1
    end

    # Finally we resize the adjacency list
    resize!(g.adj, slow)

    return g
end
