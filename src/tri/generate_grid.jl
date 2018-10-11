"""
    hypercube(Tri{Float64}, n; scale = 1) -> Mesh

Create a square of size scale * (n x n) where each cell is split into two 
triangles.
"""
function hypercube(::Type{<:Tri{Tv}}, n::Int, Ti::Type{<:Integer} = Int; scale = 1, origin = (1, 1), sorted = true) where {Tv}
    Nn = (n + 1) * (n + 1)
    Ne = 2 * n * n
    nn = reshape(1 : Nn, n + 1, n + 1)

    nodes = Vector{SVector{2,Tv}}(undef, Nn)
    elements = Vector{NTuple{3,Ti}}(undef, Ne)

    # Construct the nodes
    node_idx = 0
    @inbounds for x = 1 : n + 1, y = 1 : n + 1
        nodes[node_idx += 1] = (scale * (x - 1) + origin[1], scale * (y - 1) + origin[2])
    end

    # Construct the elements
    element_idx = 0
    @inbounds for x = 1 : n, y = 1 : n
        n1 = nn[x    , y    ]
        n2 = nn[x + 1, y    ]
        n3 = nn[x    , y + 1]
        n4 = nn[x + 1, y + 1]

        elements[element_idx += 1] = (n1, n2, n3)
        elements[element_idx += 1] = (n2, n3, n4)
    end

    sorted && sort_element_nodes!(elements)

    return Mesh(nodes, elements)
end
