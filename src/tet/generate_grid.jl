"""
    hypercube(Tet{Float64}, n; scale = 1) -> Mesh

Create a cube of size scale * (n x n x n) where each cell is split into 5 
tetrahedra.
"""
function hypercube(::Type{<:Tet{Tv}}, n::Int, Ti::Type{<:Integer} = Int; scale = 1, origin = (1,1,1)) where {Tv}
    Nn = (n + 1) * (n + 1) * (n + 1)
    Ne = 6 * n * n * n
    nn = reshape(1 : Nn, n + 1, n + 1, n + 1)

    nodes = Vector{SVector{3,Tv}}(undef, Nn)
    elements = Vector{NTuple{4,Ti}}(undef, Ne)

    # Construct the nodes
    node_idx = 0
    @inbounds for x = 1 : n + 1, y = 1 : n + 1, z = 1 : n + 1
        nodes[node_idx += 1] = (scale * (x - 1) + origin[1], scale * (y - 1) + origin[2], scale * (z - 1) + origin[3])
    end

    # Construct the elements
    element_idx = 0
    @inbounds for x = 1 : n, y = 1 : n, z = 1 : n
        n1 = nn[x    , y    , z]
        n2 = nn[x + 1, y    , z]
        n3 = nn[x    , y + 1, z]
        n4 = nn[x + 1, y + 1, z]
        n5 = nn[x    , y    , z + 1]
        n6 = nn[x + 1, y    , z + 1]
        n7 = nn[x    , y + 1, z + 1]
        n8 = nn[x + 1, y + 1, z + 1]

        # Specific ordering allows refine_uniformly to generate tets
        # aligned with a uniform grid.
        elements[element_idx += 1] = (n1,n2,n3,n7)
        elements[element_idx += 1] = (n1,n2,n5,n7)
        elements[element_idx += 1] = (n2,n4,n3,n7)
        elements[element_idx += 1] = (n2,n4,n7,n8)
        elements[element_idx += 1] = (n2,n6,n5,n7)
        elements[element_idx += 1] = (n2,n6,n7,n8)
    end

    sort_element_nodes!(elements)

    return Mesh(nodes, elements)
end
