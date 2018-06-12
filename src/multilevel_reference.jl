using Base: @propagate_inbounds

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

"""
ReferenceNumbering keeps track of the local numbering of the faces, edges 
and nodes of a refined reference element.
"""
struct ReferenceNumbering{Ti}
    faces::Vector{Vector{Ti}}
    faces_interior::Vector{Vector{Ti}}
    edges::Vector{Vector{Ti}}
    edges_interior::Vector{Vector{Ti}}
    nodes::Vector{Ti}
end

"""
MultilevelReference stores a bunch of refined meshes with interpolation operators
and at the same time keeps track of the local numbering of the faces, edges and
nodes.
"""
struct MultilevelReference{dim,N,Tv,Ti}
    levels::Vector{Mesh{dim,N,Tv,Ti}}
    numbering::Vector{ReferenceNumbering{Ti}}
    interops::Vector{SparseMatrixCSC{Tv,Ti}}
end

"""
Return a multilevel structure of refined grids with interpolation operators.
"""
function refined_element(n::Int, m::Type{Mesh{dim,N,Tv,Ti}}) where {dim,N,Tv,Ti}
    levels = Vector{m}(n)
    interops = Vector{SparseMatrixCSC{Tv,Ti}}(n - 1)
    numbering = Vector{ReferenceNumbering{Ti}}(n)
    levels[1] = reference_element(m)
    numbering[1] = get_local_numbering(levels[1])

    for i = 1 : n - 1
        graph = edge_graph(levels[i])
        levels[i + 1] = refine_uniformly(levels[i], graph)
        numbering[i + 1] = get_local_numbering(levels[i + 1])
        interops[i] = interpolation_operator(levels[i], graph)
    end

    # Note sure if necessary.
    for mesh in levels
        sort_element_nodes!(mesh.elements)
    end

    MultilevelReference(levels, numbering, interops)
end

function nodes_on_ref_faces(m::Tets{Tv}) where {Tv}
    return [
        find(x -> x[3] == 0, m.nodes),
        find(x -> x[2] == 0, m.nodes),
        find(x -> x[1] == 0, m.nodes),
        find(x -> x[1] + x[2] + x[3] ≥ 1 - 10 * eps(Tv), m.nodes) # rounding errors!
    ]
end

function nodes_on_ref_edges(m::Tris{Tv}) where {Tv}
    return [
        find(x -> x[2] == 0, m.nodes),
        find(x -> x[1] == 0, m.nodes),
        find(x -> x[1] + x[2] ≥ 1 - 10 * eps(Tv), m.nodes) # rounding errors!
    ]
end

"""
Given two points a and b, tests whether x is on a → b.
"""
struct IsOnEdge{N,Tv} <: Function
    unit::SVector{N,Tv}
    origin::SVector{N,Tv}

    IsOnEdge{N,Tv}(a,b) where {N,Tv} = new(a,b)
end

# Constructor
@inline function IsOnEdge(from::SVector{N,Tv}, to::SVector{N,Tv}) where {N,Tv}
    diff = to - from
    IsOnEdge{N,Tv}(diff / norm(diff), from)
end

# Application on a point x
@inline function (n::IsOnEdge{N,Tv})(x::SVector{N,Tv}) where {N,Tv}
    vec = x - n.origin
    proj = dot(n.unit, vec)
    abs(proj * proj - dot(vec, vec)) < 1e-7 # todo, fix this.
end

function nodes_on_ref_edges(m::Tets{Tv,Ti}) where {Tv,Ti}
    ref_nodes = get_reference_nodes(Tet{Tv})

    @inbounds nodes_per_edge = [
        find(IsOnEdge(ref_nodes[1], ref_nodes[2]), m.nodes),
        find(IsOnEdge(ref_nodes[1], ref_nodes[3]), m.nodes),
        find(IsOnEdge(ref_nodes[1], ref_nodes[4]), m.nodes),
        find(IsOnEdge(ref_nodes[2], ref_nodes[3]), m.nodes),
        find(IsOnEdge(ref_nodes[2], ref_nodes[4]), m.nodes),
        find(IsOnEdge(ref_nodes[3], ref_nodes[4]), m.nodes)
    ]

    nodes_per_edge
end

"""
    get_local_numbering(mesh::Tets) -> ReferenceNumbering

Get the indices of the nodes in (the interior of) the faces, the edges and the 
corner points. There's probably a cheaper way to do it by keeping track of
them during consecutive refinements, but well...
"""
function get_local_numbering(m::Tets{Tv}) where {Tv}
    # First collect the nodes on all four faces.
    face_to_nodes = nodes_on_ref_faces(m)
    ref_nodes = get_reference_nodes(Tet{Tv})

    # Then find the subset of nodes on the edges
    # From face 1 = 1→2→3 we find edge 1→2, 1→3
    # From face 2 = 1→2→4 we fine edge 1→4
    # From face 3 = 1→3→4 we find nothing
    # From face 4 = 2→3→4 we find edge 2→3, 2→3, 3→4

    edge_to_nodes = [
        filter(i -> IsOnEdge(ref_nodes[1], ref_nodes[2])(m.nodes[i]), face_to_nodes[1]),
        filter(i -> IsOnEdge(ref_nodes[1], ref_nodes[3])(m.nodes[i]), face_to_nodes[1]),
        filter(i -> IsOnEdge(ref_nodes[1], ref_nodes[4])(m.nodes[i]), face_to_nodes[2]),
        filter(i -> IsOnEdge(ref_nodes[2], ref_nodes[3])(m.nodes[i]), face_to_nodes[4]),
        filter(i -> IsOnEdge(ref_nodes[2], ref_nodes[4])(m.nodes[i]), face_to_nodes[4]),
        filter(i -> IsOnEdge(ref_nodes[3], ref_nodes[4])(m.nodes[i]), face_to_nodes[4]),
    ]

    # Finally find the nodes in the corners (well...)
    nodes_to_nodes = collect(1:4)

    interior_face_to_nodes = deepcopy(face_to_nodes)
    interior_edge_to_nodes = deepcopy(edge_to_nodes)

    # Now remove the boundaries from the faces and the endpoints from the edges.
    left_minus_right!(interior_face_to_nodes[1], edge_to_nodes[1]) # 1→2
    left_minus_right!(interior_face_to_nodes[1], edge_to_nodes[2]) # 1→3
    left_minus_right!(interior_face_to_nodes[1], edge_to_nodes[4]) # 2→3

    left_minus_right!(interior_face_to_nodes[2], edge_to_nodes[1]) # 1→2
    left_minus_right!(interior_face_to_nodes[2], edge_to_nodes[3]) # 1→4
    left_minus_right!(interior_face_to_nodes[2], edge_to_nodes[5]) # 2→4

    left_minus_right!(interior_face_to_nodes[3], edge_to_nodes[2]) # 1→3
    left_minus_right!(interior_face_to_nodes[3], edge_to_nodes[3]) # 1→4
    left_minus_right!(interior_face_to_nodes[3], edge_to_nodes[6]) # 3→4

    left_minus_right!(interior_face_to_nodes[4], edge_to_nodes[4]) # 2→3
    left_minus_right!(interior_face_to_nodes[4], edge_to_nodes[5]) # 2→4
    left_minus_right!(interior_face_to_nodes[4], edge_to_nodes[6]) # 3→4

    # Remove the end points from the edges
    for i = 1 : 6
        left_minus_right!(interior_edge_to_nodes[i], nodes_to_nodes)
    end

    return ReferenceNumbering(
        face_to_nodes,
        interior_face_to_nodes,
        edge_to_nodes,
        interior_edge_to_nodes,
        nodes_to_nodes
    )
end

function get_local_numbering(m::Tris{Tv}) where {Tv}
    # First collect the nodes on all three edges
    edge_to_nodes = nodes_on_ref_edges(m)
    nodes_to_nodes = collect(1:3)
    interior_edge_to_nodes = deepcopy(edge_to_nodes)

    # Remove the end points from the edges
    for i = 1 : 3
        left_minus_right!(interior_edge_to_nodes[i], nodes_to_nodes)
    end

    face_to_nodes = [Int[]]
    interior_face_to_nodes = [Int[]]

    return ReferenceNumbering(
        face_to_nodes,
        interior_face_to_nodes,
        edge_to_nodes,
        interior_edge_to_nodes,
        nodes_to_nodes
    )
end

"""
    nodes_per_face(::MultilevelReference, level)

Return the number of nodes on the faces at refinement level `level`
"""
@propagate_inbounds nodes_per_face(ref::MultilevelReference, level::Int) = length(ref.numbering[level].faces[1])

"""
    nodes_per_face_interior(::MultilevelReference, level)

Return the number of nodes on the interior of the faces at refinement level `level`
"""
@propagate_inbounds nodes_per_face_interior(ref::MultilevelReference, level::Int) = length(ref.numbering[level].faces_interior[1])

"""
    nodes_per_edge(::MultilevelReference, level)

Return the number of nodes on the edges at refinement level `level`
"""
@propagate_inbounds nodes_per_edge(ref::MultilevelReference, level::Int) = length(ref.numbering[level].edges[1])

"""
    nodes_per_edge_interior(::MultilevelReference, level)

Return the number of nodes on the interior of the edges at refinement level `level`
"""
@propagate_inbounds nodes_per_edge_interior(ref::MultilevelReference, level::Int) = length(ref.numbering[level].edges_interior[1])
