using Base: @propagate_inbounds

struct Mesh{dim,N,Tv,Ti}
    nodes::Vector{SVector{dim,Tv}}
    elements::Vector{NTuple{N,Ti}}
end

const Tris{Tv,Ti} = Mesh{2,3,Tv,Ti}
const Tets{Tv,Ti} = Mesh{3,4,Tv,Ti}
const Tris64 = Tris{Float64,Int64}
const Tets64 = Tets{Float64,Int64}

abstract type ElementType{dim,N,Tv} end
abstract type Tri{Tv} <: ElementType{2,3,Tv} end
abstract type Tet{Tv} <: ElementType{3,4,Tv} end

const Tet64 = Tet{Float64}

cell_type(m::Mesh{2,3,Tv}) where {Tv} = Tri{Tv}
cell_type(m::Mesh{3,4,Tv}) where {Tv} = Tet{Tv}

function get_reference_nodes(::Type{Tet{Tv}}) where {Tv}
    return (SVector{3,Tv}(0,0,0),SVector{3,Tv}(1,0,0),SVector{3,Tv}(0,1,0),SVector{3,Tv}(0,0,1))
end

function get_reference_nodes(::Type{Tri{Tv}}) where {Tv}
    return (SVector{3,Tv}(0,0),SVector{3,Tv}(1,0),SVector{3,Tv}(0,1))
end

function get_reference_normals(::Type{Tet{Tv}}) where {Tv}
    return (
        SVector{3,Tv}(0,0,-1),
        SVector{3,Tv}(0,-1,0),
        SVector{3,Tv}(-1,0,0),
        SVector{3,Tv}(1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
    )
end

function refine_uniformly(m::Mesh; times::Int = 1)
    for i = 1 : times
        m = refine_uniformly(m, edge_graph(m))
    end
    m
end

"""
    nnodes(mesh) -> Int

Returns the number of nodes in the mesh
"""
nnodes(mesh::Mesh) = length(mesh.nodes)


"""
    nelements(mesh) -> Int

Returns the number of elements in the mesh
"""
nelements(mesh::Mesh) = length(mesh.elements)

"""
    dimension(mesh) -> Int

Returns the dimension of the mesh
"""
dimension(mesh::Mesh{dim}) where {dim} = dim

const TET_FACES = ((1,2,3),(1,2,4),(1,3,4),(2,3,4))
const TET_EDGES = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))
const TRI_EDGES = ((1,2),(1,3),(2,3))

"""
Returns the indices of an element that makes up face `i`
"""
@propagate_inbounds face_indices(::Type{<:Tet}, i::Int) = TET_FACES[i]


"""
Returns the indices of an element that makes up edge `i`
"""
@propagate_inbounds edge_indices(::Type{<:Tet}, i::Int) = TET_EDGES[i]

"""
Basically returns m.elements[element_idx][indices], but does not create a vector
of indices.
"""
@propagate_inbounds function get_node_numbers(m::Mesh{dim,N,Tv,Ti}, element_idx::Integer, indices::NTuple{M,Ti}) where {dim,N,Tv,Ti,M}
    el = m.elements[element_idx]
    NTuple{M,Ti}(el[i] for i in indices)
end

@propagate_inbounds function get_nodes(m::Mesh{dim,N,Tv,Ti}, element::NTuple{M,Ti}) where {dim,N,Tv,Ti,M}
    NTuple{M,SVector{dim,Tv}}(m.nodes[i] for i in element)
end

"""
Returns the affine map from the reference element to the given element.
"""
@inline function affine_map(m::Tris{Tv,Ti}, el::NTuple{3,Ti}) where {Tv,Ti}
    @inbounds p1, p2, p3 = get_nodes(m, el)
    return [p2 - p1 p3 - p1], p1
end

"""
Returns the offset of the affine map from the reference element to the given element.
"""
@inline function affine_map_shift(m::Tris{Tv,Ti}, el::NTuple{3,Ti}) where {Tv,Ti}
    @inbounds return m.nodes[el[1]]
end

@inline function affine_map(m::Tets{Tv,Ti}, el::NTuple{4,Ti}) where {Tv,Ti}
    @inbounds p1, p2, p3, p4 = m.nodes[el[1]], m.nodes[el[2]], m.nodes[el[3]], m.nodes[el[4]]
    return [p2 - p1 p3 - p1 p4 - p1], p1
end

@inline function affine_map_shift(m::Tets{Tv,Ti}, el::NTuple{4,Ti}) where {Tv,Ti}
    @inbounds return m.nodes[el[1]]
end

"""
List the faces of tets
"""
function list_faces(m::Mesh{3,4,Tv,Ti}) where {Tv,Ti}
    @assert all(issorted, m.elements)

    faces = Vector{NTuple{3,Ti}}(4 * nelements(m))
    idx = 1
    @inbounds for el in m.elements
        faces[idx + 0] = (el[1], el[2], el[3])
        faces[idx + 1] = (el[1], el[2], el[4])
        faces[idx + 2] = (el[1], el[3], el[4])
        faces[idx + 3] = (el[2], el[3], el[4])
        idx += 4
    end
    faces
end

"""
List the "faces" of triangles, i.e. edges.
"""
function list_faces(m::Mesh{2,3,Tv,Ti}) where {Tv,Ti}
    @assert all(issorted, m.elements)
    
    faces = Vector{NTuple{2,Ti}}(3 * nelements(m))
    idx = 1
    @inbounds for el in m.elements
        faces[idx + 0] = (el[1], el[2])
        faces[idx + 1] = (el[1], el[3])
        faces[idx + 2] = (el[2], el[3])
        idx += 3
    end
    faces
end

function list_interior_nodes(m::Mesh{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}

    # Get a vector [(n1,n2,n3),(n4,n5,n6),...]
    faces = list_faces(m)

    # Sort the faces
    radix_sort!(faces, nnodes(m))

    # Remove the interior faces
    remove_repeated_pairs!(faces)

    # Reinterpret at list of nodes [n1,n2,n3]
    nodes = reinterpret(Tuple{Ti},faces)

    # Sort them once more
    radix_sort!(nodes, nnodes(m))

    # Remove duplicates.
    remove_duplicates!(nodes)

    # Return the interior nodes
    complement(reinterpret(Ti, nodes), nnodes(m))
end
