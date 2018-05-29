import Base: getindex, @propagate_inbounds, ==, show

"""
Reference to global element index with local node / edge / face number.
"""
struct ElementId{Ti}
    element::Ti
    local_id::Ti
end

"""
Wraps a cell (set of nodes, either a node, an edge or a face) and metadata (element index & 
local index.)
"""
struct CellToEl{N,Ti}
    nodes::NTuple{N,Ti}
    data::ElementId{Ti}
end

# This is a bit of sugar.
const FaceToEl{Ti} = CellToEl{3,Ti}
const EdgeToEl{Ti} = CellToEl{2,Ti}
const NodeToEl{Ti} = CellToEl{1,Ti}

"""
Pretty much a sparse matrix csc. `cells` is a vector of cells (either a node, an edge or a
face). The range `offset[i] : offset[i+1]-1` is a list of indices of `values`, which contains
the corresponding element indices and local cell number. So `values[offset[2] : offset[3] - 1]`
would return the elements that `cells[2]` belongs to.
"""
struct SparseCellToElementMap{N,Ti}
    offset::Vector{Ti}
    cells::Vector{NTuple{N,Ti}}
    values::Vector{ElementId{Ti}}
end

"""
The nodes, edges and faces members are sparse mappings from a node, edge or face
on the interface to the corresponding element with the local index of the node,
edge or face. We also store `all_nodes`, which includes the nodes that do not
lie on an interface.
"""
struct Interfaces{Nn,Ne,Nf,Ti}
    all_nodes::SparseCellToElementMap{Nn,Ti}
    nodes::SparseCellToElementMap{Nn,Ti}
    edges::SparseCellToElementMap{Ne,Ti}
    faces::SparseCellToElementMap{Nf,Ti}
end

@propagate_inbounds getindex(f::CellToEl, i) = f.nodes[i]
@inline (==)(a::CellToEl, b::CellToEl) = a.nodes === b.nodes

function interfaces(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    # First order the element nodes.
    @assert all(issorted, mesh.elements)
    # sort_element_nodes!(mesh.elements)

    all_nodes, nodes = node_to_elements(mesh)
    edges = edge_to_elements(mesh)
    faces = face_to_elements(mesh)

    Interfaces(all_nodes, nodes, edges, faces)
end

"""
    node_to_elements(mesh) -> SparseCellToElementMap

Returns two sparse maps from face -> node with local face index. The first has all nodes,
the second only has nodes that are on the interface between elements.
"""
function node_to_elements(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    node_list = list_nodes_with_element(mesh)
    radix_sort!(node_list, nnodes(mesh), 1)
    all_nodes = copy(node_list)    
    remove_singletons!(node_list)
    return compress(all_nodes), compress(node_list)
end

"""
    edge_to_elements(mesh) -> SparseCellToElementMap

Returns a sparse map from edge -> element with local edge index, where the edges lie on the
interface between elements.
"""
function edge_to_elements(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    edge_list = list_edges_with_element(mesh)
    radix_sort!(edge_list, nnodes(mesh), 2)
    remove_singletons!(edge_list)
    return compress(edge_list)
end

"""
    face_to_elements(mesh) -> SparseCellToElementMap

Returns a sparse map from face -> element with local face index, where the faces lie on the
interface between elements.
"""
function face_to_elements(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    face_list = list_faces_with_element(mesh)
    radix_sort!(face_list, nnodes(mesh), 3)
    remove_singletons!(face_list)
    return compress(face_list)
end

"""
    list_faces_with_element(mesh) -> Vector{CellToEl}

Make a list of all faces with their corresponding element index and their local face number.
"""
function list_faces_with_element(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    face_list = Vector{FaceToEl{Ti}}(4 * nelements(mesh))
    idx = 1
    @inbounds for (el_idx, el) in enumerate(mesh.elements)
        face_list[idx + 0] = FaceToEl{Ti}((el[1], el[2], el[3]), ElementId(el_idx, 1))
        face_list[idx + 1] = FaceToEl{Ti}((el[1], el[2], el[4]), ElementId(el_idx, 2))
        face_list[idx + 2] = FaceToEl{Ti}((el[1], el[3], el[4]), ElementId(el_idx, 3))
        face_list[idx + 3] = FaceToEl{Ti}((el[2], el[3], el[4]), ElementId(el_idx, 4))
        idx += 4
    end
    return face_list
end

"""
    list_edges_with_element(mesh) -> Vector{CellToEl}

Make a list of all edges with their corresponding element index and their local edge number.
"""
function list_edges_with_element(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    edge_list = Vector{EdgeToEl{Ti}}(6 * nelements(mesh))
    idx = 1
    @inbounds for (el_idx, el) in enumerate(mesh.elements)
        edge_list[idx + 0] = EdgeToEl{Ti}((el[1], el[2]), ElementId(el_idx, 1))
        edge_list[idx + 1] = EdgeToEl{Ti}((el[1], el[3]), ElementId(el_idx, 2))
        edge_list[idx + 2] = EdgeToEl{Ti}((el[1], el[4]), ElementId(el_idx, 3))
        edge_list[idx + 3] = EdgeToEl{Ti}((el[2], el[3]), ElementId(el_idx, 4))
        edge_list[idx + 4] = EdgeToEl{Ti}((el[2], el[4]), ElementId(el_idx, 5))
        edge_list[idx + 5] = EdgeToEl{Ti}((el[3], el[4]), ElementId(el_idx, 6))
        idx += 6
    end
    return edge_list
end

"""
    list_nodes_with_element(mesh) -> Vector{CellToEl}

Make a list of all nodes with their corresponding element index and their local node number.
"""
function list_nodes_with_element(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    node_list = Vector{NodeToEl{Ti}}(4 * nelements(mesh))
    idx = 1
    @inbounds for (el_idx, el) in enumerate(mesh.elements)
        node_list[idx + 0] = NodeToEl{Ti}((el[1],), ElementId(el_idx, 1))
        node_list[idx + 1] = NodeToEl{Ti}((el[2],), ElementId(el_idx, 2))
        node_list[idx + 2] = NodeToEl{Ti}((el[3],), ElementId(el_idx, 3))
        node_list[idx + 3] = NodeToEl{Ti}((el[4],), ElementId(el_idx, 4))
        idx += 4
    end
    return node_list
end

"""
    list_boundary_nodes_edges_faces(m::Tets) -> NTuple{3,SparseCellToElementMap}

For a given input mesh we return sparse mappings from face, cell and node on the
boundary to the element with the local face, cell and node number.
"""
function list_boundary_nodes_edges_faces(m::Tets{Tv,Ti}) where {Tv,Ti}

    faces = list_faces_with_element(m)

    # Sort the faces
    radix_sort!(faces, nnodes(m), 3)

    # Remove the interior faces
    remove_repeated_pairs!(faces)

    # Convert to sorted list of boundary edges
    boundary_edges = Vector{Tuple{Ti,Ti}}(3 * length(faces))

    idx = 1
    @inbounds for face in faces
        boundary_edges[idx + 0] = (face.nodes[1], face.nodes[2])
        boundary_edges[idx + 1] = (face.nodes[1], face.nodes[3])
        boundary_edges[idx + 2] = (face.nodes[2], face.nodes[3])
        idx += 3
    end

    radix_sort!(boundary_edges, nnodes(m), 2)
    remove_duplicates!(boundary_edges)

    # List all edges
    edges = list_edges_with_element(m)
    radix_sort!(edges, nnodes(m), 2)

    # Retain only those values of `edges` that occur in `boundary_edges`
    intersect!(edges, boundary_edges)

    # Convert to sorted list of boundary nodes
    boundary_nodes = Vector{Tuple{Ti}}(2 * length(boundary_edges))
    idx = 1
    @inbounds for edge in boundary_edges
        boundary_nodes[idx + 0] = (edge[1],)
        boundary_nodes[idx + 1] = (edge[2],)
        idx += 2
    end

    radix_sort!(boundary_nodes, nnodes(m), 1)
    remove_duplicates!(boundary_nodes)

    # List all nodes
    nodes = list_nodes_with_element(m)
    radix_sort!(nodes, nnodes(m), 1)
    intersect!(nodes, boundary_nodes)

    return compress(nodes), compress(edges), compress(faces)
end

"""
Remove all items from `v` that do not occur in `w`, assuming `v` and `w` are
sorted.
# todo, make generic / clean things.
"""
function intersect!(v::Vector{CellToEl{N,Ti}}, w::Vector{NTuple{N,Ti}}, by = x -> x.nodes) where {N,Ti}
    slow = 1
    fast = 1
    idx = 1

    @inbounds while fast ≤ length(v) && idx ≤ length(w)
        if by(v[fast]) < w[idx]
            fast += 1
        elseif by(v[fast]) === w[idx]
            v[slow] = v[fast]
            slow += 1
            fast += 1
        else
            idx += 1
        end
    end

    resize!(v, slow - 1)
end

"""
    compress(v::Vector{CellToEl{N,Ti}}) -> SparseCellToElementMap

Compress a mapping from nodes, edges or faces to elements similar as what 
`sparse` does for a set of triplets.
"""
function compress(v::Vector{CellToEl{N,Ti}}) where {N,Ti}
    # Count unique guys.
    unique = 1
    @inbounds for i = 2 : length(v)
        if v[i - 1] != v[i]
            unique += 1
        end
    end

    offset = Vector{Ti}(unique + 1)
    values = Vector{ElementId{Ti}}(length(v))
    cells = Vector{NTuple{N,Ti}}(unique)

    @inbounds begin
        offset[1] = 1
        values[1] = v[1].data
        cells[1] = v[1].nodes
        idx = 1

        for i = 2 : length(v)
            values[i] = v[i].data

            # If we find a new cell, update the bookkeeping of offset pointers, 
            # and update the value.
            if v[i] != v[i - 1]
                cells[idx += 1] = v[i].nodes
                offset[idx] = i
            end
        end

        offset[end] = length(v) + 1
    end

    return SparseCellToElementMap(offset, cells, values)
end