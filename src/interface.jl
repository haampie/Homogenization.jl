import Base: getindex, @propagate_inbounds, ==, show

"""
Reference to global element id with local node / edge / face number.
"""
struct ElementId{Ti}
    element::Ti
    local_id::Ti
end

struct CellToEl{N,Ti}
    nodes::NTuple{N,Ti}
    data::ElementId{Ti}
end

const FaceToEl{Ti} = CellToEl{3,Ti}
const EdgeToEl{Ti} = CellToEl{2,Ti}
const NodeToEl{Ti} = CellToEl{1,Ti}

"""
Pretty much a sparse matrix csc.
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

Returns two sparse maps from face -> node with local face id's. The first has all nodes,
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

Returns a sparse map from edge -> element with local edge id's, where the edges lie on the
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

Returns a sparse map from face -> element with local face id's, where the faces lie on the
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

Make a list of all faces with their corresponding element id and their local
face number.
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

Make a list of all edges with their corresponding element id and their local
edge number.
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

Make a list of all nodes with their corresponding element id and their local
node number.
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
    compress(v::Vector{CellToEll{N,Ti}}) -> SparseCellToElementMap

Compress a mapping from nodes, edges or faces to elements similar as what `sparse` does for
a set of triplets.
"""
function compress(v::Vector{CellToEl{N,Ti}}) where {N,Ti}
    # Count unique guys.
    unique = 1
    @inbounds for i = 1 : length(v) - 1
        if v[i] != v[i + 1]
            unique += 1
        end
    end

    @inbounds if v[end] != v[end-1]
        unique += 1
    end

    offset = Vector{Ti}(unique + 1)
    values = Vector{ElementId{Ti}}(length(v))
    cells = Vector{NTuple{N,Ti}}(unique)

    @inbounds begin
        offset[1] = 1
        values[1] = v[1].data
        cells[1] = v[1].nodes
        offset[end] = length(v) + 1
        offset_idx = 1

        for i = 2 : length(v)
            values[i] = v[i].data
            if v[i] != v[i - 1]
                offset_idx += 1
                offset[offset_idx] = i
                cells[offset_idx] = v[i].nodes
            end
        end
    end

    return SparseCellToElementMap(offset, cells, values)
end