import Base: getindex, @propagate_inbounds, ==, !=, show

"""
Reference to global element id with local node / edge/ face number.
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

struct Interfaces{Nn,Ne,Nf,Ti}
    nodes::SparseCellToElementMap{Nn,Ti}
    edges::SparseCellToElementMap{Ne,Ti}
    faces::SparseCellToElementMap{Nf,Ti}
end

@propagate_inbounds getindex(f::CellToEl, i) = f.nodes[i]
@inline (==)(a::CellToEl, b::CellToEl) = a.nodes == b.nodes

function interfaces(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    # First order the element nodes.
    @assert all(issorted, mesh.elements)
    # sort_element_nodes!(mesh.elements)

    Interfaces(
        node_to_elements(mesh),
        edge_to_elements(mesh),
        face_to_elements(mesh)
    )
end

function node_to_elements(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    total_nodes = length(mesh.nodes)
    node_list = Vector{NodeToEl{Ti}}(4 * length(mesh.elements))

    idx = 1
    @inbounds for (el_idx, el) in enumerate(mesh.elements)
        node_list[idx + 0] = NodeToEl{Ti}((el[1],), ElementId(el_idx, 1))
        node_list[idx + 1] = NodeToEl{Ti}((el[2],), ElementId(el_idx, 2))
        node_list[idx + 2] = NodeToEl{Ti}((el[3],), ElementId(el_idx, 3))
        node_list[idx + 3] = NodeToEl{Ti}((el[4],), ElementId(el_idx, 4))
        idx += 4
    end

    radix_sort!(node_list, total_nodes, 1)
    remove_singletons!(node_list)

    return compress(node_list)
end

function edge_to_elements(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    total_nodes = length(mesh.nodes)
    edge_list = Vector{EdgeToEl{Ti}}(6 * length(mesh.elements))

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

    radix_sort!(edge_list, total_nodes, 2)
    remove_singletons!(edge_list)

    return compress(edge_list)
end

function face_to_elements(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    total_nodes = length(mesh.nodes)
    face_list = list_faces_with_element(mesh)
    radix_sort!(face_list, total_nodes, 3)
    remove_singletons!(face_list)
    return compress(face_list)
end

"""
    list_faces_with_element(mesh) -> Vector{CellToEl}

Make a list of all faces with their corresponding element id and their local
face number.
"""
function list_faces_with_element(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    face_list = Vector{FaceToEl{Ti}}(4 * length(mesh.elements))
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