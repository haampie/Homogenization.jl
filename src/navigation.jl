import Base: getindex, @propagate_inbounds, ==

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

@propagate_inbounds getindex(f::CellToEl, i) = f.nodes[i]
@inline (==)(a::CellToEl, b::CellToEl) = a.nodes == b.nodes

function navigation(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    # First order the element nodes.
    sort_element_nodes!(mesh.elements)

    node_to_elements(mesh), edge_to_elements(mesh), face_to_elements(mesh)
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

    node_list
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

    edge_list
end

function face_to_elements(mesh::Tets{Tv,Ti}) where {Tv,Ti}
    total_nodes = length(mesh.nodes)
    face_list = Vector{FaceToEl{Ti}}(4 * length(mesh.elements))
    idx = 1
    @inbounds for (el_idx, el) in enumerate(mesh.elements)
        face_list[idx + 0] = FaceToEl{Ti}((el[1], el[2], el[3]), ElementId(el_idx, 1))
        face_list[idx + 1] = FaceToEl{Ti}((el[1], el[2], el[4]), ElementId(el_idx, 2))
        face_list[idx + 2] = FaceToEl{Ti}((el[1], el[3], el[4]), ElementId(el_idx, 3))
        face_list[idx + 3] = FaceToEl{Ti}((el[2], el[3], el[4]), ElementId(el_idx, 4))
        idx += 4
    end

    radix_sort!(face_list, total_nodes, 3)
    remove_singletons!(face_list)

    face_list
end

"""
Remove non-repeated elements from an array
"""
function remove_singletons!(v::Vector)
    length(v) == 0 && return v

    count = 0
    slow, fast = 1, 1

    @inbounds while true
        value = v[fast]
        count = 0

        # Copy them over while equal
        while fast ≤ length(v) && v[fast] == value
            v[slow] = v[fast]
            slow += 1
            fast += 1
            count += 1
        end

        # If it occurs only once, we may overwrite it
        if count == 1
            slow -= 1
        end

        if fast > length(v)
            break
        end
    end

    resize!(v, slow - 1)
end