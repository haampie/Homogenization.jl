import Base: @propagate_inbounds, show
"""
ImplicitFineGrid holds the base mesh, the refinements of the reference element and the
interfaces on the base mesh.
"""
struct ImplicitFineGrid{dim,N,Tv,Ti,Nn,Ne,Nf}
    levels::Int
    reference::MultilevelReference{dim,N,Tv,Ti}
    interfaces::Interfaces{Nn,Ne,Nf,Ti}
    base::Mesh{dim,N,Tv,Ti}
end

function ImplicitFineGrid(base::Mesh{dim,N,Tv,Ti}, levels::Int) where {dim,N,Tv,Ti}
    reference = refined_element(levels, typeof(base))
    inter = interfaces(base)
    ImplicitFineGrid(levels, reference, inter, base)
end

nlevels(g::ImplicitFineGrid) = g.levels
base_mesh(g::ImplicitFineGrid) = g.base
@propagate_inbounds refined_mesh(g::ImplicitFineGrid, level::Int) = g.reference.levels[level]
@propagate_inbounds local_numbering(g::ImplicitFineGrid, level::Int) = g.reference.numbering[level]

function show(io::IO, g::ImplicitFineGrid)
    base = base_mesh(g)
    finest = refined_mesh(g, nlevels(g))
    print(io, "Implicit grid of cell type ", cell_type(base),
              ". Base mesh has ", nnodes(base), " nodes and ", nelements(base), " elements.",
              " Finest level (", nlevels(g), ") has ", nnodes(finest), " nodes and ", nelements(finest), " elements.",
              " In total at most ", nnodes(finest) * nelements(base), " unknowns.")
end

"""
    construct_full_grid(g::ImplicitFineGrid, level::Int) -> Mesh

Builds the full mesh at a certain level with nodes on the interface repeated.
Be very scared, cause the number of nodes gets large!
"""
function construct_full_grid(g::ImplicitFineGrid{dim,N,Tv,Ti}, level::Int) where {dim,N,Tv,Ti}
    base = base_mesh(g)
    ref_mesh = refined_mesh(g, level)

    # Since we copy nodes on the interface, we have #coarse * #ref nodes & elements
    total_nodes = nelements(base) * nnodes(ref_mesh)
    total_elements = nelements(base) * nelements(ref_mesh)

    nodes = Vector{SVector{dim,Tv}}(total_nodes)
    elements = Vector{NTuple{N,Ti}}(total_elements)

    # Now for each base element we simply apply the coordinate transform to each
    # node, and we copy over each fine element. We only have to renumber the
    # fine elements by the offset of the base element number.

    node_idx = 0
    element_idx = 0
    offset = 0

    @inbounds for element in base.elements
        # Get the coordinate mapping
        J, b = affine_map(base, element)
        
        # Copy the transformed nodes over
        for node in ref_mesh.nodes
            nodes[node_idx += 1] = J * node + b
        end

        # Copy over the elements
        for element in ref_mesh.elements
            elements[element_idx += 1] = element .+ offset
        end

        offset += nnodes(ref_mesh)
    end

    return Mesh(nodes, elements)
end

struct ZeroDirichletConstraint{Ti}
    list_of_faces::Vector{ElementId{Ti}}
end

"""
    apply_constraint!(u, level, ::ZeroDirichletConstraint, ::ImplicitFineGrid)

Apply zero Dirichlet conditions to the nodes on the boundary of implicitly refined
vector `u`. ZeroDirichletConstraint contains the faces of the base mesh, and via
the ImplicitFineGrid we get the local numbering of the faces to zero them out.
"""
function apply_constraint!(u::Matrix{Tv}, level::Int, z::ZeroDirichletConstraint{Ti}, g::ImplicitFineGrid{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    
    # Maybe get rid of these assertions one day.
    @assert size(u, 1) == nnodes(refined_mesh(g, level))
    @assert size(u, 2) == nelements(base_mesh(g))

    numbering = local_numbering(g, level)

    @inbounds for faceInfo in z.list_of_faces
        # Extract the base element id
        element = faceInfo.element

        # And the local face number.
        face = faceInfo.local_id

        # Zero out each node on the boundary.
        for node = numbering.faces[face]
            u[node, element] = zero(Tv)
        end
    end

    u
end

"""
    copy_to_base!(u, v, ::ImplicitFineGrid)

Copy the unknowns of the coarse grid `v` to a local vector `u`. Here we assume
that `v` is a matrix of size `# nnodes(coarse ref element) × nelements(base)`,
and `u` is a vector of size `# nnodes(base)`.
"""
function copy_to_base!(u::Vector{Tv}, v::Matrix{Tv}, implicit::ImplicitFineGrid{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    numbering = local_numbering(implicit, 1)
    node_to_element = implicit.interfaces.all_nodes

    # Loop over all nodes and copy only the first occurence to the base vector.
    @inbounds for (i, node) = enumerate(node_to_element.cells)
        global_idx = node[1]

        # First index of this node
        j = node_to_element.offset[i]

        # Get the global element id
        element_data = node_to_element.values[j]

        # Find the local numbering
        local_node = numbering.nodes[element_data.local_id]

        # Element id
        element_idx = element_data.element

        # Copy the value over.
        u[global_idx] = v[local_node, element_idx]
    end
end

"""
    distribute!(v, u, ::ImplicitFineGrid)

Copy the local unknown `u` over to the global distributed version `v`.
"""
function distribute!(v::Matrix{Tv}, u::Vector{Tv}, implicit::ImplicitFineGrid{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    numbering = local_numbering(implicit, 1)
    node_to_element = implicit.interfaces.all_nodes

    # Loop over all nodes and copy each value to all distributed elements
    @inbounds for (i, node) = enumerate(node_to_element.cells)
        global_idx, = node
        value = u[global_idx]

        # Loop over all the the elements connected to this node.
        for j = node_to_element.offset[i] : node_to_element.offset[i + 1] - 1
            # Get the global element data
            element_data = node_to_element.values[j]

            # Find the local numbering
            local_node = numbering.nodes[element_data.local_id]

            # Element id
            element_idx = element_data.element

            # Copy the value over.
            v[local_node, element_idx] = value
        end
    end
end

"""
    broadcast_interfaces!(x, ::ImplicitFineGrid, level::Int)

Sums the values of `x` along the boundary and updates it locally.
"""
function broadcast_interfaces!(x::AbstractMatrix{Tv}, implicit::ImplicitFineGrid, level::Int) where {Tv}

    local_numbering = implicit.reference.numbering[level]

    # FACES
    nodes_per_face = nodes_per_face_interior(implicit.reference, level)

    let buffer = zeros(Tv, nodes_per_face)
        face_to_element = implicit.interfaces.faces
        for i = 1 : length(face_to_element.cells)
            fill!(buffer, zero(Tv))

            # Reduce
            # Loop over the two connected elements
            for j = face_to_element.offset[i] : face_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = face_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.faces_interior[element_data.local_id]

                # Add the values to the buffer
                for k = 1 : nodes_per_face
                    buffer[k] += x[nodes[k], element_data.element]
                end
            end

            # Broadcast
            for j = face_to_element.offset[i] : face_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = face_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.faces_interior[element_data.local_id]

                # Overwrite with the sum.
                for k = 1 : nodes_per_face
                    x[nodes[k], element_data.element] = buffer[k]
                end
            end
        end
    end

    # EDGES
    nodes_per_edge = nodes_per_edge_interior(implicit.reference, level)

    let buffer = zeros(Tv, nodes_per_edge)
        edge_to_element = implicit.interfaces.edges
        for i = 1 : length(edge_to_element.cells)
            fill!(buffer, zero(Tv))

            # Reduce
            # Loop over the two connected elements
            for j = edge_to_element.offset[i] : edge_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = edge_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.edges_interior[element_data.local_id]

                # Add the values to the buffer
                for k = 1 : nodes_per_edge
                    buffer[k] += x[nodes[k], element_data.element]
                end
            end

            # Broadcast
            for j = edge_to_element.offset[i] : edge_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = edge_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.edges_interior[element_data.local_id]

                # Overwrite with the sum.
                for k = 1 : nodes_per_edge
                    x[nodes[k], element_data.element] = buffer[k]
                end
            end
        end
    end

    # NODES
    let buffer = zero(Tv)
        node_to_element = implicit.interfaces.nodes
        for i = 1 : length(node_to_element.cells)
            buffer = zero(Tv)

            # Reduce
            # Loop over the two connected elements
            for j = node_to_element.offset[i] : node_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = node_to_element.values[j]

                # Find the local numbering
                local_node = local_numbering.nodes[element_data.local_id]

                # Add the values to the buffer
                buffer += x[local_node, element_data.element]
            end

            # Broadcast
            for j = node_to_element.offset[i] : node_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = node_to_element.values[j]

                # Find the local numbering
                local_node = local_numbering.nodes[element_data.local_id]

                # Overwrite with the sum.
                x[local_node, element_data.element] = buffer
            end
        end
    end

    x
end