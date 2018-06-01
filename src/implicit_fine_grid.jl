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
    @assert all(issorted, base.elements)
    reference = refined_element(levels, typeof(base))
    inter = interfaces(base)
    ImplicitFineGrid(levels, reference, inter, base)
end

nlevels(g::ImplicitFineGrid) = g.levels
base_mesh(g::ImplicitFineGrid) = g.base
@propagate_inbounds refined_mesh(g::ImplicitFineGrid, level::Int) = g.reference.levels[level]
@propagate_inbounds local_numbering(g::ImplicitFineGrid, level::Int) = g.reference.numbering[level]
cell_type(g::ImplicitFineGrid) = cell_type(g.base)

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

struct ZeroDirichletConstraint{Ti,Nn,Ne,Nf}
    nodes::SparseCellToElementMap{Nn,Ti}
    edges::SparseCellToElementMap{Ne,Ti}
    faces::SparseCellToElementMap{Nf,Ti}
end

"""
    apply_constraint!(x, level, ::ZeroDirichletConstraint, ::ImplicitFineGrid)

Apply zero Dirichlet conditions to the nodes on the boundary of implicitly refined
vector `u`. ZeroDirichletConstraint contains the faces, edges and nodes of the 
base mesh, and via the ImplicitFineGrid we get the local numbering of the faces,
edges and nodes to zero them out.
"""
function apply_constraint!(x::Matrix{Tv}, level::Int, z::ZeroDirichletConstraint{Ti}, implicit::ImplicitFineGrid{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    numbering = local_numbering(implicit, level)

    # FACES
    @inbounds for i = 1 : length(z.faces.cells), j = valrange(z.faces, i)

        # Get the global element id
        value = z.faces.values[j]

        # Find the local numbering
        nodes = numbering.faces_interior[value.local_id]

        # Add the values to the buffer
        for node in nodes
            x[node, value.element] = zero(Tv)
        end
    end

    # EDGES
    @inbounds for i = 1 : length(z.edges.cells), j = valrange(z.edges, i)

        # Get the global element id
        value = z.edges.values[j]

        # Add the values to the buffer
        for node in numbering.edges_interior[value.local_id]
            x[node, value.element] = zero(Tv)
        end
    end

    # NODES
    @inbounds for i = 1 : length(z.nodes.cells), j = valrange(z.nodes, i)
        # Get the global element id
        value = z.nodes.values[j]

        # Find the local numbering
        node = numbering.nodes[value.local_id]

        # Add the values to the buffer
        x[node, value.element] = zero(Tv)
    end

    x
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
        value = node_to_element.values[j]

        # Find the local numbering
        local_node = numbering.nodes[value.local_id]

        # Element id
        element_idx = value.element

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
        u_val = u[global_idx]

        # Loop over all the the elements connected to this node.
        for j = valrange(node_to_element, i)
            # Get the global element data
            value = node_to_element.values[j]

            # Find the local numbering
            local_node = numbering.nodes[value.local_id]

            # Element id
            element_idx = value.element

            # Copy the value over.
            v[local_node, element_idx] = u_val
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
        @inbounds for i = 1 : length(face_to_element.cells)
            fill!(buffer, zero(Tv))

            # Reduce
            for j = valrange(face_to_element, i)

                # Get the global element id
                value = face_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.faces_interior[value.local_id]

                # Add the values to the buffer
                for k = 1 : nodes_per_face
                    buffer[k] += x[nodes[k], value.element]
                end
            end

            # Broadcast
            for j = valrange(face_to_element, i)

                # Get the global element id
                value = face_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.faces_interior[value.local_id]

                # Overwrite with the sum.
                for k = 1 : nodes_per_face
                    x[nodes[k], value.element] = buffer[k]
                end
            end
        end
    end

    # EDGES
    nodes_per_edge = nodes_per_edge_interior(implicit.reference, level)

    let buffer = zeros(Tv, nodes_per_edge)
        edge_to_element = implicit.interfaces.edges
        @inbounds for i = 1 : length(edge_to_element.cells)
            fill!(buffer, zero(Tv))

            # Reduce
            for j = valrange(edge_to_element, i)

                # Get the global element id
                value = edge_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.edges_interior[value.local_id]

                # Add the values to the buffer
                for k = 1 : nodes_per_edge
                    buffer[k] += x[nodes[k], value.element]
                end
            end

            # Broadcast
            for j = valrange(edge_to_element, i)

                # Get the global element id
                value = edge_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.edges_interior[value.local_id]

                # Overwrite with the sum.
                for k = 1 : nodes_per_edge
                    x[nodes[k], value.element] = buffer[k]
                end
            end
        end
    end

    # NODES
    node_to_element = implicit.interfaces.nodes
    for i = 1 : length(node_to_element.cells)
        buffer = zero(Tv)

        # Reduce
        @inbounds for j = valrange(node_to_element, i)

            # Get the global element id
            value = node_to_element.values[j]

            # Find the local numbering
            local_node = local_numbering.nodes[value.local_id]

            # Add the values to the buffer
            buffer += x[local_node, value.element]
        end

        # Broadcast
        for j = valrange(node_to_element, i)

            # Get the global element id
            value = node_to_element.values[j]

            # Find the local numbering
            local_node = local_numbering.nodes[value.local_id]

            # Overwrite with the sum.
            x[local_node, value.element] = buffer
        end
    end

    x
end

"""
If a node is shared among multiple multiple coarse elements, we will zero out
all of them except in the first listed element.
"""
function zero_out_all_but_one!(x::AbstractMatrix{Tv}, implicit::ImplicitFineGrid, level::Int) where {Tv}

    local_numbering = implicit.reference.numbering[level]

    # FACES
    nodes_per_face = nodes_per_face_interior(implicit.reference, level)
    face_to_element = implicit.interfaces.faces
    @inbounds for i = 1 : length(face_to_element.cells), j = face_to_element.offset[i] + 1 : face_to_element.offset[i + 1] - 1
        # Get the global element id
        value = face_to_element.values[j]

        # Find the local numbering
        nodes = local_numbering.faces_interior[value.local_id]

        # Add the values to the buffer
        for k = 1 : nodes_per_face
            x[nodes[k], value.element] = zero(Tv)
        end
    end

    # EDGES
    nodes_per_edge = nodes_per_edge_interior(implicit.reference, level)
    edge_to_element = implicit.interfaces.edges
    @inbounds for i = 1 : length(edge_to_element.cells), j = edge_to_element.offset[i] + 1 : edge_to_element.offset[i + 1] - 1
        # Get the global element id
        value = edge_to_element.values[j]

        # Find the local numbering
        nodes = local_numbering.edges_interior[value.local_id]

        # Add the values to the buffer
        for k = 1 : nodes_per_edge
            x[nodes[k], value.element] = zero(Tv)
        end
    end

    # NODES
    node_to_element = implicit.interfaces.nodes
    @inbounds for i = 1 : length(node_to_element.cells), j = node_to_element.offset[i] + 1 : node_to_element.offset[i + 1] - 1
        # Get the global element id
        value = node_to_element.values[j]

        # Find the local numbering
        local_node = local_numbering.nodes[value.local_id]

        # Add the values to the buffer
        x[local_node, value.element] = zero(Tv)
    end
    x
end

"""
Build a local rhs with functional ∫v
"""
function local_rhs!(b::AbstractMatrix, implicit::ImplicitFineGrid{dim,N,Tv,Ti}) where {dim,N,Tv,Ti}
    base = base_mesh(implicit)
    fine = refined_mesh(implicit, nlevels(implicit))
    
    @assert size(b) == (nnodes(fine), nelements(base))

    # Construct b on the reference element
    b_ref = assemble_vector(fine, identity)

    cell = cell_type(base)
    quadrature = default_quad(cell)
    weights = get_weights(quadrature)
    element_values = ElementValues(cell, quadrature, update_det_J)

    @inbounds for (idx, element) in enumerate(base.elements)
        reinit!(element_values, base, element)
        b[:, idx] .= b_ref .* get_det_jac(element_values)
    end
end



"""
    full_grid_without_duplicates(g::ImplicitFineGrid, level::Int) -> Mesh

Builds the full mesh at a certain level with elements on the interfaces occurring
just once. Be very scared, cause the number of nodes gets large! There are still
duplicate nodes though, but we don't care.
"""
function full_grid_without_duplicates(g::ImplicitFineGrid{dim,N,Tv,Ti}, level::Int) where {dim,N,Tv,Ti}
    base = base_mesh(g)
    ref_mesh = refined_mesh(g, level)

    # Since we copy nodes on the interface, we have #coarse * #ref nodes & elements
    total_nodes = nelements(base) * nnodes(ref_mesh)
    total_elements = nelements(base) * nelements(ref_mesh)

    nodes = Vector{SVector{dim,Tv}}(total_nodes)
    elements = Matrix{NTuple{N,Ti}}(total_elements)

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

    # Fix the repeated nodes, edges & faces.
    local_numbering = implicit.reference.numbering[level]

    # FACES
    f2e = implicit.interfaces.faces

    @inbounds for i = 1 : length(f2e.cells)

        # Get the first face.
        value = f2e.values[f2e.offset[i]]

        # Get the element indices of this face
        face_idxs = face_indices(cell_type(g), value.local_id)

        # Get the node numbers that make up this face.
        face_node_numbers = get_node_numbers(ref_mesh, value.element, face_idxs)
        
        # Overwrite the other faces with these node numbers
        for j = f2e.offset[i] + 1 : f2e.offset[i + 1] - 1
            # Get the global element id
            value_ = f2e.values[j]

            # Find the local numbering
            face_idxs_ = face_indices(cell_type(g), value.local_id)

            # Current element id.
            el_ = value.element

            offset_ = (nnodes(ref_mesh) - 1) * el_ + el_

            # Overwrite the face
            for k in face_idxs_
                elements[]
            end

        end
    end

    return Mesh(nodes, elements)
end