using Rewrite: refined_element, build_local_operators, Tets, Tris, Mesh, Tris64, Tets64, affine_map, 
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, interfaces, face_to_elements, edge_to_elements,
               nelements, nnodes, nodes_per_face_interior, nodes_per_edge_interior, 
               get_reference_normals, Tet, node_to_elements, ImplicitFineGrid,
               construct_full_grid, ZeroDirichletConstraint, list_boundary_faces,
               refined_mesh, apply_dirichlet_constraint!, cell_type, default_quad, ElementValues,
               update_det_J, update_inv_J, reinit!, get_inv_jac, get_detjac
using StaticArrays, WriteVTK

function local_operator(total_levels = 2, inspect_level = 2)
    # Unit cube
    nodes = SVector{3,Float64}[
        (0,0,0),
        (1,0,0),
        (0,1,0),
        (1,1,0),
        (0,0,1),
        (1,0,1),
        (0,1,1),
        (1,1,1)
    ]

    map!(x -> x .+ randn(3)/10, nodes, nodes)
    
    # Split in tetrahedra
    elements = [
        (1,2,3,5),
        (2,3,4,8),
        (3,5,7,8),
        (2,5,6,8),
        (2,3,5,8)
    ]

    base = Mesh(nodes, elements)

    # Build a multilevel grid
    implicit = ImplicitFineGrid(base, total_levels)

    # Operators for each level
    ops = build_local_operators(implicit.reference)

    # Select the finest level ops.
    ∫ϕₓᵢϕₓⱼ_finest = ops[inspect_level]

    # Verify things on this grid
    finest = refined_mesh(implicit, inspect_level)
    
    # Find the number of nodes per element at level `inspect_level`
    nodes_per_element = nnodes(refined_mesh(implicit, inspect_level))

    x_base = map(x -> 1 - sum(x), base.nodes)

    x_distributed = zeros(4, nelements(base))
    
    for j = 1 : nelements(base), i = 1 : 4
        x_distributed[i, j] = x_base[base.elements[j][i]]
    end

    for P in implicit.reference.interops
        x_distributed = P * x_distributed
        @show size(P) size(x_distributed)
    end

    # Output matrix is y.
    # x_distributed = rand(nodes_per_element, nelements(base))
    y_distributed = zeros(nodes_per_element, nelements(base))

    # Apply the operator on each base element
    cell = cell_type(base)
    quadrature = default_quad(cell)
    element_values = ElementValues(cell, quadrature, update_det_J | update_inv_J)
    
    # Do the mv product.
    for (el_idx, element) in enumerate(base.elements)
        reinit!(element_values, base, element)

        Jinv = get_inv_jac(element_values)
        detJ = get_detjac(element_values)

        P = Jinv' * Jinv

        x_local = view(x_distributed, :, el_idx)
        y_local = view(y_distributed, :, el_idx)

        # Apply the op finally.
        for i = 1 : 3, j = 1 : 3
            A_mul_B!(P[i, j] * detJ, ∫ϕₓᵢϕₓⱼ_finest.ops[i, j], x_local, 1.0, y_local)
        end
    end

    # Distribute the faces.
    local_numbering = implicit.reference.numbering[inspect_level]
    nodes_per_face = nodes_per_face_interior(implicit.reference, inspect_level)

    let buffer = zeros(nodes_per_face)
        face_to_element = implicit.interfaces.faces
        for (i, face) in enumerate(face_to_element.cells)
            fill!(buffer, 0.0)

            # Reduce
            # Loop over the two connected elements
            for j = face_to_element.offset[i] : face_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = face_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.faces_interior[element_data.local_id]

                # Add the values to the buffer
                for k = 1 : nodes_per_face
                    buffer[k] += y_distributed[nodes[k], element_data.element]
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
                    y_distributed[nodes[k], element_data.element] = buffer[k]
                end
            end
        end
    end

    nodes_per_edge = nodes_per_edge_interior(implicit.reference, inspect_level)

    let buffer = zeros(nodes_per_edge)
        edge_to_element = implicit.interfaces.edges
        for (i, edge) in enumerate(edge_to_element.cells)
            fill!(buffer, 0.0)

            # Reduce
            # Loop over the two connected elements
            for j = edge_to_element.offset[i] : edge_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = edge_to_element.values[j]

                # Find the local numbering
                nodes = local_numbering.edges_interior[element_data.local_id]

                # Add the values to the buffer
                for k = 1 : nodes_per_edge
                    buffer[k] += y_distributed[nodes[k], element_data.element]
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
                    y_distributed[nodes[k], element_data.element] = buffer[k]
                end
            end
        end
    end

    let buffer = 0.0
        node_to_element = implicit.interfaces.nodes
        for (i, node) in enumerate(node_to_element.cells)
            buffer = 0.0

            # Reduce
            # Loop over the two connected elements
            for j = node_to_element.offset[i] : node_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = node_to_element.values[j]

                # Find the local numbering
                local_node = local_numbering.nodes[element_data.local_id]

                # Add the values to the buffer
                buffer += y_distributed[local_node, element_data.element]
            end

            # Broadcast
            for j = node_to_element.offset[i] : node_to_element.offset[i + 1] - 1

                # Get the global element id
                element_data = node_to_element.values[j]

                # Find the local numbering
                local_node = local_numbering.nodes[element_data.local_id]

                # Overwrite with the sum.
                y_distributed[local_node, element_data.element] = buffer
            end
        end
    end

    # Construct the full grid (expensive if `inspect_level` is large)
    fine_mesh = construct_full_grid(implicit, inspect_level)

    # Save the full grid
    vtk = vtk_grid("multiplication", fine_mesh) do vtk
        vtk_point_data(vtk, reshape(x_distributed, :), "x")
        vtk_point_data(vtk, reshape(y_distributed, :), "A * u")
    end
    
end