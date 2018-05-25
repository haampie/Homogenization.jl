using Rewrite: refined_element, build_local_operators, Tets, Tris, Mesh, Tris64, Tets64, affine_map, 
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, navigation, face_to_elements, edge_to_elements,
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

    # We generate random numbers for each node on the base mesh and the linearly
    # interpolate this to the fine grid, then apply our ∇u ⋅ ∇v operator.
    x_base = rand(nnodes(base))

    x_distributed = zeros(4, nelements(base))
    
    for j = 1 : nelements(base), i = 1 : 4
        x_distributed[i, j] = x_base[base.elements[j][i]]
    end

    for P in implicit.reference.interops
        @show size(P) size(x_distributed)
        x_distributed = P * x_distributed
    end

    # Output matrix is y.
    y_distributed = zeros(nodes_per_element, nelements(base))

    # Apply the operator on each base element
    cell = cell_type(base)
    quadrature = default_quad(cell)
    element_values = ElementValues(cell, quadrature, update_det_J | update_inv_J)
    
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

        # Broadcast the stuff.
    end

    # Construct the full grid (expensive if `inspect_level` is large)
    fine_mesh = construct_full_grid(implicit, inspect_level)

    # Distribute the faces.
    for face in implicit.reference.number

    # Save the full grid
    vtk = vtk_grid("multiplication", fine_mesh) do vtk
        vtk_point_data(vtk, reshape(y_distributed, :), "u")
    end
    
end