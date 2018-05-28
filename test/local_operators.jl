using Rewrite: refined_element, build_local_operators, Tets, Tris, Mesh, Tris64, Tets64, affine_map, 
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, interfaces, face_to_elements, edge_to_elements,
               nelements, nnodes, nodes_per_face_interior, nodes_per_edge_interior, 
               get_reference_normals, Tet, node_to_elements, ImplicitFineGrid,
               construct_full_grid, ZeroDirichletConstraint, list_boundary_faces,
               refined_mesh, apply_constraint!, cell_type, default_quad, ElementValues,
               update_det_J, update_inv_J, reinit!, get_inv_jac, get_detjac, distribute!, broadcast_interfaces!
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

    # Perturb things a bit.
    map!(x -> x .+ randn(3)/20, nodes, nodes)
    
    # Split in tetrahedra
    elements = [
        (1,2,3,5),
        (2,3,4,8),
        (3,5,7,8),
        (2,5,6,8),
        (2,3,5,8)
    ]

    # Build a multilevel grid
    implicit = ImplicitFineGrid(Mesh(nodes, elements), total_levels)

    # Operators for each level
    ops = build_local_operators(implicit.reference)

    # Select the finest level ops.
    ∫ϕₓᵢϕₓⱼ_finest = ops[inspect_level]

    # Verify things on this grid
    finest = refined_mesh(implicit, inspect_level)
    
    # Find the number of nodes per element at level `inspect_level`
    nodes_per_element = nnodes(refined_mesh(implicit, inspect_level))

    x_base = map(x -> 1 - sum(x), implicit.base.nodes)

    x_distributed = zeros(4, nelements(implicit.base))
    
    # Distribute the unknowns
    distribute!(x_distributed, x_base, implicit)

    # Interpolate to finer levels
    for P in implicit.reference.interops
        x_distributed = P * x_distributed
    end

    # Output matrix is y.
    y_distributed = zeros(nodes_per_element, nelements(implicit.base))

    # Apply the operator on each base element
    A_mul_B!(1.0, implicit.base, ∫ϕₓᵢϕₓⱼ_finest, x_distributed, 0.0, y_distributed)

    # Accumulate the values along the interfaces and store them locally.
    broadcast_interfaces!(y_distributed, implicit, inspect_level)

    # Construct the full grid (expensive if `inspect_level` is large)
    fine_mesh = construct_full_grid(implicit, inspect_level)

    # Save the full grid
    vtk = vtk_grid("multiplication", fine_mesh) do vtk
        vtk_point_data(vtk, reshape(x_distributed, :), "x")
        vtk_point_data(vtk, reshape(y_distributed, :), "A * u")
    end
    
end