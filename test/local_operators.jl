using StaticArrays, WriteVTK
using Rewrite: refined_element, build_local_operators, Tets, Tris, Mesh, Tris64, Tets64, affine_map, 
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, interfaces, face_to_elements, edge_to_elements,
               nelements, nnodes, nodes_per_face_interior, nodes_per_edge_interior, 
               get_reference_normals, Tet, node_to_elements, ImplicitFineGrid,
               construct_full_grid, ZeroDirichletConstraint, list_boundary_faces,
               refined_mesh, apply_constraint!, cell_type, default_quad, ElementValues,
               update_det_J, update_inv_J, reinit!, get_inv_jac, get_detjac, distribute!, 
               broadcast_interfaces!, LevelState, LevelOperator, base_mesh, vcycle!,
               list_interior_nodes, assemble_matrix, BaseLevel

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
    map!(x -> x .+ randn(3) / 20, nodes, nodes)
    
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
    A_mul_B!(1.0, implicit.base, ops[inspect_level], x_distributed, 0.0, y_distributed)

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

function solve_things(total_levels = 5)
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
    map!(x -> x .+ randn(3) / 50, nodes, nodes)
    
    # Split in tetrahedra
    elements = [
        (1,2,3,5),
        (2,3,4,8),
        (3,5,7,8),
        (2,5,6,8),
        (2,3,5,8)
    ]

    # Factorize the coarse grid operator.
    coarse_mesh = refine_uniformly(Mesh(nodes, elements), times = 3)
    interior = list_interior_nodes(coarse_mesh)
    Ac = assemble_matrix(coarse_mesh, dot)
    F = factorize(Ac[interior, interior])

    base_level = BaseLevel(Float64, F, nnodes(coarse_mesh), interior)

    # Build a multilevel grid
    implicit = ImplicitFineGrid(coarse_mesh, total_levels)

    @show implicit
    
    # Dirichlet condition
    constraint = ZeroDirichletConstraint(list_boundary_faces(implicit.base))

    # Allocate the x's, r's and b's.
    level_states = map(1 : total_levels) do i
        mesh = refined_mesh(implicit, i)
        LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)
    end

    # Build the operators
    level_operators = map(build_local_operators(implicit.reference)) do op
        LevelOperator(op, constraint)
    end

    # Set up the problem Ax = b.
    
    # x is initially random with values matching on the interfaces and 0 on the boundary
    randn!(level_states[end].x)
    broadcast_interfaces!(level_states[total_levels].x, implicit, total_levels)
    apply_constraint!(level_states[total_levels].x, total_levels, constraint, implicit)

    # b is just ones and matching on the interfaces.
    fill!(level_states[end].b, 1.0)

    ωs = fill(0.9, total_levels)

    # Do a v-cycle :tada:
    vcycle!(implicit, base_level, level_operators, level_states, ωs, total_levels)
    
end