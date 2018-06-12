using StaticArrays, WriteVTK
using Rewrite: refined_element, build_local_operators, Tets, Tris, Mesh, Tris64, Tets64, affine_map, 
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, interfaces, face_to_elements, edge_to_elements,
               nelements, nnodes, nodes_per_face_interior, nodes_per_edge_interior, 
               get_reference_normals, Tet, Tri, node_to_elements, ImplicitFineGrid,
               construct_full_grid, ZeroDirichletConstraint, list_boundary_nodes_edges_faces,
               refined_mesh, apply_constraint!, cell_type, default_quad, ElementValues,
               update_det_J, update_inv_J, reinit!, get_inv_jac, get_det_jac, distribute!, 
               broadcast_interfaces!, LevelState, LevelOperator, base_mesh, vcycle!,
               list_interior_nodes, assemble_matrix, BaseLevel, zero_out_all_but_one!,
               local_rhs!, assemble_vector, local_residual!, ElementType

function simple_mesh(::Type{<:Tet})
    # Unit cube
    nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]

    # Perturb things a bit.
    map!(x -> x .+ randn(3) / 20, nodes, nodes)
    
    # Split in tetrahedra
    elements = [(1,2,3,5),(2,3,4,8),(3,5,7,8),(2,5,6,8),(2,3,5,8)]

    Mesh(nodes, elements)
end

function simple_mesh(::Type{<:Tri})
    # Unit square
    nodes = SVector{2,Float64}[(0,0),(1,0),(0,1),(1,1)]

    # Perturb things a bit.
    map!(x -> x .+ randn(2) / 20, nodes, nodes)
    
    # Split in tets
    elements = [(1,2,3),(2,3,4)]

    Mesh(nodes, elements)
end

function test_matrix_vector_product(T::Type{<:ElementType} = Tet, total_levels = 2)
    # Build a multilevel grid
    implicit = ImplicitFineGrid(simple_mesh(T), total_levels)

    # Operators for each level
    ops = build_local_operators(implicit.reference)

    # Verify things on this grid
    finest = refined_mesh(implicit, total_levels)
    nodes_per_element = nnodes(refined_mesh(implicit, total_levels))

    x_base = map(x -> 1 - sum(x), implicit.base.nodes)

    x_distributed = zeros(nnodes(refined_mesh(implicit, 1)), nelements(implicit.base))
    
    # Distribute the unknowns
    distribute!(x_distributed, x_base, implicit)

    # Interpolate to finer levels
    for P in implicit.reference.interops
        x_distributed = P * x_distributed
    end

    # Output matrix is y.
    y_distributed = zeros(nodes_per_element, nelements(implicit.base))

    # Apply the operator on each base element (y ← A * x)
    A_mul_B!(1.0, implicit.base, ops[total_levels], x_distributed, y_distributed)

    @show implicit.reference.numbering[total_levels].edges_interior

    # Accumulate the values along the interfaces and store them locally.
    broadcast_interfaces!(y_distributed, implicit, total_levels)

    # Construct the full grid
    fine_mesh = construct_full_grid(implicit, total_levels)

    # Save the full grid
    vtk = vtk_grid("multiplication", fine_mesh) do vtk
        vtk_point_data(vtk, reshape(x_distributed, :), "x")
        vtk_point_data(vtk, reshape(y_distributed, :), "A * u")
    end
end

function test_multigrid(total_levels = 2, iterations = 25, save_base_level = false)
    # Unit cube
    nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]
    elements = [(1,2,3,5),(2,3,4,8),(3,5,7,8),(2,5,6,8),(2,3,5,8)]

    # Unit square
    # nodes = SVector{2,Float64}[(0,0),(1,0),(0,1),(1,1)]
    # elements = [(1,2,3),(2,3,4)]

    # Factorize the coarse grid operator.
    coarse_mesh = refine_uniformly(Mesh(nodes, elements), times = 4)
    println("Building the coarse grid operator")
    sort_element_nodes!(coarse_mesh.elements)
    interior = list_interior_nodes(coarse_mesh)
    Ac = assemble_matrix(coarse_mesh, dot)
    F = factorize(Ac[interior, interior])

    base_level = BaseLevel(Float64, F, nnodes(coarse_mesh), interior)

    # Build a multilevel grid
    println("Building the implicit fine grid stuff")
    implicit = ImplicitFineGrid(coarse_mesh, total_levels)

    @show implicit
    
    # Dirichlet condition
    nodes, edges, faces = list_boundary_nodes_edges_faces(implicit.base)
    constraint = ZeroDirichletConstraint(nodes, edges, faces)

    # Allocate the x's, r's and b's.
    println("Allocating some x's, b's and r's")
    level_states = map(1 : total_levels) do i
        mesh = refined_mesh(implicit, i)
        LevelState(nelements(base_mesh(implicit)), nnodes(mesh), Float64)
    end

    # Build the operators
    println("Building the local operators")
    level_operators = map(build_local_operators(implicit.reference)) do op
        LevelOperator(op, constraint)
    end

    ## BUILD FULL A
    # let 
    #     finegrid = refine_uniformly(coarse_mesh, times = total_levels - 1)
    #     sort_element_nodes!(finegrid.element)
    #     interiornodes = list_interior_nodes(finegrid)
    #     fullA = assemble_matrix(finegrid, dot)[interiornodes,interiornodes]
    #     fullb = assemble_vector(finegrid, identity)[interiornodes]
    #     fullx = zeros(nnodes(finegrid))
    #     fullx[interiornodes] .= fullA \ fullb

    #     vtk_grid("mg_0", finegrid) do vtk
    #         vtk_point_data(vtk, fullx, "x")
    #     end
    # end
    ## END BUILD FULL A

    # x is initially random with values matching on the interfaces and 0 on the boundary
    finest_level = level_states[total_levels]

    rand!(finest_level.x)
    zero_out_all_but_one!(finest_level.x, implicit, total_levels)
    broadcast_interfaces!(finest_level.x, implicit, total_levels)
    apply_constraint!(finest_level.x, total_levels, constraint, implicit)

    # b is just ones and matching on the interfaces.
    local_rhs!(finest_level.b, implicit)

    ωs = [1.1, 1.8, 3.2, 5.5, 10.1, 18.6, 32.0] ./ 0.9
    # ωs = [0.2, 0.2, 0.2, 0.2, 0.2]

    tmpgrid = construct_full_grid(implicit, 1)
    pvd = paraview_collection("my_pvd_file")

    # Do a v-cycle :tada:
    println("Starting the v-cycles; note that the residual norm lags one step behind to avoid another expensive mv-product!")
    residuals = Float64[]
    for i = 1 : iterations
        vcycle!(implicit, base_level, level_operators, level_states, ωs, total_levels)

        if save_base_level
            let
                println("Saving to step_$(lpad(i,3,0)).vtu")
                vtk = vtk_grid("step_$(lpad(i,3,0))", tmpgrid)
                n = nnodes(refined_mesh(implicit, 1))
                vtk_point_data(vtk, level_states[end].r[1 : n, :], "r")
                vtk_point_data(vtk, level_states[end].x[1 : n, :], "x")
                vtk_save(vtk)
                collection_add_timestep(pvd, vtk, float(i))
            end
        end

        # local_residual!(implicit, level_operators[total_levels], finest_level, total_levels)
        # broadcast_interfaces!(finest_level.r, implicit, total_levels)
        zero_out_all_but_one!(finest_level.r, implicit, total_levels)        
        push!(residuals, vecnorm(finest_level.r))
        @show last(residuals)
    end

    if save_base_level
        vtk_save(pvd)
    end

    return residuals
end