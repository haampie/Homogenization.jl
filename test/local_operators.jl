using StaticArrays, WriteVTK
using Homogenization: refined_element, build_local_diffusion_operators, Tets, Tris, Mesh, 
               Tris64, Tets64, affine_map, 
               refine_uniformly, edge_graph, sort_element_nodes!, nodes_on_ref_faces,
               nodes_on_ref_edges, interfaces, face_to_elements, edge_to_elements,
               nelements, nnodes, nodes_per_face_interior, nodes_per_edge_interior, 
               get_reference_normals, Tet, Tri, node_to_elements, ImplicitFineGrid,
               construct_full_grid, ZeroDirichletConstraint, list_boundary_nodes_edges_faces,
               refined_mesh, apply_constraint!, cell_type, default_quad, ElementValues,
               update_det_J, update_inv_J, reinit!, get_inv_jac, get_det_jac, distribute!, 
               broadcast_interfaces!, LevelState, SimpleDiffusion, base_mesh, vcycle!,
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

function test_multigrid(T::Type{<:ElementType} = Tet; iterations = 25, save = false, save_level = 1, refine = 4, total_levels = 4)

    coarse_mesh = refine_uniformly(simple_mesh(T), times = refine)

    ωs = if T <: Tet
        [1.1, 1.8, 3.2, 5.5, 10.1, 18.6, 32.0, 57.0, 103.0]
    else
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    end

    # Factorize the coarse grid operator.
    println("Building the coarse grid operator")
    sort_element_nodes!(coarse_mesh.elements)
    interior = list_interior_nodes(coarse_mesh)
    Ac = assemble_matrix(coarse_mesh, dot)
    F = factorize(Ac[interior, interior])
    base_level = BaseLevel(Float64, F, nnodes(coarse_mesh), interior)

    # Build a multilevel grid
    println("Building the implicit fine grid stuff")
    implicit = ImplicitFineGrid(coarse_mesh, total_levels)
    println(implicit)
    
    # Dirichlet condition
    println("Setting up Dirichlet b.c.")
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
    level_operators = map(build_local_diffusion_operators(implicit.reference)) do op
        SimpleDiffusion(op, constraint)
    end

    # x is initially random with values matching on the interfaces and 0 on the boundary
    finest_level = level_states[total_levels]
    rand!(finest_level.x)
    broadcast_interfaces!(finest_level.x, implicit, total_levels)
    apply_constraint!(finest_level.x, total_levels, constraint, implicit)

    # b is just ones and matching on the interfaces.
    local_rhs!(finest_level.b, implicit)


    tmpgrid = construct_full_grid(implicit, save_level)
    pvd = paraview_collection("multigrid_steps")

    # Do a v-cycle :tada:
    println("Starting the v-cycles; note that the residual norm lags one step ",
            "behind to avoid another expensive mv-product!")
    residuals = Float64[]
    @time for i = 1 : iterations
        vcycle!(implicit, base_level, level_operators, level_states, ωs, total_levels)

        if save
            let
                println("Saving to step_$(lpad(i,3,0)).vtu")
                vtk = vtk_grid("step_$(lpad(i,3,0))", tmpgrid)
                n = nnodes(refined_mesh(implicit, save_level))
                vtk_point_data(vtk, level_states[end].r[1 : n, :], "r")
                vtk_point_data(vtk, level_states[end].x[1 : n, :], "x")
                vtk_save(vtk)
                collection_add_timestep(pvd, vtk, float(i))
            end
        end

        zero_out_all_but_one!(finest_level.r, implicit, total_levels)        
        push!(residuals, vecnorm(finest_level.r))
        @show last(residuals)
    end

    if save
        vtk_save(pvd)
    end

    return residuals
end