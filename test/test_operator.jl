using Base.Test
using StaticArrays, WriteVTK
using Rewrite: build_local_diffusion_operators, Mesh, refine_uniformly, nelements, nnodes, 
               ImplicitFineGrid, construct_full_grid, refined_mesh, 
               broadcast_interfaces!, assemble_matrix

@testset "Verify the matrix-vector product is working" begin
    levels = 5
    # Unit cube
    nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]
    elements = [(1,2,3,5),(2,3,4,8),(3,5,7,8),(2,5,6,8),(2,3,5,8)]
    base = refine_uniformly(Mesh(nodes, elements), times = 1)
    sort_element_nodes!(base.elements)

    # Local mesh, matrices and state vectors
    implicit = ImplicitFineGrid(base, levels)
    local_fine = refined_mesh(implicit, levels)
    local_x = rand(nnodes(local_fine), nelements(implicit.base))
    broadcast_interfaces!(local_x, implicit, levels)
    local_y = zeros(nnodes(local_fine), nelements(implicit.base))
    local_A = build_local_diffusion_operators(implicit.reference)[levels]

    # Construct fine grid with repeated nodes
    fine_grid_repeated_nodes = construct_full_grid(implicit, levels)

    # Global mesh, matrices and state vectors
    total_fine = refine_uniformly(implicit.base, times = levels - 1)
    total_A = assemble_matrix(total_fine, dot)
    total_x = zeros(nnodes(total_fine))
    total_y = zeros(nnodes(total_fine))

    # Create a dumb mapping from fine_grid_repeated_nodes -> total_fine
    mapping = Vector{Int}(nnodes(fine_grid_repeated_nodes))
    for (i, node) in enumerate(fine_grid_repeated_nodes.nodes)
        match = false
        for (j, other_node) in enumerate(total_fine.nodes)
            if node ≈ other_node
                mapping[i] = j
                match = true
                break
            end
        end
        # Just test that each node matches another
        @test match
    end

    # Copy over `local_x` to `total_x`
    total_x[mapping] = local_x[:]

    # Do the mv product
    A_mul_B!(1.0, implicit.base, local_A, local_x, local_y)
    broadcast_interfaces!(local_y, implicit, levels)
    A_mul_B!(total_y, total_A, total_x)

    # vtk_grid("stuff_1", total_fine) do vtk
    #     vtk_point_data(vtk, total_x, "x")
    #     vtk_point_data(vtk, total_y, "y")
    # end

    # vtk_grid("stuff_2", fine_grid_repeated_nodes) do vtk
    #     vtk_point_data(vtk, local_x[:], "x")
    #     vtk_point_data(vtk, local_y[:], "y")
    # end

    # Verify the element-wise error is small
    @test maximum(abs.(total_y[mapping] .- local_y[:])) ≤ 20 * eps()
end