using Rewrite: Mesh, refine_uniformly, sort_element_nodes!, nnodes, nelements,
               refined_mesh, base_mesh, ImplicitFineGrid, distribute!,
               construct_full_grid
using StaticArrays
using WriteVTK
using Test

@testset "test_interpolation" begin
    total_levels = 6
    nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]
    elements = [(1,2,3,5),(2,3,4,8),(3,5,7,8),(2,5,6,8),(2,3,5,8)]
    coarse_mesh = refine_uniformly(Mesh(nodes, elements), times = 0)
    sort_element_nodes!(coarse_mesh.elements)
    implicit = ImplicitFineGrid(coarse_mesh, total_levels)

    direction = randn(3)

    xs = map(x -> 10.0 + dot(x, direction), nodes)
    ys = zeros(nnodes(refined_mesh(implicit, 1)), nelements(base_mesh(implicit)))

    distribute!(ys, xs, implicit)

    for level = 2 : total_levels
        ys = implicit.reference.interops[level - 1] * ys
        full_grid = construct_full_grid(implicit, level)

        for (node, value) = zip(full_grid.nodes, ys)
            @test 10.0 + dot(direction, node) ≈ value
        end

        vtk_grid("interpolation_test_$level", full_grid) do vtk
            vtk_point_data(vtk, ys, "ys")
        end
    end
end
