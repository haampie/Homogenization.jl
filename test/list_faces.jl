using Rewrite: list_boundary_nodes_edges_faces, Mesh, refine_uniformly, list_interior_nodes
using Base.Test
using StaticArrays

@testset "Listing boundary faces" begin
    nodes = SVector{3,Float64}[(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
    elements = [(1,2,3,4)]

    # A tetrahedron should have 4 faces.
    let mesh = Mesh(nodes, elements)
        n, e, f = list_boundary_nodes_edges_faces(mesh)
        @test length(f.cells) == 4
        @test length(e.cells) == 6
        @test length(n.cells) == 4
    end

    # Two refinements shoud split each face in 16 faces.
    let mesh = refine_uniformly(Mesh(nodes, elements), times = 2)
        interior_nodes = list_interior_nodes(mesh)

        n, e, f = list_boundary_nodes_edges_faces(mesh)
        @test length(f.cells) == 4 * 16
        @test length(e.cells) == 2 * 16 * 3 # 4 * 16 * 3 edges counted twice.
        @test length(n.cells) == sum(1:5) * 4 - 6 * 3 - 2 * 4
    end
end