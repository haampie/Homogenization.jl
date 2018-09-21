using Homogenization: Mesh, refine_uniformly, list_boundary_edges, compress
using StaticArrays
using Test

@testset "Find boundary stuff" begin
    # Unit cube
    nodes = SVector{3,Float64}[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]

    # Split in tetrahedra
    elements = [(1,2,3,5),(2,3,4,8),(3,5,7,8),(2,5,6,8),(2,3,5,8)]

    mesh = refine_uniformly(Mesh(nodes, elements), times = 5)

    boundary_edges = list_boundary_edges(mesh)

    @show length(boundary_edges.offset)
    @show length(boundary_edges.values)
    @show length(boundary_edges.cells)

    # @test all(x -> issorted(x.nodes), boundary_edges)
    # @test issorted(boundary_edges, by = x -> x.nodes)
end
