using Rewrite: hypercube, Tet, nnodes, nelements
using Test

@testset "cube" begin
    mesh = hypercube(Tet{Float64}, 20)

    @test all(issorted, mesh.elements)
    @test nnodes(mesh) == 21 * 21 * 21
    @test nelements(mesh) == 6 * 20 * 20 * 20
end
