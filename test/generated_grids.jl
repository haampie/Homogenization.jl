using Rewrite: cube, nnodes, nelements
using Base.Test

@testset "cube" begin
    mesh = cube(20)

    @test all(issorted, mesh.elements)
    @test nnodes(mesh) == 21 * 21 * 21
    @test nelements(mesh) == 6 * 20 * 20 * 20
end