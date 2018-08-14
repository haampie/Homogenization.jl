using Test
using Rewrite: sort_bitonic

@testset "Bitonic sorting tricks" begin

    @test issorted(sort_bitonic((1, 2, 3, 4)))
    @test issorted(sort_bitonic((1, 2, 4, 3)))
    @test issorted(sort_bitonic((1, 3, 2, 4)))
    @test issorted(sort_bitonic((1, 3, 4, 2)))
    @test issorted(sort_bitonic((1, 4, 2, 3)))
    @test issorted(sort_bitonic((1, 4, 3, 2)))
    @test issorted(sort_bitonic((2, 1, 3, 4)))
    @test issorted(sort_bitonic((2, 1, 4, 3)))
    @test issorted(sort_bitonic((2, 3, 4, 1)))
    @test issorted(sort_bitonic((2, 3, 4, 1)))
    @test issorted(sort_bitonic((2, 4, 1, 3)))
    @test issorted(sort_bitonic((2, 4, 3, 1)))
    @test issorted(sort_bitonic((3, 1, 2, 4)))
    @test issorted(sort_bitonic((3, 1, 4, 2)))
    @test issorted(sort_bitonic((3, 2, 4, 1)))
    @test issorted(sort_bitonic((3, 2, 4, 1)))
    @test issorted(sort_bitonic((3, 4, 1, 2)))
    @test issorted(sort_bitonic((3, 4, 2, 1)))
    @test issorted(sort_bitonic((4, 1, 2, 3)))
    @test issorted(sort_bitonic((4, 1, 3, 2)))
    @test issorted(sort_bitonic((4, 2, 3, 1)))
    @test issorted(sort_bitonic((4, 2, 3, 1)))
    @test issorted(sort_bitonic((4, 3, 1, 2)))
    @test issorted(sort_bitonic((4, 3, 2, 1)))

    @test issorted(sort_bitonic((1, 2, 3)))
    @test issorted(sort_bitonic((1, 3, 2)))
    @test issorted(sort_bitonic((2, 1, 3)))
    @test issorted(sort_bitonic((2, 3, 1)))
    @test issorted(sort_bitonic((3, 1, 2)))
    @test issorted(sort_bitonic((3, 2, 1)))

    @test issorted(sort_bitonic((1, 2)))
    @test issorted(sort_bitonic((2, 1)))
end
