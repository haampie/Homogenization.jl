using Homogenization: compress, EdgeToEl, ElementId
using Test

@testset "Sparse cell to element" begin
    let sparse_map = compress([
            EdgeToEl{Int}((1, 2), ElementId(1, 2)),
            EdgeToEl{Int}((1, 2), ElementId(2, 3)),
            EdgeToEl{Int}((2, 3), ElementId(3, 4)),
            EdgeToEl{Int}((2, 3), ElementId(5, 6))
        ])

        @test sparse_map.offset == [1, 3, 5]
        @test sparse_map.cells == [(1,2), (2,3)]
        @test sparse_map.values == [ElementId(1, 2), ElementId(2, 3), ElementId(3, 4), ElementId(5, 6)]
    end


    let sparse_map = compress([
            EdgeToEl{Int}((1, 2), ElementId(1, 2)),
            EdgeToEl{Int}((2, 3), ElementId(3, 4)),
        ])

        @test sparse_map.offset == [1, 2, 3]
        @test sparse_map.cells == [(1,2), (2,3)]
        @test sparse_map.values == [ElementId(1, 2), ElementId(3, 4)]
    end
end
