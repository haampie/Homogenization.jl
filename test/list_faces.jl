using Base.Test
using Rewrite

@testset "Remove duplicates from sorted vector" begin
    @test Rewrite.remove_duplicates!([]) == []
    @test Rewrite.remove_duplicates!([2]) == [2]
    @test Rewrite.remove_duplicates!([1, 1, 2, 3, 4, 4, 4, 5]) == [1,2,3,4,5]
end