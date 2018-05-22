using Base.Test
using Rewrite: remove_duplicates!, remove_singletons!

@testset "Remove duplicates from sorted vector" begin
    @test remove_duplicates!([]) == []
    @test remove_duplicates!([2]) == [2]
    @test remove_duplicates!([1,1,2,3,4,4,4,5]) == [1,2,3,4,5]
end

@testset "Remove singletons from sorted vector" begin
    @test remove_singletons!([]) == []
    @test remove_singletons!([2]) == []
    @test remove_singletons!([2,3]) == []
    @test remove_singletons!([1,1]) == [1,1]
    @test remove_singletons!([1,2,2,4,4,4,5]) == [2,2,4,4,4]
    @test remove_singletons!([1,1,2,3,4,4,4,5]) == [1,1,4,4,4]
end