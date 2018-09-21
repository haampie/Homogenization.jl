using Test

using Homogenization: radix_sort!, generate_random_elements, sort_element_nodes!

@testset "Counting sort" begin
    v = generate_random_elements(100, 25, 4)
    sort_element_nodes!(v)
    radix_sort!(v, 25)

    @test issorted(v)
end
