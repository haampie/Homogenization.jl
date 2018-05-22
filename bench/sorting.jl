using BenchmarkTools
using Base: OneTo
using Rewrite: sort_element_nodes!, radix_sort!

function just_the_elements()
    return @benchmark sort_element_nodes!(v) setup = (v = generate_random_elements(1_000, 69, 4))
end

function bench_radix_sort(n = 1_000_000, m = 500_000, k = 2)
    v = sort_element_nodes!(generate_random_elements(n, m, k))

    a = @benchmark sort!(w) setup = (w = copy($v))
    b = @benchmark radix_sort!(w, $m) setup = (w = copy($v))

    a, b
end