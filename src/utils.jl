"""
    generate_random_elements(n::Int, m::Ti, k::Int) -> Vector{NTuple{k,Ti}}

Return a bogus mesh of `n` elements with `k` points with global node numbered from 1 to m.
"""
generate_random_elements(n::Int, m::Ti, k::Int) where {Ti} = 
    [Tuple(rand(OneTo(m)) for j = 1 : k) for i = 1 : n]