@inline sort_bitonic(nodes::NTuple{1,<:Integer}) = nodes

@inline function sort_bitonic(nodes::NTuple{2,<:Integer})
    @inbounds a, b = nodes
    min(a, b), max(a, b)
end

@inline function sort_bitonic(nodes::NTuple{3,<:Integer})
    @inbounds a, b, c = nodes
    a, b = min(a, b), max(a, b)
    b, c = min(b, c), max(b, c)
    a, b = min(a, b), max(a, b)
    a, b, c
end

@inline function sort_bitonic(nodes::NTuple{4,<:Integer})
    @inbounds a, b, c, d = nodes

    a, b = min(a, b), max(a, b)
    c, d = min(c, d), max(c, d)

    a, c = min(a, c), max(a, c)
    b, d = min(b, d), max(b, d)

    a, d = min(a, d), max(a, d)
    b, c = min(b, c), max(b, c)
    
    a, b, c, d
end

"""
Sort each small tuple of a large vector.
"""
function sort_element_nodes!(v::Vector{NTuple{N,Ti}}) where {N,Ti<:Integer}
    @inbounds for i = eachindex(v)
        v[i] = sort_bitonic(v[i])
    end
    v
end

"""
Sort a vector by its tuple value using counting sort.
"""
function counting_sort!(v::Vector{NTuple{N,Ti}}, max::Int) where {N,Ti<:Integer}
    n = length(v)
    aux = similar(v)
    count = Vector{Ti}(max + 1)

    @inbounds for d = N : -1 : 1
        # Reset the counter
        for i = 1 : max + 1
            count[i] = 0
        end
        
        # Frequency count
        for i = 1 : n
            count[v[i][d] + 1] += 1
        end

        # Cumulate
        for i = 1 : max
            count[i+1] += count[i]
        end

        # Move
        for i = 1 : n
            aux[count[v[i][d]] += 1] = v[i];
        end

        # Copy
        v, aux = aux, v
    end

    return v
end
