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
function sort_element_nodes!(v::Vector)
    @inbounds for i = eachindex(v)
        v[i] = sort_bitonic(v[i])
    end
    v
end

"""
Sort a vector by its tuple value using counting sort.
"""
function radix_sort!(v::AbstractVector, max::Int, radix::Int, value::Tf = getindex, count = Vector{Int}(max + 1)) where {Tf}
    n = length(v)
    aux = similar(v)

    @inbounds for d = radix : -1 : 1
        # Reset the counter
        for i = 1 : max + 1
            count[i] = 0
        end
        
        # Frequency count
        for i = 1 : n
            count[value(v[i], d) + 1] += 1
        end

        # Cumulate
        for i = 1 : max
            count[i + 1] += count[i]
        end

        # Move
        for i = 1 : n
            aux[count[value(v[i], d)] += 1] = v[i]
        end

        # Copy
        copy!(v, aux)
    end

    v
end

radix_sort!(v::Vector{NTuple{N,Ti}}, max::Int) where {N,Ti<:Integer} = radix_sort!(v, max, N)

function remove_duplicates!(vec::Vector)
    n = length(vec)

    # Can only be unique
    n ≤ 1 && return vec

    # Discard repeated entries
    slow = 1
    @inbounds for fast = 2 : n
        vec[slow] == vec[fast] && continue
        slow += 1
        vec[slow] = vec[fast]
    end

    # Return the resized vector with unique elements
    return resize!(vec, slow)
end


"""
    binary_search(v, x, lo, hi)

Return the index of the first occurence of x in v[lo:hi]
"""
function binary_search(v::AbstractVector, x, lo::Ti, hi::Ti) where {Ti <: Integer}
    lo -= one(Ti)
    hi += one(Ti)
    @inbounds while lo < hi - one(Ti)
        m = (lo + hi) >>> 1
        if v[m] < x
            lo = m
        else
            hi = m
        end
    end
    return hi
end

"""
Sort the nodes in the adjacency list
"""
function sort_edges!(g::SparseGraph)
    @inbounds for i = 1 : length(g.ptr) - 1
        sort!(g.adj, Int(g.ptr[i]), g.ptr[i + 1] - 1, QuickSort, Base.Order.Forward)
    end

    return g
end

"""
Remove non-repeated elements from an array
"""
function remove_singletons!(v::Vector)
    count = 0
    slow = 1
    fast = 1

    @inbounds while fast ≤ length(v)
        value = v[fast]
        count = 0

        # Copy them over while equal
        while fast ≤ length(v) && v[fast] == value
            v[slow] = v[fast]
            slow += 1
            fast += 1
            count += 1
        end

        # If it occurs only once, we may overwrite it
        if count == 1
            slow -= 1
        end
    end

    resize!(v, slow - 1)
end

"""
Given a sorted array lhs and a sorted array rhs, remove all items in rhs from
lhs, modifying lhs in-place.
"""
function left_minus_right!(lhs::Vector, rhs::Vector)
    slow = 1
    fast = 1
    idx = 1

    @inbounds while fast ≤ length(lhs) && idx ≤ length(rhs)
        if lhs[fast] < rhs[idx]
            # Copy stuff while smaller
            lhs[slow] = lhs[fast]
            slow += 1
            fast += 1
        elseif lhs[fast] == rhs[idx]
            # If equal we should not copy (so slow stays put)
            fast += 1
            idx += 1
        else
            # Otherwise lhs[fast] > rhs[idx], so we should increment idx
            idx += 1
        end
    end

    # Copy the tail of lhs.
    @inbounds while fast ≤ length(lhs)
        lhs[slow] = lhs[fast]
        slow += 1
        fast += 1
    end

    resize!(lhs, slow - 1)
end