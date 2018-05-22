struct Mesh{dim,N,Tv,Ti}
    nodes::Vector{SVector{dim,Tv}}
    elements::Vector{NTuple{N,Ti}}
end

const Tris{Tv,Ti} = Mesh{2,3,Tv,Ti}
const Tets{Tv,Ti} = Mesh{3,4,Tv,Ti}
const Tris64 = Tris{Float64,Int64}
const Tets64 = Tets{Float64,Int64}

function refine_uniformly(m::Mesh; times::Int = 1)
    for i = 1 : times
        m = refine_uniformly(m, edge_graph(m))
    end
    m
end

"""
Returns the affine map from the reference element to the given element.
"""
function affine_map(m::Tris{Tv,Ti}, el::NTuple{3,Ti}) where {Tv,Ti}
    @inbounds begin
        p1 = m.nodes[el[1]]
        p2 = m.nodes[el[2]]
        p3 = m.nodes[el[3]]
        return [p2 - p1 p3 - p1], p1
    end
end

function affine_map(m::Tets{Tv,Ti}, el::NTuple{4,Ti}) where {Tv,Ti}
    @inbounds begin
        p1, p2, p3, p4 = m.nodes[el[1]], m.nodes[el[2]], m.nodes[el[3]], m.nodes[el[4]]
        return [p2 - p1 p3 - p1 p4 - p1], p1
    end
end