module Rewrite

using StaticArrays

include("sorting_tricks.jl")
include("utils.jl")

function my_grid()
    nodes = SVector{2,Float64}[
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (2.0, 1.0)
    ]

    cells = [(1, 2, 4),(2, 3, 4),(3, 4, 5)]
end

end