module Rewrite

using StaticArrays, WriteVTK

include("grid.jl")

include("sparse_graph.jl")
include("interpolation.jl")
include("sorting_tricks.jl")
include("utils.jl")

include("tri.jl")

include("tri/refine.jl")
include("tet/refine.jl")

include("multilevel_reference.jl")

end