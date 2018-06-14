module Rewrite

using StaticArrays, WriteVTK

export Mesh, refine_uniformly

include("grid.jl")

include("sparse_graph.jl")
include("interpolation.jl")
include("sorting_tricks.jl")
include("utils.jl")

include("tri/refine.jl")
include("tet/generate_grid.jl")
include("tet/refine.jl")

include("cell_values.jl")

include("multilevel_reference.jl")
include("interface.jl")
include("assembly.jl")
include("implicit_fine_grid.jl")
include("build_local_operators.jl")
include("multigrid.jl")

include("examples/checkercube.jl")

end