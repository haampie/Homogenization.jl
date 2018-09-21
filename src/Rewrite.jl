module Homogenization

export hypercube, Tets, Tet, Tris, Tri, refine_uniformly, Mesh

using StaticArrays, WriteVTK
using SparseArrays, LinearAlgebra, Random, SuiteSparse

export Mesh, refine_uniformly

include("grid.jl")

include("sparse_graph.jl")
include("interpolation.jl")
include("sorting_tricks.jl")
include("utils.jl")

include("tri/refine.jl")
include("tet/refine.jl")

include("tet/generate_grid.jl")
include("tri/generate_grid.jl")

include("cell_values.jl")

include("multilevel_reference.jl")
include("interface.jl")
include("assembly.jl")
include("implicit_fine_grid.jl")
include("build_local_operators.jl")
include("multigrid.jl")
include("apply_local_operators.jl")

include("examples/checkercube.jl")
include("../tools/generate_st1_field.jl")
include("fast_mv_product.jl")
include("examples/large_matrix_free_mg.jl")

end
