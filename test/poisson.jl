using Rewrite: assemble_matrix_and_rhs, refine_uniformly, Mesh, list_interior_nodes, nnodes, nelements
using StaticArrays
using WriteVTK

function poisson()
    nodes = SVector{3,Float64}[(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
    elements = [(1,2,3,4)]
    # nodes = SVector{2,Float64}[(0,0), (1,0), (0,1)]
    # elements = [(1,2,3)]
    mesh = refine_uniformly(Mesh(nodes,elements), times = 8)

    @show nnodes(mesh) nelements(mesh)

    interior = list_interior_nodes(mesh)

    println("Assembling")
    A, b = assemble_matrix_and_rhs(mesh)

    x = zeros(nnodes(mesh))
    # x[interior] .= 1.0

    println("Solving")
    x[interior] .= A[interior,interior] \ b[interior]

    println("Saving")
    vtk_grid("poisson", mesh) do vtk
        vtk_point_data(vtk, x, "x")
        # vtk_point_data(vtk, b, "b")
    end
end