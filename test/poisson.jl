using Homogenization: assemble_matrix, assemble_vector, refine_uniformly, 
                      Mesh, list_interior_nodes, nnodes, nelements, sort_element_nodes!,
                      cube, list_faces, remove_repeated_pairs!, remove_duplicates!,
                      radix_sort!, complement

using StaticArrays
using WriteVTK

function poisson()
    mesh = cube(88)
    interior = list_interior_nodes(mesh)

    println("Assembling")
    @show nnodes(mesh)
    @show nelements(mesh)

    A = assemble_matrix(mesh, dot)
    b = assemble_vector(mesh, identity)
    x = zeros(nnodes(mesh))

    println("Solving")
    @time x[interior] .= A[interior,interior] \ b[interior]

    println("Saving")
    vtk_grid("poisson", mesh) do vtk
        vtk_point_data(vtk, x, "x")
        vtk_point_data(vtk, b, "b")
    end
end
